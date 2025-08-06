"""Document ingestion pipeline for DocQuest."""
import argparse
import sys
from pathlib import Path

# Add backend root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingestion.extractors import extract_text
from ingestion.processors.chunker import chunk_text
from ingestion.processors.embedder import embed_texts
from ingestion.storage.vector_store import VectorStore
from shared.config import settings
from shared.utils import get_file_hash
from rich.progress import track


def process_file(path: Path, store: VectorStore):
    file_id = get_file_hash(path)
    units = extract_text(path)
    all_chunks, texts, meta = [], [], []
    for unit_id, text in units:
        chunks = chunk_text(text, file_id, unit_id,
                            settings.chunk_size, settings.overlap)
        for ch in chunks:
            all_chunks.append(ch)
            texts.append(ch['text'])
            meta.append((ch['id'], str(path), unit_id, ch['text'], path.stat().st_mtime, 1))
    if not texts:
        return
    vectors = embed_texts(texts, settings.embed_model)
    store.upsert([c['id'] for c in all_chunks], vectors, meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'incremental'], default='incremental')
    parser.add_argument('--file-type', help='Process only specific file types (e.g., xlsx, pdf, docx)')
    parser.add_argument('--target', help='Process specific file by name (for --all-sheets option)')
    parser.add_argument('--all-sheets', action='store_true', help='Process ALL sheets in Excel files (removes 15-sheet limit)')
    args = parser.parse_args()
    
    # Set all-sheets mode if requested
    if args.all_sheets:
        set_all_sheets_mode(True)
        print("🔄 ALL-SHEETS MODE enabled: Will process all Excel sheets")
    
    store = VectorStore(Path(settings.vector_path), Path(settings.db_path), dim=384)
    
    # Get all files or filter by type
    if args.file_type:
        files = list(Path(settings.sync_root).rglob(f'*.{args.file_type}'))
        if args.file_type in ['xls', 'xlsx']:
            files.extend(list(Path(settings.sync_root).rglob('*.xls')))
            files.extend(list(Path(settings.sync_root).rglob('*.xlsx')))
        print(f"Processing {len(files)} {args.file_type} files...")
    else:
        files = list(Path(settings.sync_root).rglob('*.*'))
    
    # Filter by target filename if specified
    if args.target:
        files = [f for f in files if args.target.lower() in f.name.lower()]
        print(f"Filtered to {len(files)} files matching '{args.target}'")
    
    processed_count = 0
    failed_count = 0
    
    for file in track(files, description='Processing'):
        if '~$' in file.name:
            continue
        try:
            process_file(file, store)
            processed_count += 1
        except Exception as e:
            print(f"❌ Failed to process {file.name}: {e}")
            failed_count += 1
    
    print(f"\n✅ Processing complete: {processed_count} files processed, {failed_count} failed")

if __name__ == '__main__':
    main()
