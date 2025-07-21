"""CLI for full or incremental ingest."""
import argparse, hashlib, sys
from pathlib import Path

# Add parent directory to path so we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.extractor import extract_text
from ingest.chunker import chunk_text
from ingest.embed import embed_texts
from ingest.vector_store import VectorStore
from config.config import settings
from rich.progress import track


def process_file(path: Path, store: VectorStore):
    file_id = hashlib.sha1(str(path).encode()).hexdigest()[:10]
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
    args = parser.parse_args()
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
