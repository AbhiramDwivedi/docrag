"""Document ingestion pipeline for DocQuest."""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add backend src to path for absolute imports  
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.extractors import extract_text, set_all_sheets_mode
from ingestion.processors.chunker import chunk_text
from ingestion.processors.embedder import embed_texts
from ingestion.storage.enhanced_vector_store import EnhancedVectorStore
from ingestion.storage.knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder
from shared.config import settings
from shared.utils import get_file_hash
from rich.progress import track


def process_file(path: Path, store: EnhancedVectorStore, kg: KnowledgeGraph):
    file_id = get_file_hash(path)
    units = extract_text(path)
    all_chunks, texts, meta = [], [], []
    
    # Collect all text content for knowledge graph processing
    full_text = ""
    
    for unit_id, text in units:
        full_text += f"\n{text}"  # Accumulate text for entity extraction
        chunks = chunk_text(text, file_id, unit_id,
                            settings.chunk_size, settings.overlap)
        for ch in chunks:
            all_chunks.append(ch)
            texts.append(ch['text'])
            meta.append((ch['id'], str(path), unit_id, ch['text'], path.stat().st_mtime, 1))
    
    if not texts:
        return
    
    # Extract entities and relationships for knowledge graph
    try:
        kg_builder = KnowledgeGraphBuilder(kg)
        entities, relationships = kg_builder.extract_entities_from_text(full_text, str(path))
        for entity in entities:
            kg.add_entity(entity)
        for relationship in relationships:
            kg.add_relationship(relationship)
    except Exception as e:
        # Don't fail the entire pipeline if KG extraction fails
        print(f"⚠️ Knowledge graph extraction failed for {path.name}: {e}")
    
    vectors = embed_texts(texts, settings.embed_model)
    
    # Collect file metadata for enhanced storage
    file_stat = path.stat()
    file_metadata = {
        'file_path': str(path),
        'file_name': path.name,
        'file_extension': path.suffix.lower(),
        'file_size': file_stat.st_size,
        'created_time': file_stat.st_ctime,
        'modified_time': file_stat.st_mtime,
        'accessed_time': file_stat.st_atime,
        'file_type': path.suffix.upper().lstrip('.'),  # PDF, DOCX, etc.
        'chunk_count': len(all_chunks),
        'ingestion_time': datetime.now().timestamp()
    }
    
    # Use enhanced upsert with metadata
    store.upsert_with_metadata([c['id'] for c in all_chunks], vectors, meta, file_metadata)


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
    
    store = EnhancedVectorStore(Path(settings.vector_path), Path(settings.db_path), dim=384)
    
    # Initialize knowledge graph (mandatory)
    kg_path = Path("data/knowledge_graph.db")
    kg = KnowledgeGraph(str(kg_path))
    print("🧠 Knowledge graph initialized")
    
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
            process_file(file, store, kg)
            processed_count += 1
        except Exception as e:
            print(f"❌ Failed to process {file.name}: {e}")
            failed_count += 1
    
    # Print knowledge graph statistics
    try:
        stats = kg.get_statistics()
        print(f"\n🧠 Knowledge Graph Stats:")
        print(f"   📊 Total entities: {stats.get('total_entities', 0)}")
        print(f"   🔗 Total relationships: {stats.get('total_relationships', 0)}")
        entity_types = stats.get('entity_types', {})
        if entity_types:
            print(f"   📋 Entity types: {dict(entity_types)}")
    except Exception as e:
        print(f"⚠️ Failed to get knowledge graph stats: {e}")
    
    print(f"\n✅ Processing complete: {processed_count} files processed, {failed_count} failed")

if __name__ == '__main__':
    main()
