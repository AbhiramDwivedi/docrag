from ingestion.storage.enhanced_vector_store import EnhancedVectorStore
from pathlib import Path

# Test enhanced vector store migration
try:
    print("Testing Enhanced Vector Store migration...")
    
    # Load enhanced vector store with existing data
    enhanced_store = EnhancedVectorStore.load(
        Path("data/vector.index"),
        Path("data/enhanced_vector_store.db"),
        dim=384
    )
    
    print("✅ Enhanced vector store loaded successfully")
    
    # Run migration from basic store
    print("Running migration from basic store...")
    enhanced_store.migrate_from_basic_store()
    
    # Check new tables were created
    cursor = enhanced_store.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print('Enhanced tables:', [t[0] for t in tables])
    
    # Check migrated file count
    cursor.execute("SELECT COUNT(*) FROM files")
    file_count = cursor.fetchone()[0]
    print(f'Migrated files: {file_count}')
    
    print("✅ Migration completed successfully")
    
except Exception as e:
    print(f"❌ Migration failed: {e}")
    import traceback
    traceback.print_exc()
