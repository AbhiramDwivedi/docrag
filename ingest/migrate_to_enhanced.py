#!/usr/bin/env python3
"""Migration utility to upgrade from Phase 1 to Phase 2 database schema.

This script safely migrates existing vector store databases to the enhanced
schema that supports rich metadata for Phase 2 features.
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import get_settings
from ingest.enhanced_vector_store import EnhancedVectorStore
from ingest.embed import get_embed_model


def backup_existing_database(db_path: Path) -> Path:
    """Create a backup of the existing database.
    
    Args:
        db_path: Path to the database to backup
        
    Returns:
        Path to the backup file
    """
    if not db_path.exists():
        print(f"No existing database found at {db_path}")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_suffix(f".backup_{timestamp}.db")
    
    print(f"Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    
    return backup_path


def migrate_database():
    """Perform the migration from Phase 1 to Phase 2 schema."""
    settings = get_settings()
    
    print("🔄 Starting Phase 2 database migration...")
    print(f"Database path: {settings.db_path}")
    print(f"Vector index path: {settings.vector_path}")
    
    # Create backup
    backup_path = backup_existing_database(settings.db_path)
    if backup_path:
        print(f"✅ Backup created: {backup_path}")
    
    # Initialize vector store (this will create new tables)
    embed_model = get_embed_model()
    embed_dim = len(embed_model.encode("test"))
    
    print(f"Initializing vector store (dim={embed_dim})...")
    vector_store = EnhancedVectorStore(
        index_path=settings.vector_path,
        db_path=settings.db_path, 
        dim=embed_dim
    )
    
    # Run migration
    print("🔄 Migrating existing data to new schema...")
    vector_store.migrate_from_basic_store()
    
    # Get statistics
    stats = vector_store.get_file_statistics()
    
    print("\n✅ Migration completed successfully!")
    print(f"📊 Collection statistics:")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Total emails: {stats['total_emails']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Total size: {stats['total_size_bytes']:,} bytes")
    
    if stats['file_types']:
        print(f"   File types:")
        for file_type in stats['file_types'][:5]:
            print(f"     - {file_type['type']}: {file_type['count']} files")
    
    if backup_path:
        print(f"\n💾 Original database backed up to: {backup_path}")
    
    print("\n🎉 Phase 2 enhanced metadata is now available!")
    print("   You can now use advanced queries like:")
    print("   - 'Find emails from john@example.com last week'")
    print("   - 'Show me the newest Excel files'") 
    print("   - 'List PDF files larger than 1MB'")


def main():
    """Main migration entry point."""
    try:
        migrate_database()
    except KeyboardInterrupt:
        print("\n❌ Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("Your original database has been preserved.")
        sys.exit(1)


if __name__ == "__main__":
    main()