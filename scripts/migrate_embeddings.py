#!/usr/bin/env python3
"""
Embedding Model Migration Script

This script migrates existing embeddings to a new embedding model while preserving
all existing data. It supports progress tracking and can resume from interruptions.

Usage:
    python scripts/migrate_embeddings.py --model intfloat/e5-base-v2 --version 2.0.0
    python scripts/migrate_embeddings.py --resume
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

try:
    import sqlite3
    import numpy as np
    import faiss
    from rich.console import Console
    from rich.progress import Progress, TaskID
    from rich import print as rprint
    
    from shared.config import get_settings
    from ingestion.processors.embedder import embed_texts, get_model_config
    from shared.logging_config import setup_logging
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install the project: pip install -e backend/")
    sys.exit(1)

console = Console()

class EmbeddingMigrator:
    """Handles migration of embeddings to a new model with progress tracking."""
    
    def __init__(self, target_model: str, target_version: str, batch_size: int = 100):
        self.target_model = target_model
        self.target_version = target_version
        self.batch_size = batch_size
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Paths
        self.db_path = self.settings.resolve_storage_path(self.settings.db_path)
        self.vector_path = self.settings.resolve_storage_path(self.settings.vector_path)
        self.backup_dir = Path("data/migration_backup")
        self.progress_file = Path("data/migration_progress.json")
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def create_backup(self) -> None:
        """Create backup of existing data before migration."""
        timestamp = int(time.time())
        backup_db = self.backup_dir / f"docmeta_{timestamp}.db"
        backup_vector = self.backup_dir / f"vector_{timestamp}.index"
        
        rprint(f"📦 Creating backup...")
        
        if self.db_path.exists():
            import shutil
            shutil.copy2(self.db_path, backup_db)
            rprint(f"   ✅ Database backed up to: {backup_db}")
        
        if self.vector_path.exists():
            import shutil
            shutil.copy2(self.vector_path, backup_vector)
            rprint(f"   ✅ Vector index backed up to: {backup_vector}")
        
        # Save backup info
        backup_info = {
            "timestamp": timestamp,
            "original_model": self.settings.embed_model,
            "original_version": getattr(self.settings, 'embed_model_version', '1.0.0'),
            "target_model": self.target_model,
            "target_version": self.target_version,
            "db_backup": str(backup_db),
            "vector_backup": str(backup_vector)
        }
        
        with open(self.backup_dir / f"backup_info_{timestamp}.json", 'w') as f:
            json.dump(backup_info, f, indent=2)
            
    def load_progress(self) -> Dict:
        """Load migration progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "total_chunks": 0,
            "processed_chunks": 0,
            "last_chunk_id": 0,
            "target_model": self.target_model,
            "target_version": self.target_version,
            "started_at": None,
            "completed": False
        }
    
    def save_progress(self, progress: Dict) -> None:
        """Save migration progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def get_chunk_data(self, start_id: int = 0) -> List[Tuple[int, str]]:
        """Get chunk text data from database starting from given ID."""
        if not self.db_path.exists():
            rprint("❌ Database not found. Nothing to migrate.")
            return []
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get chunks with text content
        cursor.execute("""
            SELECT id, content FROM chunks 
            WHERE id > ? AND content IS NOT NULL 
            ORDER BY id
        """, (start_id,))
        
        chunks = cursor.fetchall()
        conn.close()
        
        return chunks
    
    def count_total_chunks(self) -> int:
        """Count total number of chunks to migrate."""
        if not self.db_path.exists():
            return 0
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE content IS NOT NULL")
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def migrate_embeddings(self, resume: bool = False) -> bool:
        """Perform the embedding migration with progress tracking."""
        progress = self.load_progress()
        
        if resume and progress.get("completed"):
            rprint("✅ Migration already completed!")
            return True
            
        if not resume:
            # Fresh migration - create backup and reset progress
            self.create_backup()
            total_chunks = self.count_total_chunks()
            progress = {
                "total_chunks": total_chunks,
                "processed_chunks": 0,
                "last_chunk_id": 0,
                "target_model": self.target_model,
                "target_version": self.target_version,
                "started_at": time.time(),
                "completed": False
            }
            self.save_progress(progress)
            
        if progress["total_chunks"] == 0:
            rprint("📭 No chunks found to migrate.")
            return True
            
        rprint(f"🚀 Migrating to model: {self.target_model}")
        rprint(f"📊 Progress: {progress['processed_chunks']}/{progress['total_chunks']} chunks")
        
        # Initialize new vector index
        model_config = get_model_config(self.target_model)
        rprint(f"🔧 Model config: {model_config}")
        
        # Load first batch to determine embedding dimension
        sample_chunks = self.get_chunk_data(progress["last_chunk_id"])
        if not sample_chunks:
            rprint("✅ No more chunks to process!")
            progress["completed"] = True
            self.save_progress(progress)
            return True
            
        # Get embedding dimension from first sample
        sample_texts = [chunk[1] for chunk in sample_chunks[:1]]
        sample_embeddings = embed_texts(sample_texts, self.target_model, normalize=True, text_type="passage")
        embedding_dim = sample_embeddings.shape[1]
        
        # Create new FAISS index
        new_index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for normalized vectors
        chunk_id_to_index = {}
        
        with Progress() as progress_bar:
            task = progress_bar.add_task("Migrating embeddings...", total=progress["total_chunks"])
            progress_bar.update(task, completed=progress["processed_chunks"])
            
            while True:
                # Get batch of chunks
                chunks = self.get_chunk_data(progress["last_chunk_id"])
                if not chunks:
                    break
                    
                batch = chunks[:self.batch_size]
                if not batch:
                    break
                    
                # Extract texts and generate new embeddings
                chunk_ids = [chunk[0] for chunk in batch]
                texts = [chunk[1] for chunk in batch]
                
                try:
                    # Generate embeddings with model-specific formatting
                    embeddings = embed_texts(texts, self.target_model, normalize=True, text_type="passage")
                    
                    # Add to new index
                    start_idx = new_index.ntotal
                    new_index.add(embeddings.astype(np.float32))
                    
                    # Map chunk IDs to index positions
                    for i, chunk_id in enumerate(chunk_ids):
                        chunk_id_to_index[chunk_id] = start_idx + i
                    
                    # Update progress
                    progress["processed_chunks"] += len(batch)
                    progress["last_chunk_id"] = max(chunk_ids)
                    self.save_progress(progress)
                    
                    progress_bar.update(task, completed=progress["processed_chunks"])
                    
                except Exception as e:
                    self.logger.error(f"Error processing batch starting at chunk {progress['last_chunk_id']}: {e}")
                    rprint(f"❌ Error: {e}")
                    return False
                    
                # Check if we've processed all in this batch
                if len(batch) < self.batch_size:
                    break
        
        # Save new vector index
        rprint(f"💾 Saving new vector index with {new_index.ntotal} embeddings...")
        faiss.write_index(new_index, str(self.vector_path))
        
        # Update chunk ID mapping in database
        rprint("🔄 Updating database with new embedding mappings...")
        self.update_chunk_mappings(chunk_id_to_index)
        
        # Mark migration as completed
        progress["completed"] = True
        progress["completed_at"] = time.time()
        self.save_progress(progress)
        
        rprint("✅ Migration completed successfully!")
        rprint(f"📈 Migrated {progress['processed_chunks']} chunks to {self.target_model}")
        
        return True
    
    def update_chunk_mappings(self, chunk_id_to_index: Dict[int, int]) -> None:
        """Update database with new chunk-to-vector mappings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for chunk_id, vector_idx in chunk_id_to_index.items():
            cursor.execute("""
                UPDATE chunks SET vector_index = ? WHERE id = ?
            """, (vector_idx, chunk_id))
        
        conn.commit()
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Migrate embeddings to a new model")
    parser.add_argument("--model", type=str, help="Target embedding model name")
    parser.add_argument("--version", type=str, default="2.0.0", help="Target model version")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--resume", action="store_true", help="Resume previous migration")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without doing it")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if args.resume:
        # Load target model from progress file
        progress_file = Path("data/migration_progress.json")
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                target_model = progress.get("target_model")
                target_version = progress.get("target_version", "2.0.0")
        else:
            rprint("❌ No previous migration found to resume")
            return 1
    else:
        if not args.model:
            rprint("❌ Model name required for new migration")
            rprint("Example: python scripts/migrate_embeddings.py --model intfloat/e5-base-v2")
            return 1
        target_model = args.model
        target_version = args.version
    
    migrator = EmbeddingMigrator(target_model, target_version, args.batch_size)
    
    if args.dry_run:
        total_chunks = migrator.count_total_chunks()
        rprint(f"🔍 Dry run mode:")
        rprint(f"   Target model: {target_model}")
        rprint(f"   Total chunks to migrate: {total_chunks}")
        rprint(f"   Batch size: {args.batch_size}")
        return 0
    
    try:
        success = migrator.migrate_embeddings(resume=args.resume)
        return 0 if success else 1
    except KeyboardInterrupt:
        rprint("\n⏸️  Migration interrupted. Use --resume to continue later.")
        return 1
    except Exception as e:
        rprint(f"❌ Migration failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())