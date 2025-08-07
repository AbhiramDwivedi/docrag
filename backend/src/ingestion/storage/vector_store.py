"""Thin wrapper over FAISS index + SQLite metadata."""
import faiss
import sqlite3
from pathlib import Path
import numpy as np
from typing import List, Any, Tuple

class VectorStore:
    def __init__(self, index_path: Path, db_path: Path, dim: int):
        self.index_path = index_path
        self.db_path = db_path
        self.dim = dim
        self._load_index()
        self._init_db()

    @classmethod
    def load(cls, index_path: Path, db_path: Path = None, dim: int = 384):
        """Load an existing vector store from files with automatic migration.
        
        Args:
            index_path: Path to FAISS index file
            db_path: Path to SQLite database file (derived from index_path if None)
            dim: Vector dimension (default 384 for sentence-transformers)
            
        Returns:
            VectorStore instance with migrated schema
        """
        if db_path is None:
            db_path = index_path.with_suffix('.db')
        
        instance = cls(index_path, db_path, dim)
        
        # Automatically migrate schema for existing databases
        try:
            instance.migrate_database_schema()
        except Exception as e:
            print(f"Warning: Could not migrate database schema: {e}")
        
        return instance

    def _load_index(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatL2(self.dim)

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        cur = self.conn.cursor()
        
        # Check if chunks table exists and get its current schema
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
        table_exists = cur.fetchone() is not None
        
        if table_exists:
            # Get current columns
            cur.execute("PRAGMA table_info(chunks)")
            existing_columns = [col[1] for col in cur.fetchall()]
            
            # Check if we have the new enhanced schema
            has_enhanced_schema = all(col in existing_columns for col in [
                'document_id', 'document_path', 'document_title', 
                'section_id', 'chunk_index', 'total_chunks', 'document_type'
            ])
            
            if not has_enhanced_schema:
                # Need to migrate - just create table with basic schema for now
                # Migration will be handled separately
                if 'faiss_idx' not in existing_columns:
                    cur.execute("ALTER TABLE chunks ADD COLUMN faiss_idx INTEGER")
                self.conn.commit()
                return
        
        # Create new table with enhanced schema
        cur.execute("""CREATE TABLE IF NOT EXISTS chunks
                       (id TEXT PRIMARY KEY, file TEXT, unit TEXT,
                        text TEXT, mtime REAL, current INTEGER, faiss_idx INTEGER,
                        document_id TEXT, document_path TEXT, document_title TEXT,
                        section_id TEXT, chunk_index INTEGER, total_chunks INTEGER,
                        document_type TEXT)""")
        
        # Add indexes for efficient document-level queries (only if columns exist)
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_path ON chunks (document_path)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_type ON chunks (document_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks (chunk_index)")
        except sqlite3.OperationalError:
            # Columns might not exist yet - indexes will be created during migration
            pass
        
        self.conn.commit()

    def upsert(self, chunk_ids: List[str], vectors: Any, metadata_rows: List[Tuple]):
        vectors = np.asarray(vectors, dtype='float32')
        
        # Get current FAISS index size to assign new indices
        current_size = self.index.ntotal
        
        # Add vectors to FAISS
        self.index.add(vectors)
        
        # Update metadata with FAISS indices
        cur = self.conn.cursor()
        for i, row in enumerate(metadata_rows):
            # Support both legacy and new formats for backward compatibility
            if len(row) == 6:
                # Legacy format: (id, file, unit, text, mtime, current)
                # Extend with FAISS index and default document fields
                extended_row = row + (current_size + i, None, None, None, None, None, None, None)
                cur.execute("REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", extended_row)
            elif len(row) == 13:
                # New format: (id, file, unit, text, mtime, current, document_id, document_path, 
                #             document_title, section_id, chunk_index, total_chunks, document_type)
                # Add FAISS index
                extended_row = row + (current_size + i,)
                cur.execute("REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", extended_row)
            else:
                raise ValueError(f"Invalid metadata row length: {len(row)}. Expected 6 or 13 fields.")
        
        self.conn.commit()
        faiss.write_index(self.index, str(self.index_path))

    def query(self, vector: Any, k: int = 8) -> List[dict]:
        vec = np.asarray(vector, dtype='float32').reshape(1, -1)
        dists, idxs = self.index.search(vec, k)
        
        # Map FAISS indices back to chunk data via SQLite
        results = []
        cur = self.conn.cursor()
        for i, (dist, idx) in enumerate(zip(dists[0], idxs[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            # Get chunk by FAISS index with document metadata
            cur.execute("""SELECT id, file, unit, text, document_id, document_path, 
                                 document_title, section_id, chunk_index, total_chunks, document_type 
                          FROM chunks WHERE faiss_idx = ? AND current = 1""", (int(idx),))
            row = cur.fetchone()
            if row:
                results.append({
                    'id': row[0],
                    'file': row[1], 
                    'unit': row[2],
                    'text': row[3],
                    'distance': float(dist),
                    'document_id': row[4],
                    'document_path': row[5],
                    'document_title': row[6],
                    'section_id': row[7],
                    'chunk_index': row[8],
                    'total_chunks': row[9],
                    'document_type': row[10]
                })
        return results
    
    def get_chunks_by_document(self, document_id: str, expand_context: bool = True, window_size: int = 3) -> List[dict]:
        """Get all chunks for a specific document with optional context expansion.
        
        Args:
            document_id: Document identifier to retrieve chunks for
            expand_context: Whether to expand context around found chunks
            window_size: Number of surrounding chunks to include for context
            
        Returns:
            List of chunk dictionaries with expanded context if requested
        """
        cur = self.conn.cursor()
        
        if expand_context:
            # Get chunks with context window
            cur.execute("""
                SELECT id, file, unit, text, document_id, document_path, 
                       document_title, section_id, chunk_index, total_chunks, document_type 
                FROM chunks 
                WHERE document_id = ? AND current = 1
                ORDER BY chunk_index
            """, (document_id,))
        else:
            # Get just the specific chunks
            cur.execute("""
                SELECT id, file, unit, text, document_id, document_path, 
                       document_title, section_id, chunk_index, total_chunks, document_type 
                FROM chunks 
                WHERE document_id = ? AND current = 1
                ORDER BY chunk_index
            """, (document_id,))
        
        rows = cur.fetchall()
        results = []
        
        for row in rows:
            chunk_data = {
                'id': row[0],
                'file': row[1],
                'unit': row[2], 
                'text': row[3],
                'document_id': row[4],
                'document_path': row[5],
                'document_title': row[6],
                'section_id': row[7],
                'chunk_index': row[8],
                'total_chunks': row[9],
                'document_type': row[10]
            }
            results.append(chunk_data)
        
        return results
    
    def get_document_context(self, chunk_ids: List[str], window_size: int = 3) -> List[dict]:
        """Get document context around specific chunks.
        
        Args:
            chunk_ids: List of chunk IDs to get context for
            window_size: Number of surrounding chunks to include
            
        Returns:
            List of chunks with expanded context
        """
        cur = self.conn.cursor()
        all_chunks = []
        
        for chunk_id in chunk_ids:
            # Get the target chunk to determine its position
            cur.execute("""
                SELECT chunk_index, document_id FROM chunks 
                WHERE id = ? AND current = 1
            """, (chunk_id,))
            
            chunk_info = cur.fetchone()
            if not chunk_info:
                continue
                
            chunk_index, document_id = chunk_info
            
            # Get surrounding chunks within the window
            start_index = max(0, chunk_index - window_size)
            end_index = chunk_index + window_size + 1
            
            cur.execute("""
                SELECT id, file, unit, text, document_id, document_path, 
                       document_title, section_id, chunk_index, total_chunks, document_type 
                FROM chunks 
                WHERE document_id = ? AND chunk_index >= ? AND chunk_index < ? AND current = 1
                ORDER BY chunk_index
            """, (document_id, start_index, end_index))
            
            context_chunks = cur.fetchall()
            
            for row in context_chunks:
                chunk_data = {
                    'id': row[0],
                    'file': row[1],
                    'unit': row[2],
                    'text': row[3],
                    'document_id': row[4],
                    'document_path': row[5],
                    'document_title': row[6],
                    'section_id': row[7],
                    'chunk_index': row[8],
                    'total_chunks': row[9],
                    'document_type': row[10],
                    'is_original': row[0] == chunk_id  # Mark if this was an original search result
                }
                
                # Avoid duplicates
                if chunk_data not in all_chunks:
                    all_chunks.append(chunk_data)
        
        # Sort by document and chunk index
        all_chunks.sort(key=lambda x: (x['document_id'], x['chunk_index']))
        return all_chunks
    
    def rank_documents_by_relevance(self, chunk_scores: List[dict]) -> List[dict]:
        """Rank documents by aggregated chunk relevance scores.
        
        Args:
            chunk_scores: List of chunk results with distance scores
            
        Returns:
            List of documents ranked by relevance with aggregated scores
        """
        document_scores = {}
        
        for chunk in chunk_scores:
            doc_id = chunk.get('document_id')
            if not doc_id:
                continue
                
            distance = chunk.get('distance', 1.0)
            relevance = 1.0 - distance  # Convert distance to relevance score
            
            if doc_id not in document_scores:
                document_scores[doc_id] = {
                    'document_id': doc_id,
                    'document_path': chunk.get('document_path'),
                    'document_title': chunk.get('document_title'),
                    'document_type': chunk.get('document_type'),
                    'relevance_score': 0.0,
                    'chunk_count': 0,
                    'chunks': []
                }
            
            document_scores[doc_id]['relevance_score'] += relevance
            document_scores[doc_id]['chunk_count'] += 1
            document_scores[doc_id]['chunks'].append(chunk)
        
        # Calculate average relevance and sort
        ranked_docs = []
        for doc_info in document_scores.values():
            doc_info['avg_relevance'] = doc_info['relevance_score'] / doc_info['chunk_count']
            ranked_docs.append(doc_info)
        
        # Sort by average relevance (higher is better)
        ranked_docs.sort(key=lambda x: x['avg_relevance'], reverse=True)
        return ranked_docs
    
    def migrate_database_schema(self):
        """Migrate existing database to include document-level metadata fields.
        
        This method safely adds new columns to existing chunks table without losing data.
        """
        cur = self.conn.cursor()
        
        # Get current table info
        cur.execute("PRAGMA table_info(chunks)")
        columns = [col[1] for col in cur.fetchall()]
        
        # Check which columns need to be added
        new_columns = [
            ('document_id', 'TEXT'),
            ('document_path', 'TEXT'), 
            ('document_title', 'TEXT'),
            ('section_id', 'TEXT'),
            ('chunk_index', 'INTEGER'),
            ('total_chunks', 'INTEGER'),
            ('document_type', 'TEXT')
        ]
        
        for col_name, col_type in new_columns:
            if col_name not in columns:
                try:
                    cur.execute(f"ALTER TABLE chunks ADD COLUMN {col_name} {col_type}")
                    print(f"Added column {col_name} to chunks table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise e
        
        # Add indexes if they don't exist
        indexes = [
            ("idx_chunks_document_id", "chunks", "document_id"),
            ("idx_chunks_document_path", "chunks", "document_path"),
            ("idx_chunks_document_type", "chunks", "document_type"),
            ("idx_chunks_chunk_index", "chunks", "chunk_index")
        ]
        
        for idx_name, table_name, column_name in indexes:
            try:
                cur.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} ({column_name})")
            except sqlite3.OperationalError:
                pass  # Index might already exist
        
        self.conn.commit()
        print("Database schema migration completed successfully")
