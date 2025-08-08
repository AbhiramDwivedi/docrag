"""Vector store with rich metadata support for Phase 2.

This module provides comprehensive metadata storage for files, emails, and
other document types to support complex queries like "Find emails from John 
last week" and "Show newest Excel files".
"""
import faiss
import sqlite3
from pathlib import Path
import numpy as np
from typing import List, Any, Tuple, Dict, Optional, Union
from datetime import datetime
import json

class VectorStore:
    """Vector store with rich metadata support.
    
    Provides comprehensive document storage and search capabilities:
    - File metadata (size, dates, types, paths)  
    - Email-specific metadata (senders, recipients, subjects)
    - Efficient querying for metadata-based searches
    - Backward compatibility with existing schema
    """
    
    def __init__(self, index_path: Path, db_path: Path, dim: int):
        """Initialize vector store with metadata tables.
        
        Args:
            index_path: Path to FAISS index file
            db_path: Path to SQLite database file
            dim: Vector dimension
        """
        self.index_path = index_path
        self.db_path = db_path
        self.dim = dim
        self._load_index()
        self._init_db()
        self._init_metadata_tables()

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
                # Add missing faiss_idx column if needed
                if 'faiss_idx' not in existing_columns:
                    try:
                        cur.execute("ALTER TABLE chunks ADD COLUMN faiss_idx INTEGER")
                    except sqlite3.OperationalError:
                        pass  # Column might already exist
                
                # Run migration to add enhanced columns
                try:
                    self.migrate_database_schema()
                except Exception as e:
                    print(f"Warning: Migration failed: {e}")
                
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

    def _init_metadata_tables(self):
        """Initialize additional metadata tables for Phase 2."""
        cur = self.conn.cursor()
        
        # Files table for comprehensive file metadata
        cur.execute("""
            CREATE TABLE IF NOT EXISTS files (
                file_path TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_extension TEXT NOT NULL,
                file_size INTEGER,
                created_time REAL,
                modified_time REAL,
                accessed_time REAL,
                file_type TEXT,  -- PDF, DOCX, EMAIL, etc.
                chunk_count INTEGER DEFAULT 0,
                ingestion_time REAL,
                metadata_json TEXT  -- Additional flexible metadata as JSON
            )
        """)
        
        # Email metadata table for email-specific data
        cur.execute("""
            CREATE TABLE IF NOT EXISTS email_metadata (
                file_path TEXT PRIMARY KEY,
                message_id TEXT,
                thread_id TEXT,
                sender_email TEXT,
                sender_name TEXT,
                recipients TEXT,  -- JSON array of recipients
                cc_recipients TEXT,  -- JSON array of CC recipients  
                bcc_recipients TEXT,  -- JSON array of BCC recipients
                subject TEXT,
                email_date REAL,  -- Email date as timestamp
                has_attachments BOOLEAN DEFAULT 0,
                attachment_count INTEGER DEFAULT 0,
                conversation_index TEXT,  -- For threading
                importance TEXT,  -- High, Normal, Low
                flags TEXT,  -- JSON array of email flags
                FOREIGN KEY (file_path) REFERENCES files (file_path)
            )
        """)
        
        # Create indexes for efficient querying
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_extension ON files (file_extension)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_modified ON files (modified_time)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_type ON files (file_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_size ON files (file_size)")
        
        cur.execute("CREATE INDEX IF NOT EXISTS idx_email_sender ON email_metadata (sender_email)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_email_date ON email_metadata (email_date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_email_subject ON email_metadata (subject)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_email_thread ON email_metadata (thread_id)")
        
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
                # Insert with FAISS index in the correct position (after current, before document_id)
                extended_row = row[:6] + (current_size + i,) + row[6:]
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
    
    def upsert_with_metadata(self, 
                           chunk_ids: List[str], 
                           vectors: Any, 
                           metadata_rows: List[Tuple],
                           file_metadata: Optional[Dict[str, Any]] = None,
                           email_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Upsert chunks with rich file and email metadata.
        
        Args:
            chunk_ids: List of chunk IDs
            vectors: Vector embeddings
            metadata_rows: Chunk metadata tuples  
            file_metadata: File-level metadata dict
            email_metadata: Email-specific metadata dict (if file is email)
        """
        # First, call parent upsert for backward compatibility
        self.upsert(chunk_ids, vectors, metadata_rows)
        
        # Then handle rich metadata if provided
        if file_metadata:
            self._upsert_file_metadata(file_metadata)
        
        if email_metadata:
            self._upsert_email_metadata(email_metadata)
    
    def _upsert_file_metadata(self, file_metadata: Dict[str, Any]) -> None:
        """Insert or update file metadata."""
        cur = self.conn.cursor()
        
        # Convert datetime objects to timestamps
        for time_field in ['created_time', 'modified_time', 'accessed_time', 'ingestion_time']:
            if time_field in file_metadata and isinstance(file_metadata[time_field], datetime):
                file_metadata[time_field] = file_metadata[time_field].timestamp()
        
        # Convert additional metadata to JSON
        if 'metadata_json' in file_metadata and isinstance(file_metadata['metadata_json'], dict):
            file_metadata['metadata_json'] = json.dumps(file_metadata['metadata_json'])
        
        cur.execute("""
            INSERT OR REPLACE INTO files 
            (file_path, file_name, file_extension, file_size, created_time, 
             modified_time, accessed_time, file_type, chunk_count, 
             ingestion_time, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_metadata.get('file_path'),
            file_metadata.get('file_name'),
            file_metadata.get('file_extension'),
            file_metadata.get('file_size'),
            file_metadata.get('created_time'),
            file_metadata.get('modified_time'),
            file_metadata.get('accessed_time'),
            file_metadata.get('file_type'),
            file_metadata.get('chunk_count', 0),
            file_metadata.get('ingestion_time'),
            file_metadata.get('metadata_json')
        ))
        
        self.conn.commit()
    
    def _upsert_email_metadata(self, email_metadata: Dict[str, Any]) -> None:
        """Insert or update email metadata."""
        cur = self.conn.cursor()
        
        # Convert datetime to timestamp
        if 'email_date' in email_metadata and isinstance(email_metadata['email_date'], datetime):
            email_metadata['email_date'] = email_metadata['email_date'].timestamp()
        
        # Convert lists to JSON
        for list_field in ['recipients', 'cc_recipients', 'bcc_recipients', 'flags']:
            if list_field in email_metadata and isinstance(email_metadata[list_field], list):
                email_metadata[list_field] = json.dumps(email_metadata[list_field])
        
        cur.execute("""
            INSERT OR REPLACE INTO email_metadata
            (file_path, message_id, thread_id, sender_email, sender_name,
             recipients, cc_recipients, bcc_recipients, subject, email_date,
             has_attachments, attachment_count, conversation_index, 
             importance, flags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            email_metadata.get('file_path'),
            email_metadata.get('message_id'),
            email_metadata.get('thread_id'),
            email_metadata.get('sender_email'),
            email_metadata.get('sender_name'),
            email_metadata.get('recipients'),
            email_metadata.get('cc_recipients'),
            email_metadata.get('bcc_recipients'),
            email_metadata.get('subject'),
            email_metadata.get('email_date'),
            email_metadata.get('has_attachments', False),
            email_metadata.get('attachment_count', 0),
            email_metadata.get('conversation_index'),
            email_metadata.get('importance'),
            email_metadata.get('flags')
        ))
        
        self.conn.commit()
    
    def query_files_by_metadata(self, 
                               file_types: Optional[List[str]] = None,
                               date_after: Optional[datetime] = None,
                               date_before: Optional[datetime] = None,
                               size_min: Optional[int] = None,
                               size_max: Optional[int] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Query files based on metadata criteria.
        
        Args:
            file_types: List of file types to filter by (e.g., ['PDF', 'DOCX'])
            date_after: Only files modified after this date
            date_before: Only files modified before this date  
            size_min: Minimum file size in bytes
            size_max: Maximum file size in bytes
            limit: Maximum number of results
            
        Returns:
            List of file metadata dictionaries
        """
        cur = self.conn.cursor()
        
        conditions = []
        params = []
        
        if file_types:
            placeholders = ','.join('?' * len(file_types))
            conditions.append(f"file_type IN ({placeholders})")
            params.extend(file_types)
        
        if date_after:
            conditions.append("modified_time >= ?")
            params.append(date_after.timestamp())
        
        if date_before:
            conditions.append("modified_time <= ?")
            params.append(date_before.timestamp())
        
        if size_min:
            conditions.append("file_size >= ?")
            params.append(size_min)
        
        if size_max:
            conditions.append("file_size <= ?")
            params.append(size_max)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT file_path, file_name, file_extension, file_size,
                   created_time, modified_time, file_type, chunk_count,
                   ingestion_time, metadata_json
            FROM files 
            WHERE {where_clause}
            ORDER BY modified_time DESC
            LIMIT ?
        """
        
        params.append(limit)
        cur.execute(query, params)
        
        results = []
        for row in cur.fetchall():
            result = {
                'file_path': row[0],
                'file_name': row[1], 
                'file_extension': row[2],
                'file_size': row[3],
                'created_time': datetime.fromtimestamp(row[4]) if row[4] else None,
                'modified_time': datetime.fromtimestamp(row[5]) if row[5] else None,
                'file_type': row[6],
                'chunk_count': row[7],
                'ingestion_time': datetime.fromtimestamp(row[8]) if row[8] else None,
                'metadata_json': json.loads(row[9]) if row[9] else {}
            }
            results.append(result)
        
        return results
    
    def query_emails_by_metadata(self,
                                sender_emails: Optional[List[str]] = None,
                                sender_names: Optional[List[str]] = None,
                                subject_contains: Optional[str] = None,
                                date_after: Optional[datetime] = None,
                                date_before: Optional[datetime] = None,
                                has_attachments: Optional[bool] = None,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """Query emails based on email-specific metadata.
        
        Args:
            sender_emails: List of sender email addresses to filter by
            sender_names: List of sender names to filter by  
            subject_contains: Filter by emails containing this text in subject
            date_after: Only emails sent after this date
            date_before: Only emails sent before this date
            has_attachments: Filter by attachment status
            limit: Maximum number of results
            
        Returns:
            List of email metadata dictionaries
        """
        cur = self.conn.cursor()
        
        conditions = []
        params = []
        
        if sender_emails:
            placeholders = ','.join('?' * len(sender_emails))
            conditions.append(f"sender_email IN ({placeholders})")
            params.extend(sender_emails)
        
        if sender_names:
            placeholders = ','.join('?' * len(sender_names))
            conditions.append(f"sender_name IN ({placeholders})")
            params.extend(sender_names)
        
        if subject_contains:
            conditions.append("subject LIKE ?")
            params.append(f"%{subject_contains}%")
        
        if date_after:
            conditions.append("email_date >= ?")
            params.append(date_after.timestamp())
        
        if date_before:
            conditions.append("email_date <= ?")
            params.append(date_before.timestamp())
        
        if has_attachments is not None:
            conditions.append("has_attachments = ?")
            params.append(has_attachments)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT e.file_path, e.message_id, e.thread_id, e.sender_email,
                   e.sender_name, e.recipients, e.subject, e.email_date,
                   e.has_attachments, e.attachment_count, f.file_name,
                   f.file_size, f.modified_time
            FROM email_metadata e
            JOIN files f ON e.file_path = f.file_path
            WHERE {where_clause}
            ORDER BY e.email_date DESC
            LIMIT ?
        """
        
        params.append(limit)
        cur.execute(query, params)
        
        results = []
        for row in cur.fetchall():
            result = {
                'file_path': row[0],
                'message_id': row[1],
                'thread_id': row[2], 
                'sender_email': row[3],
                'sender_name': row[4],
                'recipients': json.loads(row[5]) if row[5] else [],
                'subject': row[6],
                'email_date': datetime.fromtimestamp(row[7]) if row[7] else None,
                'has_attachments': bool(row[8]),
                'attachment_count': row[9],
                'file_name': row[10],
                'file_size': row[11],
                'modified_time': datetime.fromtimestamp(row[12]) if row[12] else None
            }
            results.append(result)
        
        return results
    
    def get_file_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the file collection.
        
        Returns:
            Dictionary with file collection statistics
        """
        cur = self.conn.cursor()
        
        # Basic file counts
        cur.execute("SELECT COUNT(*) FROM files")
        total_files = cur.fetchone()[0]
        
        # File type breakdown
        cur.execute("""
            SELECT file_type, COUNT(*) as count, 
                   COALESCE(SUM(file_size), 0) as total_size
            FROM files 
            GROUP BY file_type 
            ORDER BY count DESC
        """)
        file_types = [{'type': row[0], 'count': row[1], 'total_size': row[2]} 
                     for row in cur.fetchall()]
        
        # Email statistics
        cur.execute("SELECT COUNT(*) FROM email_metadata")
        total_emails = cur.fetchone()[0]
        
        cur.execute("""
            SELECT sender_email, COUNT(*) as count
            FROM email_metadata 
            GROUP BY sender_email 
            ORDER BY count DESC 
            LIMIT 10
        """)
        top_senders = [{'email': row[0], 'count': row[1]} 
                      for row in cur.fetchall()]
        
        # Recent activity
        cur.execute("""
            SELECT COUNT(*) 
            FROM files 
            WHERE modified_time >= datetime('now', '-7 days')
        """)
        recent_files = cur.fetchone()[0]
        
        # Total storage
        cur.execute("SELECT COALESCE(SUM(file_size), 0) FROM files")
        total_size = cur.fetchone()[0]
        
        return {
            'total_files': total_files,
            'total_emails': total_emails,
            'file_types': file_types,
            'top_senders': top_senders,
            'recent_files_7days': recent_files,
            'total_size_bytes': total_size,
            'total_chunks': self.index.ntotal if hasattr(self, 'index') else 0
        }
    
    def migrate_from_basic_store(self) -> None:
        """Migrate existing data from basic vector store schema.
        
        This method helps transition from Phase 1 to Phase 2 by
        extracting what metadata it can from existing chunk data.
        """
        cur = self.conn.cursor()
        
        # Get all unique files from chunks table
        cur.execute("""
            SELECT DISTINCT file, MIN(mtime)
            FROM chunks 
            WHERE current = 1
            GROUP BY file
        """)
        
        files_to_migrate = cur.fetchall()
        
        for file_path, mtime in files_to_migrate:
            if not file_path:
                continue
                
            path = Path(file_path)
            
            # Basic file metadata from path and mtime
            file_metadata = {
                'file_path': str(path),
                'file_name': path.name,
                'file_extension': path.suffix.lower(),
                'file_type': self._determine_file_type(path.suffix.lower()),
                'modified_time': datetime.fromtimestamp(mtime) if mtime else None,
                'ingestion_time': datetime.now()
            }
            
            # Try to get additional metadata if file still exists
            if path.exists():
                stat = path.stat()
                file_metadata.update({
                    'file_size': stat.st_size,
                    'created_time': datetime.fromtimestamp(stat.st_ctime),
                    'modified_time': datetime.fromtimestamp(stat.st_mtime),
                    'accessed_time': datetime.fromtimestamp(stat.st_atime)
                })
            
            # Get chunk count for this file
            cur.execute("SELECT COUNT(*) FROM chunks WHERE file = ? AND current = 1", (file_path,))
            chunk_count = cur.fetchone()[0]
            file_metadata['chunk_count'] = chunk_count
            
            self._upsert_file_metadata(file_metadata)
        
        print(f"Migrated metadata for {len(files_to_migrate)} files")
    
    def _determine_file_type(self, extension: str) -> str:
        """Determine file type from extension."""
        type_mapping = {
            '.pdf': 'PDF',
            '.docx': 'DOCX', '.doc': 'DOC',
            '.pptx': 'PPTX', '.ppt': 'PPT', 
            '.xlsx': 'XLSX', '.xls': 'XLS',
            '.txt': 'TXT',
            '.msg': 'EMAIL', '.eml': 'EMAIL',
            '.html': 'HTML', '.htm': 'HTML',
            '.md': 'MARKDOWN'
        }
        return type_mapping.get(extension.lower(), 'OTHER')
    
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
        
        changes_made = False
        for col_name, col_type in new_columns:
            if col_name not in columns:
                try:
                    cur.execute(f"ALTER TABLE chunks ADD COLUMN {col_name} {col_type}")
                    print(f"Added column {col_name} to chunks table")
                    changes_made = True
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
        
        # Only show success message if actual changes were made
        if changes_made:
            print("Database schema migration completed successfully")
