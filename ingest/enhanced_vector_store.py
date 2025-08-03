"""Enhanced vector store with rich metadata support for Phase 2.

This module extends the basic vector store with comprehensive metadata
storage for files, emails, and other document types to support complex
queries like "Find emails from John last week" and "Show newest Excel files".
"""

import faiss
import sqlite3
from pathlib import Path
import numpy as np
from typing import List, Any, Tuple, Dict, Optional, Union
from datetime import datetime
import json

from .vector_store import VectorStore


class EnhancedVectorStore(VectorStore):
    """Enhanced vector store with rich metadata support.
    
    Extends the basic VectorStore to support:
    - Comprehensive file metadata (size, dates, types, paths)  
    - Email-specific metadata (senders, recipients, subjects)
    - Efficient querying for metadata-based searches
    - Backward compatibility with existing schema
    """
    
    def __init__(self, index_path: Path, db_path: Path, dim: int):
        """Initialize enhanced vector store with metadata tables.
        
        Args:
            index_path: Path to FAISS index file
            db_path: Path to SQLite database file
            dim: Vector dimension
        """
        super().__init__(index_path, db_path, dim)
        self._init_metadata_tables()
    
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
        super().upsert(chunk_ids, vectors, metadata_rows)
        
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

    @classmethod
    def load(cls, index_path: Path, db_path: Path, dim: int = 384):
        """Load an existing enhanced vector store from files.
        
        Args:
            index_path: Path to FAISS index file
            db_path: Path to SQLite database file
            dim: Vector dimension (default 384 for sentence-transformers)
            
        Returns:
            EnhancedVectorStore instance
        """
        return cls(index_path, db_path, dim)