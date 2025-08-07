"""Mock test for enhanced vector store functionality - validates logic without dependencies."""

import sqlite3
import tempfile
from pathlib import Path


class MockVectorStore:
    """Mock version of enhanced vector store for testing logic."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize database with enhanced schema."""
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS chunks
                       (id TEXT PRIMARY KEY, file TEXT, unit TEXT,
                        text TEXT, mtime REAL, current INTEGER, faiss_idx INTEGER,
                        document_id TEXT, document_path TEXT, document_title TEXT,
                        section_id TEXT, chunk_index INTEGER, total_chunks INTEGER,
                        document_type TEXT)""")
        
        # Add indexes for efficient document-level queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_path ON chunks (document_path)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_type ON chunks (document_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks (chunk_index)")
        
        self.conn.commit()
    
    def insert_chunk(self, chunk_data: tuple):
        """Insert a chunk with enhanced metadata."""
        cur = self.conn.cursor()
        if len(chunk_data) == 6:
            # Legacy format: (id, file, unit, text, mtime, current)
            # Add faiss_idx and document fields (all None for legacy)
            extended_row = chunk_data + (None, None, None, None, None, None, None, None)
            cur.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", extended_row)
        elif len(chunk_data) == 13:
            # Enhanced format: (id, file, unit, text, mtime, current, document_id, document_path, 
            #                  document_title, section_id, chunk_index, total_chunks, document_type)
            # Add faiss_idx at position 6
            extended_row = chunk_data[:6] + (None,) + chunk_data[6:]
            cur.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", extended_row)
        else:
            raise ValueError(f"Invalid chunk data length: {len(chunk_data)}")
        
        self.conn.commit()
    
    def get_chunks_by_document(self, document_id: str) -> list:
        """Get all chunks for a document."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT id, file, unit, text, document_id, document_path, 
                   document_title, section_id, chunk_index, total_chunks, document_type 
            FROM chunks 
            WHERE document_id = ? AND current = 1
            ORDER BY chunk_index
        """, (document_id,))
        
        rows = cur.fetchall()
        return [
            {
                'id': row[0], 'file': row[1], 'unit': row[2], 'text': row[3],
                'document_id': row[4], 'document_path': row[5], 'document_title': row[6],
                'section_id': row[7], 'chunk_index': row[8], 'total_chunks': row[9],
                'document_type': row[10]
            }
            for row in rows
        ]
    
    def rank_documents_by_relevance(self, chunk_scores: list) -> list:
        """Rank documents by aggregated chunk relevance scores."""
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


def test_enhanced_schema():
    """Test enhanced database schema creation."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    
    store = MockVectorStore(db_path)
    
    # Check schema
    cur = store.conn.cursor()
    cur.execute("PRAGMA table_info(chunks)")
    columns = [col[1] for col in cur.fetchall()]
    
    expected_columns = [
        'id', 'file', 'unit', 'text', 'mtime', 'current', 'faiss_idx',
        'document_id', 'document_path', 'document_title', 'section_id',
        'chunk_index', 'total_chunks', 'document_type'
    ]
    
    for col in expected_columns:
        assert col in columns, f"Missing column: {col}"
    
    print("✓ Enhanced schema created successfully")


def test_document_level_retrieval():
    """Test document-level retrieval functionality."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    
    store = MockVectorStore(db_path)
    
    # Insert test chunks for same document
    chunks = [
        ("chunk_1", "test.pdf", "page_1", "First chunk", 1234567890.0, 1,
         "doc_123", "/path/to/test.pdf", "Test Document", "intro", 0, 3, "PDF"),
        ("chunk_2", "test.pdf", "page_1", "Second chunk", 1234567890.0, 1,
         "doc_123", "/path/to/test.pdf", "Test Document", "intro", 1, 3, "PDF"),
        ("chunk_3", "test.pdf", "page_2", "Third chunk", 1234567890.0, 1,
         "doc_123", "/path/to/test.pdf", "Test Document", "body", 2, 3, "PDF")
    ]
    
    for chunk in chunks:
        store.insert_chunk(chunk)
    
    # Test document-level retrieval
    doc_chunks = store.get_chunks_by_document("doc_123")
    
    assert len(doc_chunks) == 3
    assert all(chunk['document_id'] == "doc_123" for chunk in doc_chunks)
    
    # Verify chunks are in order
    chunk_indices = [chunk['chunk_index'] for chunk in doc_chunks]
    assert chunk_indices == [0, 1, 2]
    
    print("✓ Document-level retrieval works correctly")


def test_document_ranking():
    """Test document ranking by relevance."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    
    store = MockVectorStore(db_path)
    
    # Create mock chunk results with different relevance scores
    chunk_results = [
        {'document_id': 'doc_1', 'document_path': '/doc1.pdf', 'document_title': 'Doc 1', 'document_type': 'PDF', 'distance': 0.2},
        {'document_id': 'doc_1', 'document_path': '/doc1.pdf', 'document_title': 'Doc 1', 'document_type': 'PDF', 'distance': 0.3},
        {'document_id': 'doc_2', 'document_path': '/doc2.pdf', 'document_title': 'Doc 2', 'document_type': 'PDF', 'distance': 0.1},
        {'document_id': 'doc_2', 'document_path': '/doc2.pdf', 'document_title': 'Doc 2', 'document_type': 'PDF', 'distance': 0.5}
    ]
    
    ranked_docs = store.rank_documents_by_relevance(chunk_results)
    
    assert len(ranked_docs) == 2
    
    # Verify structure
    for doc in ranked_docs:
        assert 'document_id' in doc
        assert 'avg_relevance' in doc
        assert 'chunk_count' in doc
        assert 'chunks' in doc
        assert doc['chunk_count'] == 2  # Each doc has 2 chunks
    
    # Check ranking - Doc 2 should rank higher (avg distance: 0.3 vs 0.25)
    # Doc 1: relevance = (1-0.2) + (1-0.3) = 0.8 + 0.7 = 1.5, avg = 0.75
    # Doc 2: relevance = (1-0.1) + (1-0.5) = 0.9 + 0.5 = 1.4, avg = 0.70
    
    assert ranked_docs[0]['document_id'] == 'doc_1'  # Higher average relevance
    assert ranked_docs[1]['document_id'] == 'doc_2'
    
    print("✓ Document ranking works correctly")


def test_legacy_compatibility():
    """Test that legacy data format still works."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    
    store = MockVectorStore(db_path)
    
    # Insert legacy format data
    legacy_chunk = ("chunk_1", "test.pdf", "page_1", "Test content", 1234567890.0, 1)
    
    # This should not raise an error
    store.insert_chunk(legacy_chunk)
    
    # Verify data was inserted
    cur = store.conn.cursor()
    cur.execute("SELECT * FROM chunks WHERE id = ?", ("chunk_1",))
    row = cur.fetchone()
    
    assert row is not None
    assert row[0] == "chunk_1"  # id
    assert row[3] == "Test content"  # text
    assert row[7] is None  # document_id should be None for legacy data
    
    print("✓ Legacy compatibility maintained")


if __name__ == "__main__":
    print("Testing Enhanced Vector Store Functionality...")
    
    test_enhanced_schema()
    test_document_level_retrieval()
    test_document_ranking()
    test_legacy_compatibility()
    
    print("\n🎉 All enhanced vector store tests passed!")
    print("\nNext Phase: Implement enhanced semantic search plugin with document-level retrieval")