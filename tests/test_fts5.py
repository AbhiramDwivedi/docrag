"""Test FTS5 full-text search index creation and querying."""

import sqlite3
import tempfile
import pytest
from pathlib import Path


def check_fts5_availability() -> bool:
    """Check if SQLite was compiled with FTS5 support."""
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE VIRTUAL TABLE test_fts USING fts5(content)")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


def create_fts5_index(db_path: str, documents: list) -> bool:
    """Create FTS5 index with test documents."""
    if not check_fts5_availability():
        return False
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create FTS5 virtual table
    cursor.execute("""
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            doc_id UNINDEXED,
            title,
            content,
            file_path UNINDEXED
        )
    """)
    
    # Insert documents
    for doc in documents:
        cursor.execute(
            "INSERT INTO documents_fts (doc_id, title, content, file_path) VALUES (?, ?, ?, ?)",
            (doc['id'], doc['title'], doc['content'], doc['file_path'])
        )
    
    conn.commit()
    conn.close()
    return True


def search_fts5(db_path: str, query: str, limit: int = 10) -> list:
    """Search FTS5 index."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Use FTS5 match syntax
    cursor.execute(
        "SELECT doc_id, title, file_path, rank FROM documents_fts WHERE documents_fts MATCH ? ORDER BY rank LIMIT ?",
        (query, limit)
    )
    
    results = cursor.fetchall()
    conn.close()
    return results


class TestFTS5Index:
    """Test FTS5 full-text search functionality."""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {
                'id': 'doc1',
                'title': 'Acme Corporation Overview',
                'content': 'Acme is a leading data platform company that provides analytics solutions.',
                'file_path': 'acme_overview.txt'
            },
            {
                'id': 'doc2', 
                'title': 'BOLT Protocol Specification',
                'content': 'BOLT is a high-performance protocol for real-time data streaming.',
                'file_path': 'bolt_spec.md'
            },
            {
                'id': 'doc3',
                'title': 'Globex Financial Services',
                'content': 'Globex specializes in technology sector investments and M&A advisory.',
                'file_path': 'globex_financial.md'
            }
        ]
    
    def test_fts5_availability(self):
        """Test that FTS5 is available in the SQLite build."""
        # This test will be skipped if FTS5 is not available
        if not check_fts5_availability():
            pytest.skip("FTS5 not available in this SQLite build")
            
        assert check_fts5_availability() is True
    
    def test_fts5_index_creation(self, sample_documents):
        """Test creating FTS5 index with documents."""
        if not check_fts5_availability():
            pytest.skip("FTS5 not available in this SQLite build")
            
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
            
        try:
            success = create_fts5_index(db_path, sample_documents)
            assert success is True
            
            # Verify table was created
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'")
            result = cursor.fetchone()
            conn.close()
            
            assert result is not None
            
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    def test_fts5_simple_search(self, sample_documents):
        """Test simple FTS5 search queries."""
        if not check_fts5_availability():
            pytest.skip("FTS5 not available in this SQLite build")
            
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
            
        try:
            create_fts5_index(db_path, sample_documents)
            
            # Test exact word match
            results = search_fts5(db_path, "Acme")
            assert len(results) >= 1
            assert any("acme_overview.txt" in str(result) for result in results)
            
            # Test protocol search
            results = search_fts5(db_path, "protocol")
            assert len(results) >= 1
            assert any("bolt_spec.md" in str(result) for result in results)
            
            # Test technology search
            results = search_fts5(db_path, "technology")
            assert len(results) >= 1
            assert any("globex_financial.md" in str(result) for result in results)
            
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    def test_fts5_phrase_search(self, sample_documents):
        """Test FTS5 phrase search with quotes."""
        if not check_fts5_availability():
            pytest.skip("FTS5 not available in this SQLite build")
            
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
            
        try:
            create_fts5_index(db_path, sample_documents)
            
            # Test phrase search
            results = search_fts5(db_path, '"data platform"')
            assert len(results) >= 1
            
            # Test phrase that shouldn't match
            results = search_fts5(db_path, '"nonexistent phrase"')
            assert len(results) == 0
            
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    def test_fts5_boolean_search(self, sample_documents):
        """Test FTS5 boolean search operators."""
        if not check_fts5_availability():
            pytest.skip("FTS5 not available in this SQLite build")
            
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
            
        try:
            create_fts5_index(db_path, sample_documents)
            
            # Test AND operator
            results = search_fts5(db_path, "technology AND sector")
            assert len(results) >= 1
            
            # Test OR operator
            results = search_fts5(db_path, "Acme OR BOLT")
            assert len(results) >= 2
            
            # Test NOT operator
            results = search_fts5(db_path, "data NOT protocol")
            # Should find Acme doc but not BOLT doc
            found_paths = [result[2] for result in results]
            assert "acme_overview.txt" in found_paths
            assert "bolt_spec.md" not in found_paths
            
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    def test_fts5_case_insensitive(self, sample_documents):
        """Test that FTS5 search is case insensitive."""
        if not check_fts5_availability():
            pytest.skip("FTS5 not available in this SQLite build")
            
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
            
        try:
            create_fts5_index(db_path, sample_documents)
            
            # Test different cases
            results_upper = search_fts5(db_path, "ACME")
            results_lower = search_fts5(db_path, "acme")
            results_mixed = search_fts5(db_path, "AcMe")
            
            # Should all return same results
            assert len(results_upper) == len(results_lower) == len(results_mixed)
            assert len(results_upper) >= 1
            
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    def test_fts5_empty_query(self, sample_documents):
        """Test FTS5 behavior with empty query."""
        if not check_fts5_availability():
            pytest.skip("FTS5 not available in this SQLite build")
            
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
            
        try:
            create_fts5_index(db_path, sample_documents)
            
            # Empty query should return no results or all results
            results = search_fts5(db_path, "")
            # FTS5 typically requires non-empty queries
            assert isinstance(results, list)
            
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    def test_fts5_no_matches(self, sample_documents):
        """Test FTS5 search with no matching documents."""
        if not check_fts5_availability():
            pytest.skip("FTS5 not available in this SQLite build")
            
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
            
        try:
            create_fts5_index(db_path, sample_documents)
            
            # Search for term not in any document
            results = search_fts5(db_path, "nonexistentterm")
            assert len(results) == 0
            
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    def test_fts5_limit_parameter(self, sample_documents):
        """Test FTS5 search result limiting."""
        if not check_fts5_availability():
            pytest.skip("FTS5 not available in this SQLite build")
            
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
            
        try:
            create_fts5_index(db_path, sample_documents)
            
            # Search with different limits
            results_all = search_fts5(db_path, "data", limit=10)
            results_limited = search_fts5(db_path, "data", limit=1)
            
            assert len(results_limited) <= 1
            assert len(results_limited) <= len(results_all)
            
        finally:
            Path(db_path).unlink(missing_ok=True)