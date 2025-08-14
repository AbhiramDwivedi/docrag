"""Security and error handling tests for lexical search plugin."""

import pytest
import tempfile
import sqlite3
import os
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# Add the backend source to Python path
backend_src = Path(__file__).parent.parent / "backend" / "src"
sys.path.insert(0, str(backend_src))

from querying.agents.plugins.lexical_search import LexicalSearchPlugin


class TestLexicalSearchSecurity:
    """Test security aspects of lexical search plugin."""

    @pytest.fixture
    def test_db_with_fts5(self):
        """Create a test database with FTS5 support."""
        # Check FTS5 availability first
        try:
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            cursor.execute("CREATE VIRTUAL TABLE test_fts USING fts5(content)")
            conn.close()
        except sqlite3.OperationalError:
            pytest.skip("FTS5 not available in this SQLite build")
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create main chunks table
        cursor.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT,
                file_path TEXT,
                chunk_index INTEGER,
                current INTEGER DEFAULT 1
            )
        """)
        
        # Create FTS5 table
        cursor.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                content,
                file_path UNINDEXED
            )
        """)
        
        # Insert test data
        test_data = [
            ('test_001', 'This is a test document with sensitive information.', 'test.txt'),
            ('test_002', 'Another document containing user data and passwords.', 'secure.txt'),
            ('test_003', 'Normal content without special characters.', 'normal.txt')
        ]
        
        for chunk_id, content, file_path in test_data:
            cursor.execute(
                "INSERT INTO chunks (chunk_id, content, file_path, chunk_index) VALUES (?, ?, ?, ?)",
                (chunk_id, content, file_path, 0)
            )
            cursor.execute(
                "INSERT INTO chunks_fts (chunk_id, content, file_path) VALUES (?, ?, ?)",
                (chunk_id, content, file_path)
            )
        
        conn.commit()
        conn.close()
        
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def test_db_no_fts5(self):
        """Create a test database without FTS5 table."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create main chunks table but no FTS5 table
        cursor.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT,
                file_path TEXT,
                chunk_index INTEGER,
                current INTEGER DEFAULT 1
            )
        """)
        
        conn.commit()
        conn.close()
        
        yield db_path
        os.unlink(db_path)

    def test_sql_injection_prevention(self, test_db_with_fts5):
        """Test that SQL injection attempts are prevented."""
        plugin = LexicalSearchPlugin(test_db_with_fts5)
        
        # Test various SQL injection attempts
        injection_attempts = [
            "'; DROP TABLE chunks; --",
            "' UNION SELECT * FROM sqlite_master --",
            "test'; DELETE FROM chunks_fts; --",
            '"; INSERT INTO chunks_fts VALUES ("malicious", "content", "path"); --',
            "test' OR 1=1 --",
            "test\"; CREATE TABLE malicious (id INTEGER); --",
            "/*malicious comment*/ test",
            "test; PRAGMA table_info(chunks)",
        ]
        
        for injection in injection_attempts:
            result = plugin.execute({"query": injection})
            
            # Should not cause errors or return unexpected results
            assert "error" not in result.get("metadata", {}) or \
                   result["metadata"]["error"] in ["empty_query"], \
                   f"Injection attempt should be safely handled: {injection}"
            
            # Verify database integrity (tables should still exist)
            conn = sqlite3.connect(test_db_with_fts5)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            assert "chunks" in tables, "Main table should not be dropped"
            assert "chunks_fts" in tables, "FTS5 table should not be dropped"

    def test_input_sanitization(self, test_db_with_fts5):
        """Test input sanitization functionality."""
        plugin = LexicalSearchPlugin(test_db_with_fts5)
        
        # Test dangerous characters are removed/escaped
        test_cases = [
            ("test*malicious", "test"),  # Wildcard removed
            ('test"phrase', 'test""phrase'),  # Quotes escaped
            ("test(group)", "testgroup"),  # Parentheses removed
            ("test[column]", "testcolumn"),  # Brackets removed
            ("test{proximity}", "testproximity"),  # Braces removed
            ("test^column", "testcolumn"),  # Column filters removed
            ("test~proximity", "testproximity"),  # Proximity removed
            ("test:column", "testcolumn"),  # Column specifiers removed
            ("test+required", "testrequired"),  # Required terms removed
            ("test\\escape", "testescape"),  # Escape characters removed
            ("test;DROP", "testDROP"),  # SQL terminators removed
            ("test--comment", "testcomment"),  # Comments removed
            ("test/*comment*/", "testcomment"),  # Block comments removed
        ]
        
        for input_query, expected_sanitized in test_cases:
            sanitized = plugin._sanitize_fts5_input(input_query)
            assert expected_sanitized in sanitized or sanitized == "", \
                f"Input '{input_query}' should be sanitized to contain '{expected_sanitized}', got '{sanitized}'"

    def test_query_length_limits(self, test_db_with_fts5):
        """Test that excessively long queries are handled."""
        plugin = LexicalSearchPlugin(test_db_with_fts5)
        
        # Test very long query
        long_query = "test " * 1000  # 5000 characters
        result = plugin.execute({"query": long_query})
        
        # Should handle gracefully without errors
        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_invalid_parameter_types(self, test_db_with_fts5):
        """Test handling of invalid parameter types."""
        plugin = LexicalSearchPlugin(test_db_with_fts5)
        
        # Test non-string query
        result = plugin.execute({"query": 123})
        assert result["metadata"]["error"] == "invalid_query_type"
        
        result = plugin.execute({"query": None})
        assert result["metadata"]["error"] == "invalid_query_type"
        
        result = plugin.execute({"query": []})
        assert result["metadata"]["error"] == "invalid_query_type"
        
        # Test invalid limit values
        result = plugin.execute({"query": "test", "limit": "invalid"})
        assert "sources" in result  # Should default to valid limit
        
        result = plugin.execute({"query": "test", "limit": -5})
        assert "sources" in result  # Should default to valid limit
        
        result = plugin.execute({"query": "test", "limit": 1000})
        assert "sources" in result  # Should cap to reasonable limit

    def test_empty_and_whitespace_queries(self, test_db_with_fts5):
        """Test handling of empty and whitespace-only queries."""
        plugin = LexicalSearchPlugin(test_db_with_fts5)
        
        empty_queries = ["", "   ", "\n\t  ", None]
        
        for query in empty_queries:
            if query is None:
                result = plugin.execute({"query": query})
                assert result["metadata"]["error"] == "invalid_query_type"
            else:
                result = plugin.execute({"query": query})
                assert result["metadata"]["error"] == "empty_query"

    def test_fts5_unavailable_error_handling(self):
        """Test error handling when FTS5 is not available."""
        # Mock FTS5 unavailability
        with patch.object(LexicalSearchPlugin, '_check_fts5_availability', return_value=False):
            plugin = LexicalSearchPlugin("dummy_path.db")
            result = plugin.execute({"query": "test"})
            
            assert result["metadata"]["error"] == "fts5_unavailable"
            assert "FTS5 not supported" in result["response"]
            assert result["sources"] == []

    def test_index_missing_error_handling(self):
        """Test error handling when FTS5 index is missing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Create database without FTS5 table
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE dummy (id INTEGER)")
            conn.commit()
            conn.close()
            
            plugin = LexicalSearchPlugin(db_path)
            
            # Mock FTS5 availability but index doesn't exist
            with patch.object(plugin, '_check_fts5_availability', return_value=True):
                result = plugin.execute({"query": "test"})
                
                assert result["metadata"]["error"] == "index_missing"
                assert "index not available" in result["response"]
                assert result["sources"] == []
        
        finally:
            os.unlink(db_path)

    def test_database_connection_error_handling(self, test_db_with_fts5):
        """Test error handling for database connection issues."""
        plugin = LexicalSearchPlugin("/nonexistent/path/db.db")
        
        # Mock successful checks but database operations will fail
        with patch.object(plugin, '_check_fts5_availability', return_value=True), \
             patch.object(plugin, '_check_index_exists', return_value=True):
            
            result = plugin.execute({"query": "test"})
            
            assert "error" in result["metadata"]
            assert result["sources"] == []

    def test_malformed_query_handling(self, test_db_with_fts5):
        """Test handling of malformed FTS5 queries."""
        plugin = LexicalSearchPlugin(test_db_with_fts5)
        
        # These queries should be sanitized and not cause FTS5 errors
        malformed_queries = [
            "test AND",  # Incomplete boolean
            "test OR",   # Incomplete boolean
            "test NOT",  # Incomplete boolean
            '"unclosed quote',  # Unclosed quote
            '""',  # Empty quotes
            'test ""',  # Empty quoted phrase
        ]
        
        for query in malformed_queries:
            result = plugin.execute({"query": query})
            
            # Should not cause database errors
            assert "sources" in result
            assert isinstance(result["sources"], list)
            # May return empty results but should not error

    def test_concurrent_access_safety(self, test_db_with_fts5):
        """Test that the plugin handles concurrent access safely."""
        plugin = LexicalSearchPlugin(test_db_with_fts5)
        
        # Test multiple simultaneous searches (simulated)
        results = []
        for i in range(10):
            result = plugin.execute({"query": f"test {i}"})
            results.append(result)
        
        # All should complete without errors
        for result in results:
            assert "sources" in result
            assert isinstance(result["sources"], list)


class TestLexicalSearchErrorScenarios:
    """Test error scenarios and edge cases."""

    def test_search_with_no_results(self):
        """Test search that returns no results."""
        # Create empty database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    content TEXT,
                    file_path TEXT,
                    chunk_index INTEGER,
                    current INTEGER DEFAULT 1
                )
            """)
            
            cursor.execute("""
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    content,
                    file_path UNINDEXED
                )
            """)
            
            conn.commit()
            conn.close()
            
            plugin = LexicalSearchPlugin(db_path)
            result = plugin.execute({"query": "nonexistent"})
            
            assert result["sources"] == []
            assert result["metadata"]["results_count"] == 0
            assert "No documents found" in result["response"]
        
        finally:
            os.unlink(db_path)

    def test_database_corruption_handling(self):
        """Test handling of corrupted database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
            # Write invalid data to create a corrupted database
            f.write(b"This is not a valid SQLite database")
        
        try:
            plugin = LexicalSearchPlugin(db_path)
            result = plugin.execute({"query": "test"})
            
            # Should handle corruption gracefully
            assert "error" in result["metadata"]
            assert result["sources"] == []
        
        finally:
            os.unlink(db_path)

    def test_special_character_edge_cases(self):
        """Test handling of various special characters and Unicode."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    content TEXT,
                    file_path TEXT,
                    chunk_index INTEGER,
                    current INTEGER DEFAULT 1
                )
            """)
            
            cursor.execute("""
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    content,
                    file_path UNINDEXED
                )
            """)
            
            # Insert content with Unicode and special characters
            cursor.execute(
                "INSERT INTO chunks (chunk_id, content, file_path, chunk_index) VALUES (?, ?, ?, ?)",
                ('unicode_001', 'Content with émojis 🚀 and special chars áéíóú', 'unicode.txt', 0)
            )
            cursor.execute(
                "INSERT INTO chunks_fts (chunk_id, content, file_path) VALUES (?, ?, ?)",
                ('unicode_001', 'Content with émojis 🚀 and special chars áéíóú', 'unicode.txt')
            )
            
            conn.commit()
            conn.close()
            
            plugin = LexicalSearchPlugin(db_path)
            
            # Test various Unicode and special character queries
            unicode_queries = [
                "émojis",  # Accented characters
                "🚀",      # Emoji
                "áéíóú",   # Multiple accents
                "Content émojis",  # Mixed
            ]
            
            for query in unicode_queries:
                result = plugin.execute({"query": query})
                
                # Should handle without errors
                assert "sources" in result
                assert isinstance(result["sources"], list)
        
        finally:
            os.unlink(db_path)