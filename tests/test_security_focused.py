"""Focused security tests for lexical search functionality."""

import pytest
import tempfile
import sqlite3
import os
from pathlib import Path
import sys
import re

# Add the backend source to Python path
backend_src = Path(__file__).parent.parent / "backend" / "src"
sys.path.insert(0, str(backend_src))


class TestLexicalSearchSecurityCore:
    """Test core security functionality without full plugin imports."""

    def test_fts5_injection_sanitization(self):
        """Test FTS5 query sanitization function directly."""
        # Import just the function we need to test
        from querying.agents.plugins.lexical_search import LexicalSearchPlugin
        
        # Create a minimal instance just for testing sanitization
        plugin = LexicalSearchPlugin()
        
        # Test dangerous inputs and their expected safe outputs
        test_cases = [
            # SQL injection attempts
            ("'; DROP TABLE chunks; --", "DROP TABLE chunks"),
            ("' UNION SELECT * FROM sqlite_master --", "UNION SELECT FROM sqlite_master"),
            ("test'; DELETE FROM chunks_fts; --", "test DELETE FROM chunks_fts"),
            
            # FTS5 special characters that could cause issues
            ("test*malicious", "testmalicious"),  # Wildcard removed
            ('test"phrase', 'test""phrase'),       # Quotes escaped
            ("test(group)", "testgroup"),          # Parentheses removed
            ("test[column]", "testcolumn"),        # Brackets removed
            ("test{proximity}", "testproximity"),  # Braces removed
            ("test^column", "testcolumn"),         # Column filters removed
            ("test~proximity", "testproximity"),   # Proximity removed
            ("test:column", "testcolumn"),         # Column specifiers removed
            ("test+required", "testrequired"),     # Required terms removed
            ("test\\escape", "testescape"),        # Escape characters removed
            ("test;DROP", "testDROP"),             # SQL terminators removed
            ("test--comment", "testcomment"),      # Comments removed
            ("test/*comment*/", "testcomment"),    # Block comments removed
            
            # Unicode and edge cases
            ("", ""),                              # Empty string
            ("   ", ""),                           # Whitespace only
            ("test   multiple   spaces", "test multiple spaces"),  # Multiple spaces normalized
        ]
        
        for input_query, expected_contains in test_cases:
            try:
                sanitized = plugin._sanitize_fts5_input(input_query)
                
                # Basic safety checks
                assert isinstance(sanitized, str), f"Output should be string for input '{input_query}'"
                
                # Should not contain dangerous SQL patterns
                dangerous_patterns = [";", "--", "/*", "*/", "DROP", "DELETE", "INSERT", "UPDATE"]
                for pattern in dangerous_patterns:
                    assert pattern not in sanitized.upper(), \
                        f"Sanitized output should not contain '{pattern}' for input '{input_query}', got '{sanitized}'"
                
                # Should contain expected safe content (if not empty)
                if expected_contains and expected_contains != "":
                    assert expected_contains in sanitized, \
                        f"Sanitized output should contain '{expected_contains}' for input '{input_query}', got '{sanitized}'"
                
                print(f"✓ '{input_query}' → '{sanitized}'")
                
            except Exception as e:
                pytest.fail(f"Sanitization failed for input '{input_query}': {e}")

    def test_query_preparation_security(self):
        """Test query preparation with security measures."""
        from querying.agents.plugins.lexical_search import LexicalSearchPlugin
        
        plugin = LexicalSearchPlugin()
        
        # Test various query preparation scenarios
        test_cases = [
            # Basic queries
            ("normal query", False, False),
            ("exact phrase", True, False),
            ("prefix", False, True),
            
            # Potentially dangerous queries
            ("'; DROP TABLE test; --", False, False),
            ('"malicious"', True, False),
            ("evil*", False, True),
        ]
        
        for query, exact_phrase, prefix_match in test_cases:
            try:
                prepared = plugin._prepare_fts5_query(query, exact_phrase, prefix_match)
                
                # Should return a string
                assert isinstance(prepared, str), f"Prepared query should be string for '{query}'"
                
                # Should not contain dangerous SQL
                assert "DROP" not in prepared.upper(), f"Prepared query should not contain DROP for '{query}'"
                assert "DELETE" not in prepared.upper(), f"Prepared query should not contain DELETE for '{query}'"
                assert "--" not in prepared, f"Prepared query should not contain SQL comments for '{query}'"
                assert ";" not in prepared, f"Prepared query should not contain semicolons for '{query}'"
                
                print(f"✓ Query '{query}' prepared as '{prepared}'")
                
            except Exception as e:
                pytest.fail(f"Query preparation failed for '{query}': {e}")

    def test_input_validation_comprehensive(self):
        """Test comprehensive input validation."""
        from querying.agents.plugins.lexical_search import LexicalSearchPlugin
        
        plugin = LexicalSearchPlugin("dummy.db")  # Non-existent DB for testing
        
        # Test invalid parameter types
        invalid_inputs = [
            {"query": None},
            {"query": 123},
            {"query": []},
            {"query": {}},
            {"query": True},
        ]
        
        for params in invalid_inputs:
            result = plugin.execute(params)
            
            # Should handle gracefully with error
            assert "error" in result.get("metadata", {}), \
                f"Should return error for invalid input {params}"
            assert result["metadata"]["error"] == "invalid_query_type", \
                f"Should return correct error type for {params}"
            assert result["sources"] == [], \
                f"Should return empty sources for invalid input {params}"

    def test_length_limits_and_dos_prevention(self):
        """Test that length limits prevent DoS attacks."""
        from querying.agents.plugins.lexical_search import LexicalSearchPlugin
        
        plugin = LexicalSearchPlugin("dummy.db")
        
        # Test very long query
        long_query = "test " * 2000  # 10,000 characters
        prepared = plugin._prepare_fts5_query(long_query)
        
        # Should be truncated or handled safely
        assert len(prepared) <= 1000, "Query should be truncated to prevent DoS"

    def test_database_with_fts5_security(self):
        """Test actual database operations with security focus."""
        # Check FTS5 availability first
        try:
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            cursor.execute("CREATE VIRTUAL TABLE test_fts USING fts5(content)")
            conn.close()
        except sqlite3.OperationalError:
            pytest.skip("FTS5 not available in this SQLite build")
        
        # Create test database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create schema
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
            
            # Insert test data
            test_data = [
                ('test_001', 'Normal test content here', 'test1.txt'),
                ('test_002', 'Another document with more content', 'test2.txt')
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
            
            # Test with lexical search plugin
            from querying.agents.plugins.lexical_search import LexicalSearchPlugin
            plugin = LexicalSearchPlugin(db_path)
            
            # Test injection attempts
            injection_queries = [
                "'; DROP TABLE chunks; --",
                "' UNION SELECT * FROM sqlite_master --",
                'test"; DELETE FROM chunks_fts; --'
            ]
            
            for query in injection_queries:
                result = plugin.execute({"query": query})
                
                # Should not cause errors (should be sanitized)
                assert "sources" in result, f"Should handle injection attempt safely: {query}"
                
                # Verify database integrity after each attempt
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Check that tables still exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                assert "chunks" in tables, "Main table should not be dropped"
                assert "chunks_fts" in tables, "FTS5 table should not be dropped"
                
                # Check that data still exists
                cursor.execute("SELECT COUNT(*) FROM chunks")
                assert cursor.fetchone()[0] == 2, "Data should not be deleted"
                
                cursor.execute("SELECT COUNT(*) FROM chunks_fts")
                assert cursor.fetchone()[0] == 2, "FTS5 data should not be deleted"
                
                conn.close()
        
        finally:
            os.unlink(db_path)

    def test_error_handling_robustness(self):
        """Test error handling doesn't leak sensitive information."""
        from querying.agents.plugins.lexical_search import LexicalSearchPlugin
        
        # Test with non-existent database
        plugin = LexicalSearchPlugin("/nonexistent/path/database.db")
        
        result = plugin.execute({"query": "test"})
        
        # Should handle error gracefully
        assert "error" in result.get("metadata", {}), "Should return error for non-existent DB"
        assert result["sources"] == [], "Should return empty sources on error"
        
        # Error message should not leak sensitive paths or system info
        error_msg = str(result.get("metadata", {}).get("error", ""))
        assert "/nonexistent/path" not in error_msg, "Error should not leak sensitive paths"


class TestHybridSearchSecurityCore:
    """Test hybrid search security aspects."""

    def test_score_normalization_security(self):
        """Test that score normalization is secure against edge cases."""
        from shared.hybrid_search import normalize_scores
        
        # Test with potentially problematic inputs
        test_cases = [
            ([float('inf')], "min-max"),
            ([float('-inf')], "min-max"), 
            ([float('nan')], "min-max"),
            ([1e308, 1e308], "min-max"),  # Very large numbers
            ([-1e308, -1e308], "min-max"),  # Very small numbers
        ]
        
        for scores, method in test_cases:
            try:
                result = normalize_scores(scores, method)
                
                # Should return valid result or handle gracefully
                assert isinstance(result, list), f"Should return list for scores {scores}"
                
                # All results should be finite numbers or empty
                for score in result:
                    assert isinstance(score, (int, float)), f"Score should be numeric: {score}"
                    if score != score:  # Check for NaN
                        pytest.fail(f"Result contains NaN for input {scores}")
                    if abs(score) == float('inf'):
                        pytest.fail(f"Result contains infinity for input {scores}")
                
            except (ValueError, OverflowError, ZeroDivisionError) as e:
                # These are acceptable errors for edge cases
                print(f"Expected error for {scores}: {e}")
            except Exception as e:
                pytest.fail(f"Unexpected error for {scores}: {e}")

    def test_merge_results_security(self):
        """Test that result merging is secure."""
        from shared.hybrid_search import merge_search_results
        
        # Test with potentially problematic inputs
        test_cases = [
            # Very large result sets
            ([(f"doc_{i}", i) for i in range(1000)], [(f"doc_{i}", i) for i in range(1000)]),
            
            # Duplicate document IDs
            ([("doc1", 0.5), ("doc1", 0.7)], [("doc1", 10.0)]),
            
            # Empty and None-like inputs
            ([], []),
            ([("doc1", 0.5)], []),
            ([], [("doc1", 10.0)]),
        ]
        
        for dense_results, lexical_results in test_cases:
            try:
                merged = merge_search_results(dense_results, lexical_results)
                
                # Should return valid result
                assert isinstance(merged, list), "Should return list"
                
                # Check for duplicates
                doc_ids = [doc_id for doc_id, _ in merged]
                assert len(doc_ids) == len(set(doc_ids)), "Should not contain duplicate document IDs"
                
                # Check score validity
                for doc_id, score in merged:
                    assert isinstance(doc_id, str), f"Document ID should be string: {doc_id}"
                    assert isinstance(score, (int, float)), f"Score should be numeric: {score}"
                    assert score >= 0, f"Score should be non-negative: {score}"
                
            except Exception as e:
                pytest.fail(f"Merging failed for inputs {len(dense_results)}, {len(lexical_results)}: {e}")

if __name__ == "__main__":
    pytest.main([__file__])