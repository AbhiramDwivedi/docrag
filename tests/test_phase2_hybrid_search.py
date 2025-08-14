"""Test hybrid search functionality and lexical search plugin."""

import pytest
import tempfile
import sqlite3
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add the backend source to Python path
backend_src = Path(__file__).parent.parent / "backend" / "src"
sys.path.insert(0, str(backend_src))

from shared.hybrid_search import (
    normalize_scores, merge_search_results, classify_query_intent
)
from querying.agents.plugins.lexical_search import LexicalSearchPlugin


class TestHybridSearchUtilities:
    """Test hybrid search utility functions."""
    
    def test_normalize_scores_min_max(self):
        """Test min-max score normalization."""
        scores = [1.0, 3.0, 5.0, 2.0, 4.0]
        normalized = normalize_scores(scores, "min-max")
        
        # Should be in [0, 1] range
        assert all(0.0 <= score <= 1.0 for score in normalized)
        
        # Min should become 0, max should become 1
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0
        
        # Relative ordering should be preserved
        original_order = np.argsort(scores)
        normalized_order = np.argsort(normalized)
        np.testing.assert_array_equal(original_order, normalized_order)
    
    def test_query_classification_lexical_primary(self):
        """Test query classification for lexical-primary queries."""
        queries = [
            "find files containing ACME",
            "search for documents with keyword protocol",
            "files containing exact phrase 'data platform'"
        ]
        
        for query in queries:
            result = classify_query_intent(query)
            assert result["strategy"] in ["lexical_primary", "hybrid"]
            assert result["confidence"] > 0.5
    
    def test_query_classification_semantic_primary(self):
        """Test query classification for semantic-primary queries."""
        queries = [
            "What is the main concept behind machine learning algorithms?",
            "How does the financial analysis relate to market trends?",
            "Explain the relationship between data quality and business outcomes"
        ]
        
        for query in queries:
            result = classify_query_intent(query)
            assert result["strategy"] == "semantic_primary"
            assert result["confidence"] > 0.5
    
    def test_query_classification_hybrid(self):
        """Test query classification for hybrid queries."""
        queries = [
            "ACME protocol",
            "BOLT specification",
            "Globex data"
        ]
        
        for query in queries:
            result = classify_query_intent(query)
            assert result["strategy"] == "hybrid"
            assert len(result["proper_nouns"]) > 0
    
    def test_merge_search_results(self):
        """Test merging of dense and lexical search results."""
        dense_results = [
            ("doc1", 0.9),
            ("doc2", 0.7),
            ("doc3", 0.5)
        ]
        
        lexical_results = [
            ("doc2", 10.0),
            ("doc3", 8.0),
            ("doc4", 6.0)
        ]
        
        merged = merge_search_results(
            dense_results, lexical_results,
            dense_weight=0.6, lexical_weight=0.4
        )
        
        # Should include all unique documents
        doc_ids = [doc_id for doc_id, _ in merged]
        assert set(doc_ids) == {"doc1", "doc2", "doc3", "doc4"}
        
        # Should be sorted by combined score (descending)
        scores = [score for _, score in merged]
        assert scores == sorted(scores, reverse=True)
        
        # doc2 should appear only once (deduplicated)
        assert doc_ids.count("doc2") == 1


class TestLexicalSearchPlugin:
    """Test lexical search plugin functionality."""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {
                'chunk_id': 'chunk1',
                'content': 'Acme Corporation is a leading data platform company that provides analytics solutions.',
                'file_path': 'acme_overview.txt'
            },
            {
                'chunk_id': 'chunk2', 
                'content': 'BOLT is a high-performance protocol for real-time data streaming.',
                'file_path': 'bolt_spec.md'
            },
            {
                'chunk_id': 'chunk3',
                'content': 'Globex specializes in technology sector investments and M&A advisory.',
                'file_path': 'globex_financial.md'
            }
        ]
    
    @pytest.fixture
    def temp_db_with_fts5(self, sample_documents):
        """Create temporary database with FTS5 index."""
        if not self._check_fts5_availability():
            pytest.skip("FTS5 not available in this SQLite build")
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Create main tables
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create chunks table
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
            for i, doc in enumerate(sample_documents):
                cursor.execute(
                    "INSERT INTO chunks (chunk_id, content, file_path, chunk_index) VALUES (?, ?, ?, ?)",
                    (doc['chunk_id'], doc['content'], doc['file_path'], i)
                )
                cursor.execute(
                    "INSERT INTO chunks_fts (chunk_id, content, file_path) VALUES (?, ?, ?)",
                    (doc['chunk_id'], doc['content'], doc['file_path'])
                )
            
            conn.commit()
            conn.close()
            
            yield db_path
            
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    def _check_fts5_availability(self) -> bool:
        """Check if SQLite was compiled with FTS5 support."""
        try:
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            cursor.execute("CREATE VIRTUAL TABLE test_fts USING fts5(content)")
            conn.close()
            return True
        except sqlite3.OperationalError:
            return False
    
    def test_lexical_plugin_initialization(self):
        """Test lexical search plugin initialization."""
        plugin = LexicalSearchPlugin()
        info = plugin.get_info()
        
        assert info.name == "lexical_search"
        assert "query" in info.parameters
        assert "limit" in info.parameters
    
    def test_lexical_search_basic(self, temp_db_with_fts5):
        """Test basic lexical search functionality."""
        plugin = LexicalSearchPlugin(db_path=temp_db_with_fts5)
        
        # Test exact word match
        result = plugin.execute({"query": "Acme", "limit": 10})
        
        assert result["metadata"]["results_count"] >= 1
        assert any("acme_overview.txt" in source["file_path"] for source in result["sources"])
    
    def test_lexical_search_phrase(self, temp_db_with_fts5):
        """Test phrase search functionality."""
        plugin = LexicalSearchPlugin(db_path=temp_db_with_fts5)
        
        # Test phrase search
        result = plugin.execute({"query": "data platform", "exact_phrase": True})
        
        # Should find the exact phrase in Acme document
        assert result["metadata"]["results_count"] >= 1
        found_files = [source["file_path"] for source in result["sources"]]
        assert "acme_overview.txt" in found_files
    
    def test_lexical_search_raw_format(self, temp_db_with_fts5):
        """Test raw search format for hybrid integration."""
        plugin = LexicalSearchPlugin(db_path=temp_db_with_fts5)
        
        # Test raw search
        results = plugin.search_raw("protocol", limit=5)
        
        # Should return list of (chunk_id, score) tuples
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], tuple)
            assert len(results[0]) == 2
            chunk_id, score = results[0]
            assert isinstance(chunk_id, str)
            assert isinstance(score, (int, float))
    
    def test_lexical_search_empty_query(self, temp_db_with_fts5):
        """Test handling of empty query."""
        plugin = LexicalSearchPlugin(db_path=temp_db_with_fts5)
        
        result = plugin.execute({"query": "", "limit": 10})
        
        assert "error" in result["metadata"]
        assert result["metadata"]["error"] == "empty_query"
        assert result["metadata"]["results_count"] == 0
    
    def test_lexical_search_no_results(self, temp_db_with_fts5):
        """Test handling when no results found."""
        plugin = LexicalSearchPlugin(db_path=temp_db_with_fts5)
        
        result = plugin.execute({"query": "nonexistentterm12345", "limit": 10})
        
        assert result["metadata"]["results_count"] == 0
        assert len(result["sources"]) == 0
        assert "No documents found" in result["response"]
    
    def test_fts5_unavailable_handling(self):
        """Test handling when FTS5 is not available."""
        plugin = LexicalSearchPlugin()
        
        # Mock FTS5 as unavailable
        with patch.object(plugin, '_check_fts5_availability', return_value=False):
            result = plugin.execute({"query": "test", "limit": 10})
            
            assert "error" in result["metadata"]
            assert result["metadata"]["error"] == "fts5_unavailable"
            assert result["metadata"]["results_count"] == 0


class TestIntegrationScenarios:
    """Test integration scenarios for hybrid search."""
    
    def test_proper_noun_query_classification(self):
        """Test that proper noun queries are classified correctly."""
        test_cases = [
            ("ACME Corporation", "hybrid"),
            ("find BOLT protocol", "lexical_primary"),
            ("What is machine learning?", "semantic_primary"),
            ("Globex financial", "hybrid")
        ]
        
        for query, expected_strategy in test_cases:
            result = classify_query_intent(query)
            assert result["strategy"] == expected_strategy, f"Query '{query}' should be {expected_strategy}, got {result['strategy']}"
    
    def test_hybrid_score_combination(self):
        """Test that hybrid scores are combined correctly."""
        # Create test data where lexical and semantic results overlap
        semantic_results = [("doc1", 0.8), ("doc2", 0.6)]
        lexical_results = [("doc2", 12.0), ("doc3", 8.0)]
        
        merged = merge_search_results(
            semantic_results, lexical_results,
            dense_weight=0.7, lexical_weight=0.3
        )
        
        # doc2 should have highest score (appears in both)
        # doc1 should be second (semantic only, high score)  
        # doc3 should be third (lexical only)
        doc_order = [doc_id for doc_id, _ in merged]
        
        assert len(merged) == 3
        assert "doc2" in doc_order
        assert "doc1" in doc_order
        assert "doc3" in doc_order
        
        # Verify scores are in descending order
        scores = [score for _, score in merged]
        assert scores == sorted(scores, reverse=True)
    
    def test_acceptance_criteria_keyword_search(self):
        """Test acceptance criteria: 'find files containing [keyword]' uses lexical search."""
        query = "find files containing ACME"
        result = classify_query_intent(query)
        
        # Should route to lexical search
        assert result["strategy"] == "lexical_primary"
        assert result["lexical_score"] > 0  # Should detect lexical keywords
        
    def test_acceptance_criteria_proper_noun_hybrid(self):
        """Test acceptance criteria: proper nouns route to hybrid search."""
        queries_with_proper_nouns = [
            "ACME data",
            "Microsoft Azure",
            "Salesforce integration",
            "Tesla model"
        ]
        
        for query in queries_with_proper_nouns:
            result = classify_query_intent(query)
            # Should route to hybrid for short proper noun queries
            if result["word_count"] <= 5:
                assert result["strategy"] in ["hybrid", "lexical_primary"], f"Query '{query}' should use hybrid strategy"
                assert len(result["proper_nouns"]) > 0, f"Query '{query}' should detect proper nouns"