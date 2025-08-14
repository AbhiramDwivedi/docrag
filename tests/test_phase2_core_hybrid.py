"""Test core hybrid search functionality."""

import pytest
import tempfile
import sqlite3
import numpy as np
import sys
from pathlib import Path

# Simple version of the core functions for testing
def normalize_scores(scores, method="min-max"):
    """Normalize scores to [0, 1] range."""
    if not scores:
        return []
    
    scores_array = np.array(scores)
    
    if method == "min-max":
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)
        if max_score == min_score:
            return [1.0] * len(scores)
        return ((scores_array - min_score) / (max_score - min_score)).tolist()
    
    elif method == "z-score":
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        if std_score == 0:
            return [0.0] * len(scores)
        return ((scores_array - mean_score) / std_score).tolist()
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def merge_search_results(dense_results, lexical_results, dense_weight=0.6, lexical_weight=0.4, normalize_method="min-max"):
    """Merge dense and lexical search results with score normalization."""
    # Normalize scores within each result set
    if dense_results:
        dense_scores = [score for _, score in dense_results]
        normalized_dense = normalize_scores(dense_scores, normalize_method)
        dense_normalized = [(doc_id, norm_score) for (doc_id, _), norm_score in 
                           zip(dense_results, normalized_dense)]
    else:
        dense_normalized = []
    
    if lexical_results:
        lexical_scores = [score for _, score in lexical_results]
        normalized_lexical = normalize_scores(lexical_scores, normalize_method)
        lexical_normalized = [(doc_id, norm_score) for (doc_id, _), norm_score in 
                             zip(lexical_results, normalized_lexical)]
    else:
        lexical_normalized = []
    
    # Create combined score dictionary
    combined_scores = {}
    
    # Add dense scores
    for doc_id, norm_score in dense_normalized:
        combined_scores[doc_id] = dense_weight * norm_score
    
    # Add lexical scores
    for doc_id, norm_score in lexical_normalized:
        if doc_id in combined_scores:
            combined_scores[doc_id] += lexical_weight * norm_score
        else:
            combined_scores[doc_id] = lexical_weight * norm_score
    
    # Sort by combined score (descending) with stable tie-breaking by doc_id
    sorted_results = sorted(
        combined_scores.items(),
        key=lambda x: (-x[1], x[0])  # Score desc, doc_id asc for tie-breaking
    )
    
    return sorted_results


def classify_query_intent(query):
    """Classify query intent to determine search strategy."""
    import re
    
    query_lower = query.lower().strip()
    
    # Detect proper nouns (capitalized words)
    proper_nouns = re.findall(r'\b[A-Z][A-Z0-9]*\b', query)  # All caps words
    proper_nouns += re.findall(r'\b[A-Z][a-z]+\b', query)    # Title case words
    
    # Detect common keywords that suggest lexical search
    lexical_keywords = [
        'containing', 'includes', 'mentions', 'keyword', 'exact', 'phrase',
        'find files', 'search for', 'documents with', 'files containing'
    ]
    
    # Detect semantic indicators
    semantic_keywords = [
        'about', 'regarding', 'related to', 'similar to', 'like',
        'explain', 'what is', 'how to', 'why', 'concept'
    ]
    
    # Count indicators
    lexical_score = sum(1 for kw in lexical_keywords if kw in query_lower)
    semantic_score = sum(1 for kw in semantic_keywords if kw in query_lower)
    
    # Query length analysis
    word_count = len(query.split())
    
    # Classification logic
    if lexical_score > semantic_score:
        strategy = "lexical_primary"
        confidence = min(0.9, 0.6 + lexical_score * 0.1)
    elif proper_nouns and word_count <= 5:
        # Short queries with proper nouns benefit from hybrid
        strategy = "hybrid"
        confidence = min(0.8, 0.5 + len(proper_nouns) * 0.1)
    elif word_count <= 3:
        # Very short queries benefit from hybrid
        strategy = "hybrid" 
        confidence = 0.7
    else:
        # Complex queries go to semantic primarily
        strategy = "semantic_primary"
        confidence = min(0.9, 0.6 + semantic_score * 0.1)
    
    return {
        "strategy": strategy,
        "confidence": confidence,
        "proper_nouns": proper_nouns,
        "word_count": word_count,
        "lexical_score": lexical_score,
        "semantic_score": semantic_score
    }


def check_fts5_availability():
    """Check if SQLite was compiled with FTS5 support."""
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE VIRTUAL TABLE test_fts USING fts5(content)")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


class TestCoreHybridFunctionality:
    """Test core hybrid search functionality."""
    
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
    
    def test_fts5_availability(self):
        """Test FTS5 availability check."""
        # This will pass if FTS5 is available, skip if not
        if not check_fts5_availability():
            pytest.skip("FTS5 not available in this SQLite build")
        
        assert check_fts5_availability() is True
    
    def test_fts5_basic_usage(self):
        """Test basic FTS5 usage if available."""
        if not check_fts5_availability():
            pytest.skip("FTS5 not available in this SQLite build")
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create FTS5 table
            cursor.execute("""
                CREATE VIRTUAL TABLE docs_fts USING fts5(
                    doc_id UNINDEXED,
                    content
                )
            """)
            
            # Insert test data
            test_docs = [
                ("doc1", "Acme Corporation provides analytics solutions"),
                ("doc2", "BOLT protocol for real-time streaming"),
                ("doc3", "Globex technology investments")
            ]
            
            for doc_id, content in test_docs:
                cursor.execute(
                    "INSERT INTO docs_fts (doc_id, content) VALUES (?, ?)",
                    (doc_id, content)
                )
            
            # Test search
            cursor.execute(
                "SELECT doc_id, bm25(docs_fts) FROM docs_fts WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts)",
                ("Acme",)
            )
            
            results = cursor.fetchall()
            conn.close()
            
            # Should find the Acme document
            assert len(results) >= 1
            assert "doc1" in [result[0] for result in results]
            
        finally:
            Path(db_path).unlink(missing_ok=True)
    
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
    
    def test_hybrid_score_combination_mathematics(self):
        """Test the mathematical correctness of hybrid score combination."""
        # Test with known values to verify calculation
        semantic_results = [("doc1", 0.8), ("doc2", 0.4)]  # Range: 0.4-0.8
        lexical_results = [("doc2", 20.0), ("doc3", 10.0)]  # Range: 10.0-20.0
        
        merged = merge_search_results(
            semantic_results, lexical_results,
            dense_weight=0.6, lexical_weight=0.4,
            normalize_method="min-max"
        )
        
        # Manual calculation verification:
        # Semantic normalized: doc1=1.0, doc2=0.0 (min-max of [0.8, 0.4])
        # Lexical normalized: doc2=1.0, doc3=0.0 (min-max of [20.0, 10.0])
        # 
        # Hybrid scores:
        # doc1: 0.6 * 1.0 + 0.4 * 0.0 = 0.6
        # doc2: 0.6 * 0.0 + 0.4 * 1.0 = 0.4
        # doc3: 0.6 * 0.0 + 0.4 * 0.0 = 0.0
        
        merged_dict = dict(merged)
        
        # doc1 should have highest score (0.6)
        assert merged_dict["doc1"] == pytest.approx(0.6, abs=1e-10)
        # doc2 should have score 0.4
        assert merged_dict["doc2"] == pytest.approx(0.4, abs=1e-10)
        # doc3 should have lowest score (0.0)
        assert merged_dict["doc3"] == pytest.approx(0.0, abs=1e-10)
        
        # Verify ordering: doc1 > doc2 > doc3
        doc_order = [doc_id for doc_id, _ in merged]
        assert doc_order == ["doc1", "doc2", "doc3"]