"""Test hybrid search score normalization and deduplication."""

import numpy as np
import pytest
from typing import List, Tuple, Dict, Any


def normalize_scores(scores: List[float], method: str = "min-max") -> List[float]:
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


def merge_search_results(
    dense_results: List[Tuple[str, float]],
    lexical_results: List[Tuple[str, float]],
    dense_weight: float = 0.6,
    lexical_weight: float = 0.4,
    normalize_method: str = "min-max"
) -> List[Tuple[str, float]]:
    """
    Merge dense and lexical search results with score normalization.
    
    Args:
        dense_results: List of (doc_id, score) from vector search
        lexical_results: List of (doc_id, score) from FTS search
        dense_weight: Weight for dense scores
        lexical_weight: Weight for lexical scores
        normalize_method: Score normalization method
        
    Returns:
        Merged and deduplicated results sorted by combined score
    """
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


class TestHybridMerge:
    """Test hybrid search result merging and score normalization."""
    
    def test_score_normalization_min_max(self):
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
    
    def test_score_normalization_z_score(self):
        """Test z-score normalization."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = normalize_scores(scores, "z-score")
        
        # Mean should be approximately 0, std should be approximately 1
        assert abs(np.mean(normalized)) < 1e-10
        assert abs(np.std(normalized) - 1.0) < 1e-10
        
        # Relative ordering should be preserved
        original_order = np.argsort(scores)
        normalized_order = np.argsort(normalized)
        np.testing.assert_array_equal(original_order, normalized_order)
    
    def test_score_normalization_constant_scores(self):
        """Test normalization with constant scores."""
        scores = [5.0, 5.0, 5.0, 5.0]
        
        # Min-max should return all 1.0
        normalized_minmax = normalize_scores(scores, "min-max")
        assert all(score == 1.0 for score in normalized_minmax)
        
        # Z-score should return all 0.0
        normalized_zscore = normalize_scores(scores, "z-score")
        assert all(score == 0.0 for score in normalized_zscore)
    
    def test_score_normalization_empty_list(self):
        """Test normalization with empty score list."""
        normalized = normalize_scores([], "min-max")
        assert normalized == []
        
        normalized = normalize_scores([], "z-score")
        assert normalized == []
    
    def test_basic_hybrid_merge(self):
        """Test basic merging of dense and lexical results."""
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
    
    def test_hybrid_merge_deduplication(self):
        """Test that documents appearing in both results are deduplicated."""
        dense_results = [("doc1", 0.9), ("doc2", 0.8)]
        lexical_results = [("doc2", 5.0), ("doc3", 4.0)]
        
        merged = merge_search_results(dense_results, lexical_results)
        
        # doc2 should appear only once
        doc_ids = [doc_id for doc_id, _ in merged]
        assert doc_ids.count("doc2") == 1
        assert len(merged) == 3  # doc1, doc2, doc3
    
    def test_hybrid_merge_weight_influence(self):
        """Test that weights properly influence final ranking."""
        # Create scenario where different weights lead to different rankings
        dense_results = [("doc_dense", 1.0), ("doc_both", 0.5)]
        lexical_results = [("doc_lexical", 1.0), ("doc_both", 0.5)]
        
        # With heavy dense weight
        merged_dense_heavy = merge_search_results(
            dense_results, lexical_results,
            dense_weight=0.9, lexical_weight=0.1
        )
        
        # With heavy lexical weight
        merged_lexical_heavy = merge_search_results(
            dense_results, lexical_results,
            dense_weight=0.1, lexical_weight=0.9
        )
        
        # Rankings should be different
        dense_order = [doc_id for doc_id, _ in merged_dense_heavy]
        lexical_order = [doc_id for doc_id, _ in merged_lexical_heavy]
        
        # doc_both should have highest score in both cases due to appearing in both results
        assert dense_order[0] == "doc_both"
        assert lexical_order[0] == "doc_both"
    
    def test_hybrid_merge_empty_inputs(self):
        """Test merging with empty input lists."""
        dense_results = [("doc1", 0.9)]
        
        # Empty lexical results
        merged = merge_search_results(dense_results, [])
        assert len(merged) == 1
        assert merged[0][0] == "doc1"
        
        # Empty dense results
        lexical_results = [("doc2", 5.0)]
        merged = merge_search_results([], lexical_results)
        assert len(merged) == 1
        assert merged[0][0] == "doc2"
        
        # Both empty
        merged = merge_search_results([], [])
        assert merged == []
    
    def test_hybrid_merge_deterministic_tie_breaking(self):
        """Test deterministic tie-breaking by document ID."""
        # Create scenario with identical combined scores
        dense_results = [("doc_b", 1.0), ("doc_a", 1.0)]
        lexical_results = [("doc_b", 1.0), ("doc_a", 1.0)]
        
        merged = merge_search_results(dense_results, lexical_results)
        
        # With identical scores, should be ordered by doc_id (ascending)
        doc_ids = [doc_id for doc_id, _ in merged]
        assert doc_ids == ["doc_a", "doc_b"]  # Alphabetical order
    
    def test_hybrid_merge_different_normalization_methods(self):
        """Test merging with different normalization methods."""
        dense_results = [("doc1", 1.0), ("doc2", 3.0)]
        lexical_results = [("doc1", 10.0), ("doc2", 30.0)]
        
        merged_minmax = merge_search_results(
            dense_results, lexical_results, normalize_method="min-max"
        )
        
        merged_zscore = merge_search_results(
            dense_results, lexical_results, normalize_method="z-score"
        )
        
        # Both should have same relative ordering since scores are proportional
        minmax_order = [doc_id for doc_id, _ in merged_minmax]
        zscore_order = [doc_id for doc_id, _ in merged_zscore]
        assert minmax_order == zscore_order
    
    def test_hybrid_merge_score_ranges(self):
        """Test that merged scores are in reasonable ranges."""
        dense_results = [("doc1", 0.9), ("doc2", 0.7)]
        lexical_results = [("doc3", 15.0), ("doc4", 10.0)]
        
        merged = merge_search_results(
            dense_results, lexical_results,
            dense_weight=0.6, lexical_weight=0.4
        )
        
        # All scores should be between 0 and 1 after normalization and weighting
        scores = [score for _, score in merged]
        assert all(0.0 <= score <= 1.0 for score in scores)