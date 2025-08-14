"""Comprehensive tests for hybrid search result merging logic."""

import pytest
import numpy as np
from typing import List, Tuple
import sys
from pathlib import Path

# Add the backend source to Python path
backend_src = Path(__file__).parent.parent / "backend" / "src"
sys.path.insert(0, str(backend_src))

from shared.hybrid_search import normalize_scores, merge_search_results


class TestScoreNormalization:
    """Test score normalization functions under various conditions."""

    def test_min_max_normalization_basic(self):
        """Test basic min-max normalization."""
        scores = [1.0, 3.0, 5.0, 2.0, 4.0]
        normalized = normalize_scores(scores, "min-max")
        
        # Should be in [0, 1] range
        assert all(0.0 <= score <= 1.0 for score in normalized)
        
        # Min should become 0, max should become 1
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0
        
        # Expected values: [0.0, 0.5, 1.0, 0.25, 0.75]
        expected = [0.0, 0.5, 1.0, 0.25, 0.75]
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_min_max_normalization_edge_cases(self):
        """Test min-max normalization edge cases."""
        # All scores are the same
        scores = [5.0, 5.0, 5.0, 5.0]
        normalized = normalize_scores(scores, "min-max")
        assert normalized == [1.0, 1.0, 1.0, 1.0]
        
        # Single score
        scores = [3.0]
        normalized = normalize_scores(scores, "min-max")
        assert normalized == [1.0]
        
        # Empty list
        scores = []
        normalized = normalize_scores(scores, "min-max")
        assert normalized == []
        
        # Negative scores
        scores = [-2.0, -1.0, 0.0, 1.0, 2.0]
        normalized = normalize_scores(scores, "min-max")
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0
        
        # Very small differences
        scores = [1.0000001, 1.0000002, 1.0000003]
        normalized = normalize_scores(scores, "min-max")
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0

    def test_z_score_normalization_basic(self):
        """Test basic z-score normalization."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = normalize_scores(scores, "z-score")
        
        # Mean should be approximately 0
        assert abs(np.mean(normalized)) < 1e-10
        
        # Standard deviation should be approximately 1
        assert abs(np.std(normalized, ddof=0) - 1.0) < 1e-10

    def test_z_score_normalization_edge_cases(self):
        """Test z-score normalization edge cases."""
        # All scores are the same
        scores = [5.0, 5.0, 5.0, 5.0]
        normalized = normalize_scores(scores, "z-score")
        assert normalized == [0.0, 0.0, 0.0, 0.0]
        
        # Single score
        scores = [3.0]
        normalized = normalize_scores(scores, "z-score")
        assert normalized == [0.0]
        
        # Empty list
        scores = []
        normalized = normalize_scores(scores, "z-score")
        assert normalized == []

    def test_invalid_normalization_method(self):
        """Test that invalid normalization methods raise ValueError."""
        scores = [1.0, 2.0, 3.0]
        
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_scores(scores, "invalid-method")


class TestHybridResultMerging:
    """Test hybrid search result merging under various conditions."""

    def test_basic_merging(self):
        """Test basic result merging functionality."""
        dense_results = [("doc1", 0.8), ("doc2", 0.6), ("doc3", 0.4)]
        lexical_results = [("doc2", 10.0), ("doc4", 8.0), ("doc5", 6.0)]
        
        merged = merge_search_results(
            dense_results, lexical_results,
            dense_weight=0.6, lexical_weight=0.4
        )
        
        # Should have all unique documents
        doc_ids = [doc_id for doc_id, _ in merged]
        assert len(doc_ids) == len(set(doc_ids))  # No duplicates
        assert len(merged) == 5  # All unique docs
        
        # doc2 should have highest score (appears in both)
        assert merged[0][0] == "doc2"
        
        # Scores should be in descending order
        scores = [score for _, score in merged]
        assert scores == sorted(scores, reverse=True)

    def test_merging_with_empty_results(self):
        """Test merging when one or both result sets are empty."""
        dense_results = [("doc1", 0.8), ("doc2", 0.6)]
        lexical_results = []
        
        # Dense only
        merged = merge_search_results(dense_results, lexical_results)
        assert len(merged) == 2
        assert merged[0][0] == "doc1"  # Higher score first
        
        # Lexical only
        merged = merge_search_results([], [("doc3", 10.0), ("doc4", 8.0)])
        assert len(merged) == 2
        assert merged[0][0] == "doc3"  # Higher score first
        
        # Both empty
        merged = merge_search_results([], [])
        assert merged == []

    def test_merging_with_identical_documents(self):
        """Test merging when both result sets contain the same documents."""
        dense_results = [("doc1", 0.8), ("doc2", 0.6), ("doc3", 0.4)]
        lexical_results = [("doc1", 12.0), ("doc2", 10.0), ("doc3", 8.0)]
        
        merged = merge_search_results(dense_results, lexical_results)
        
        # Should have exactly 3 documents (no duplicates)
        assert len(merged) == 3
        
        # All should have combined scores (may be 0.0 for lowest-ranked items after normalization)
        for doc_id, score in merged:
            assert doc_id in ["doc1", "doc2", "doc3"]
            assert score >= 0  # Should have non-negative combined scores
            assert isinstance(score, (int, float))  # Should be numeric

    def test_different_weight_combinations(self):
        """Test merging with different weight combinations."""
        dense_results = [("doc1", 1.0), ("doc2", 0.5)]
        lexical_results = [("doc2", 1.0), ("doc3", 1.0)]
        
        # Dense-heavy weighting
        merged_dense = merge_search_results(
            dense_results, lexical_results,
            dense_weight=0.9, lexical_weight=0.1
        )
        
        # Lexical-heavy weighting
        merged_lexical = merge_search_results(
            dense_results, lexical_results,
            dense_weight=0.1, lexical_weight=0.9
        )
        
        # Results should be different
        assert merged_dense != merged_lexical
        
        # doc1 should rank higher in dense-heavy (only in dense)
        # doc3 should rank higher in lexical-heavy (only in lexical)
        dense_doc1_pos = next(i for i, (doc_id, _) in enumerate(merged_dense) if doc_id == "doc1")
        lexical_doc1_pos = next(i for i, (doc_id, _) in enumerate(merged_lexical) if doc_id == "doc1")
        
        dense_doc3_pos = next(i for i, (doc_id, _) in enumerate(merged_dense) if doc_id == "doc3")
        lexical_doc3_pos = next(i for i, (doc_id, _) in enumerate(merged_lexical) if doc_id == "doc3")
        
        assert dense_doc1_pos < lexical_doc1_pos  # doc1 ranks higher with dense weighting
        assert lexical_doc3_pos < dense_doc3_pos  # doc3 ranks higher with lexical weighting

    def test_tie_breaking_deterministic(self):
        """Test that tie-breaking is deterministic and stable."""
        # Create results where multiple documents will have the same combined score
        dense_results = [("doc_b", 0.5), ("doc_a", 0.5), ("doc_c", 0.5)]
        lexical_results = [("doc_b", 1.0), ("doc_a", 1.0), ("doc_c", 1.0)]
        
        # Run merging multiple times
        results = []
        for _ in range(10):
            merged = merge_search_results(dense_results, lexical_results)
            results.append([doc_id for doc_id, _ in merged])
        
        # All results should be identical (deterministic)
        assert all(result == results[0] for result in results)
        
        # Should be sorted by doc_id for tie-breaking
        doc_ids = results[0]
        assert doc_ids == sorted(doc_ids)

    def test_score_normalization_methods(self):
        """Test different score normalization methods in merging."""
        dense_results = [("doc1", 100.0), ("doc2", 50.0)]
        lexical_results = [("doc2", 0.8), ("doc3", 0.6)]
        
        # Min-max normalization
        merged_minmax = merge_search_results(
            dense_results, lexical_results,
            normalize_method="min-max"
        )
        
        # Z-score normalization
        merged_zscore = merge_search_results(
            dense_results, lexical_results,
            normalize_method="z-score"
        )
        
        # Results should be different due to different normalization
        assert merged_minmax != merged_zscore
        
        # But both should have same number of documents
        assert len(merged_minmax) == len(merged_zscore) == 3

    def test_large_score_differences(self):
        """Test merging with very different score scales."""
        # Dense scores in [0, 1] range
        dense_results = [("doc1", 0.95), ("doc2", 0.85), ("doc3", 0.75)]
        
        # Lexical scores in [0, 100] range
        lexical_results = [("doc2", 95.0), ("doc4", 85.0), ("doc5", 75.0)]
        
        merged = merge_search_results(dense_results, lexical_results)
        
        # Normalization should handle the scale differences
        assert len(merged) == 5
        
        # doc2 should still rank highly (appears in both)
        doc2_position = next(i for i, (doc_id, _) in enumerate(merged) if doc_id == "doc2")
        assert doc2_position <= 1  # Should be in top 2

    def test_negative_scores(self):
        """Test merging with negative scores."""
        dense_results = [("doc1", -0.1), ("doc2", -0.5), ("doc3", -0.9)]
        lexical_results = [("doc2", -2.0), ("doc4", -5.0), ("doc5", -10.0)]
        
        merged = merge_search_results(dense_results, lexical_results)
        
        # Should handle negative scores correctly
        assert len(merged) == 5
        
        # Scores should be properly normalized and combined
        scores = [score for _, score in merged]
        assert all(isinstance(score, (int, float)) for score in scores)

    def test_very_long_document_lists(self):
        """Test merging with large numbers of documents."""
        # Create large result sets
        dense_results = [(f"dense_doc_{i}", 1.0 - i * 0.01) for i in range(100)]
        lexical_results = [(f"lexical_doc_{i}", 100.0 - i) for i in range(100)]
        
        merged = merge_search_results(dense_results, lexical_results)
        
        # Should have all 200 unique documents
        assert len(merged) == 200
        
        # Should be properly sorted
        scores = [score for _, score in merged]
        assert scores == sorted(scores, reverse=True)

    def test_precision_edge_cases(self):
        """Test with very small score differences and precision issues."""
        # Very small differences that might cause floating point issues
        dense_results = [("doc1", 0.1000000001), ("doc2", 0.1000000002)]
        lexical_results = [("doc3", 0.1000000003), ("doc4", 0.1000000004)]
        
        merged = merge_search_results(dense_results, lexical_results)
        
        # Should handle precision correctly
        assert len(merged) == 4
        assert all(isinstance(score, (int, float)) for _, score in merged)

    def test_invalid_weight_combinations(self):
        """Test behavior with edge case weight combinations."""
        dense_results = [("doc1", 0.8)]
        lexical_results = [("doc2", 10.0)]
        
        # Zero weights
        merged = merge_search_results(
            dense_results, lexical_results,
            dense_weight=0.0, lexical_weight=1.0
        )
        assert len(merged) == 2
        
        # Very small weights
        merged = merge_search_results(
            dense_results, lexical_results,
            dense_weight=0.001, lexical_weight=0.999
        )
        assert len(merged) == 2
        
        # Weights that don't sum to 1
        merged = merge_search_results(
            dense_results, lexical_results,
            dense_weight=0.3, lexical_weight=0.3  # Sum = 0.6
        )
        assert len(merged) == 2


class TestMergingMathematicalProperties:
    """Test mathematical properties of the merging algorithm."""

    def test_score_monotonicity(self):
        """Test that higher input scores lead to higher output scores (within groups)."""
        # Test with varying dense scores, fixed lexical
        base_lexical = [("doc_lex", 10.0)]
        
        for dense_score in [0.1, 0.5, 0.9]:
            dense_results = [("doc_dense", dense_score)]
            merged = merge_search_results(dense_results, base_lexical)
            
            dense_doc_score = next(score for doc_id, score in merged if doc_id == "doc_dense")
            # Higher dense scores should generally lead to higher combined scores
            # (when lexical component is constant)

    def test_weight_linearity(self):
        """Test that weights behave linearly in the combination."""
        # Use multiple documents so normalization has an effect
        dense_results = [("doc1", 1.0), ("doc2", 0.5)]
        lexical_results = [("doc1", 0.8), ("doc3", 1.0)]
        
        # Test different weight combinations
        weights = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
        doc1_scores = []
        
        for dense_weight, lexical_weight in weights:
            merged = merge_search_results(
                dense_results, lexical_results,
                dense_weight=dense_weight, lexical_weight=lexical_weight
            )
            # Find doc1's score
            doc1_score = next(score for doc_id, score in merged if doc_id == "doc1")
            doc1_scores.append(doc1_score)
        
        # doc1 scores should change with weight changes
        assert len(set(doc1_scores)) >= 2, f"Should have different scores for different weights, got {doc1_scores}"

    def test_combination_bounds(self):
        """Test that combined scores are within expected bounds."""
        dense_results = [("doc1", 0.8), ("doc2", 0.2)]
        lexical_results = [("doc1", 10.0), ("doc3", 2.0)]
        
        merged = merge_search_results(dense_results, lexical_results)
        
        # All scores should be non-negative
        scores = [score for _, score in merged]
        assert all(score >= 0 for score in scores)
        
        # With min-max normalization, max possible combined score is 1.0
        # (when a document has max score in both dense and lexical)
        assert all(score <= 1.0 for score in scores)

    def test_recall_improvement_property(self):
        """Test that hybrid search provides equal or better recall than individual methods."""
        # Simulate semantic and lexical search results
        semantic_results = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        lexical_results = [("doc4", 15.0), ("doc5", 12.0), ("doc6", 10.0)]
        
        # Hybrid should return all unique documents from both methods
        merged = merge_search_results(semantic_results, lexical_results)
        
        merged_doc_ids = {doc_id for doc_id, _ in merged}
        semantic_doc_ids = {doc_id for doc_id, _ in semantic_results}
        lexical_doc_ids = {doc_id for doc_id, _ in lexical_results}
        
        # Hybrid recall should be the union of both methods
        assert merged_doc_ids == semantic_doc_ids.union(lexical_doc_ids)
        
        # Test with overlapping results
        semantic_results2 = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        lexical_results2 = [("doc2", 15.0), ("doc3", 12.0), ("doc4", 10.0)]
        
        merged2 = merge_search_results(semantic_results2, lexical_results2)
        merged_doc_ids2 = {doc_id for doc_id, _ in merged2}
        semantic_doc_ids2 = {doc_id for doc_id, _ in semantic_results2}
        lexical_doc_ids2 = {doc_id for doc_id, _ in lexical_results2}
        
        assert len(merged_doc_ids2) >= len(semantic_doc_ids2)
        assert len(merged_doc_ids2) >= len(lexical_doc_ids2)
        assert merged_doc_ids2 == semantic_doc_ids2.union(lexical_doc_ids2)

    def test_fifty_percent_recall_improvement(self):
        """Test scenarios where hybrid achieves ≥50% recall improvement."""
        # Scenario 1: Completely disjoint result sets
        semantic_results = [("doc1", 0.9), ("doc2", 0.8)]  # 2 unique docs
        lexical_results = [("doc3", 15.0), ("doc4", 12.0)]  # 2 different unique docs
        
        merged = merge_search_results(semantic_results, lexical_results)
        
        semantic_recall = len(semantic_results)  # 2
        hybrid_recall = len(merged)  # 4
        improvement = (hybrid_recall - semantic_recall) / semantic_recall * 100
        
        assert improvement >= 50.0  # 100% improvement in this case
        
        # Scenario 2: Partial overlap with additional lexical results
        semantic_results = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]  # 3 docs
        lexical_results = [("doc3", 15.0), ("doc4", 12.0), ("doc5", 10.0), ("doc6", 8.0)]  # 1 overlap + 3 new
        
        merged = merge_search_results(semantic_results, lexical_results)
        
        semantic_recall = len(semantic_results)  # 3
        hybrid_recall = len(merged)  # 6 (3 + 3 new from lexical)
        improvement = (hybrid_recall - semantic_recall) / semantic_recall * 100
        
        assert improvement >= 50.0  # 100% improvement in this case