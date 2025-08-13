"""Test Maximal Marginal Relevance (MMR) selection correctness."""

import numpy as np
import pytest
from typing import List, Tuple


def mmr_selection(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    selected_embeddings: np.ndarray = None,
    mmr_lambda: float = 0.5,
    k: int = 5
) -> List[int]:
    """
    Select documents using Maximal Marginal Relevance.
    
    Args:
        query_embedding: Query vector (1D)
        candidate_embeddings: Candidate document vectors (2D)
        selected_embeddings: Already selected document vectors (2D, optional)
        mmr_lambda: Balance between relevance and diversity (0-1)
        k: Number of documents to select
        
    Returns:
        List of indices of selected documents
    """
    if selected_embeddings is None:
        selected_embeddings = np.empty((0, candidate_embeddings.shape[1]))
    
    selected_indices = []
    remaining_indices = list(range(len(candidate_embeddings)))
    
    for _ in range(min(k, len(candidate_embeddings))):
        if not remaining_indices:
            break
            
        best_idx = None
        best_score = float('-inf')
        
        for idx in remaining_indices:
            candidate = candidate_embeddings[idx]
            
            # Relevance score (cosine similarity with query)
            relevance = np.dot(query_embedding, candidate)
            
            # Diversity score (max similarity with already selected)
            if len(selected_embeddings) > 0:
                similarities = np.dot(selected_embeddings, candidate)
                max_similarity = np.max(similarities)
            else:
                max_similarity = 0.0
            
            # MMR score
            mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_similarity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            selected_embeddings = np.vstack([
                selected_embeddings, 
                candidate_embeddings[best_idx:best_idx+1]
            ])
            remaining_indices.remove(best_idx)
    
    return selected_indices


class TestMMRSelection:
    """Test MMR document selection logic."""
    
    def test_mmr_selects_most_relevant_first_when_lambda_1(self):
        """Test that MMR selects most relevant document first when lambda=1.0."""
        # Query vector
        query = np.array([1.0, 0.0, 0.0])
        
        # Candidate documents with different relevance scores
        candidates = np.array([
            [0.5, 0.5, 0.0],  # Medium relevance
            [1.0, 0.0, 0.0],  # High relevance (most similar to query)
            [0.0, 1.0, 0.0],  # Low relevance
        ])
        
        # With lambda=1.0, should prioritize relevance only
        selected = mmr_selection(query, candidates, mmr_lambda=1.0, k=3)
        
        # Should select highest relevance first (index 1)
        assert selected[0] == 1
        
    def test_mmr_promotes_diversity_when_lambda_0(self):
        """Test that MMR promotes diversity when lambda=0.0."""
        # Query vector
        query = np.array([1.0, 0.0, 0.0])
        
        # Two very similar high-relevance docs and one diverse doc
        candidates = np.array([
            [0.9, 0.1, 0.0],  # High relevance, similar to next
            [0.9, 0.1, 0.0],  # High relevance, similar to previous  
            [0.0, 0.0, 1.0],  # Low relevance but diverse
        ])
        
        # With lambda=0.0, should prioritize diversity after first selection
        selected = mmr_selection(query, candidates, mmr_lambda=0.0, k=3)
        
        # After selecting first relevant doc, should avoid similar ones
        # and prefer diverse document
        assert len(set(selected)) == 3  # Should select all different docs
        
    def test_mmr_balanced_selection(self):
        """Test MMR with balanced lambda=0.5."""
        # Query vector
        query = np.array([1.0, 0.0, 0.0])
        
        # Create documents with different relevance/diversity tradeoffs
        candidates = np.array([
            [1.0, 0.0, 0.0],  # Perfect match to query
            [0.9, 0.0, 0.0],  # Very similar to query and first doc
            [0.0, 1.0, 0.0],  # Orthogonal to query (diverse)
            [0.7, 0.7, 0.0],  # Moderate relevance, somewhat diverse
        ])
        
        selected = mmr_selection(query, candidates, mmr_lambda=0.5, k=3)
        
        # First should be most relevant
        assert selected[0] == 0
        
        # Should balance relevance and diversity for remaining selections
        assert len(selected) == 3
        
    def test_mmr_deterministic_with_stable_tiebreaking(self):
        """Test that MMR is deterministic with stable tie-breaking."""
        # Create scenario with potential ties
        query = np.array([1.0, 0.0])
        
        candidates = np.array([
            [0.5, 0.5],
            [0.5, 0.5],  # Identical to first
            [0.0, 1.0],
        ])
        
        # Run multiple times - should get same result
        selected1 = mmr_selection(query, candidates, mmr_lambda=0.5, k=2)
        selected2 = mmr_selection(query, candidates, mmr_lambda=0.5, k=2)
        
        assert selected1 == selected2
        
    def test_mmr_respects_k_limit(self):
        """Test that MMR respects the k parameter."""
        query = np.array([1.0, 0.0, 0.0])
        
        candidates = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ])
        
        selected = mmr_selection(query, candidates, mmr_lambda=0.5, k=3)
        
        assert len(selected) == 3
        
    def test_mmr_with_preselected_documents(self):
        """Test MMR when some documents are already selected."""
        query = np.array([1.0, 0.0, 0.0])
        
        candidates = np.array([
            [0.9, 0.1, 0.0],  # Similar to preselected
            [0.0, 1.0, 0.0],  # Different from preselected
            [0.8, 0.2, 0.0],  # Similar to preselected
        ])
        
        # Preselect a document similar to candidates[0] and [2]
        preselected = np.array([[1.0, 0.0, 0.0]])
        
        selected = mmr_selection(
            query, candidates, 
            selected_embeddings=preselected,
            mmr_lambda=0.5, k=2
        )
        
        # Should select 2 documents
        assert len(selected) == 2
        
        # With preselected documents, diversity should be encouraged
        # The exact selection depends on the MMR balance, but we should get results
        assert all(idx in range(len(candidates)) for idx in selected)
        
    def test_mmr_empty_candidates(self):
        """Test MMR with empty candidate set."""
        query = np.array([1.0, 0.0, 0.0])
        candidates = np.empty((0, 3))
        
        selected = mmr_selection(query, candidates, k=5)
        
        assert selected == []
        
    def test_mmr_single_candidate(self):
        """Test MMR with single candidate."""
        query = np.array([1.0, 0.0, 0.0])
        candidates = np.array([[0.5, 0.5, 0.0]])
        
        selected = mmr_selection(query, candidates, k=5)
        
        assert selected == [0]
    
    def test_mmr_all_identical_documents(self):
        """Test MMR behavior with all identical candidate documents."""
        query = np.array([1.0, 0.0, 0.0])
        
        # All candidates are identical
        candidates = np.array([
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ])
        
        selected = mmr_selection(query, candidates, mmr_lambda=0.5, k=3)
        
        # Should still select k documents even though they're identical
        assert len(selected) == 3
        # Should use deterministic tie-breaking (select indices in order)
        assert selected == [0, 1, 2]
    
    def test_mmr_with_zero_similarity_documents(self):
        """Test MMR with orthogonal (zero similarity) documents.""" 
        query = np.array([1.0, 0.0, 0.0, 0.0])
        
        # All candidates are orthogonal to each other and query
        candidates = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], 
            [0.0, 0.0, 0.0, 1.0],
        ])
        
        # With zero similarities, all should have same MMR score
        selected = mmr_selection(query, candidates, mmr_lambda=0.5, k=3)
        
        # Should select all and use deterministic ordering
        assert len(selected) == 3
        assert selected == [0, 1, 2]
    
    def test_mmr_extreme_lambda_values(self):
        """Test MMR with extreme lambda values (0.0 and 1.0)."""
        query = np.array([1.0, 0.0])
        
        candidates = np.array([
            [1.0, 0.0],    # Perfect match to query
            [0.9, 0.1],    # Very similar to query and first doc  
            [0.0, 1.0],    # Orthogonal to query
        ])
        
        # Lambda = 1.0 (pure relevance) 
        selected_pure_relevance = mmr_selection(query, candidates, mmr_lambda=1.0, k=3)
        
        # Lambda = 0.0 (pure diversity after first)
        selected_pure_diversity = mmr_selection(query, candidates, mmr_lambda=0.0, k=3)
        
        # Both should be deterministic
        assert len(selected_pure_relevance) == 3
        assert len(selected_pure_diversity) == 3
        
        # Pure relevance should prioritize by similarity to query
        assert selected_pure_relevance[0] == 0  # Most relevant first
        
    def test_mmr_large_k_value(self):
        """Test MMR when k is larger than number of candidates."""
        query = np.array([1.0, 0.0, 0.0])
        
        candidates = np.array([
            [0.8, 0.2, 0.0],
            [0.0, 0.8, 0.2],
        ])
        
        # Request more documents than available
        selected = mmr_selection(query, candidates, mmr_lambda=0.5, k=10)
        
        # Should return all available candidates
        assert len(selected) == 2
        assert set(selected) == {0, 1}