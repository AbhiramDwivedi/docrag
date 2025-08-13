"""Maximal Marginal Relevance (MMR) selection for diverse document retrieval."""

import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def mmr_selection(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_results: List[Dict[str, Any]],
    selected_embeddings: np.ndarray = None,
    mmr_lambda: float = 0.5,
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Select documents using Maximal Marginal Relevance for diverse retrieval.
    
    Args:
        query_embedding: Query vector (1D)
        candidate_embeddings: Candidate document vectors (2D)
        candidate_results: List of result dictionaries corresponding to embeddings
        selected_embeddings: Already selected document vectors (2D, optional)
        mmr_lambda: Balance between relevance and diversity (0-1)
        k: Number of documents to select
        
    Returns:
        List of selected result dictionaries
    """
    if len(candidate_results) == 0:
        return []
    
    if len(candidate_results) != len(candidate_embeddings):
        raise ValueError("Number of results must match number of embeddings")
    
    if selected_embeddings is None:
        selected_embeddings = np.empty((0, candidate_embeddings.shape[1]))
    
    selected_results = []
    remaining_indices = list(range(len(candidate_embeddings)))
    
    logger.debug(f"Starting MMR selection: {len(candidate_results)} candidates, lambda={mmr_lambda}, k={k}")
    
    for selection_round in range(min(k, len(candidate_embeddings))):
        if not remaining_indices:
            break
            
        best_idx = None
        best_score = float('-inf')
        
        for idx in remaining_indices:
            candidate = candidate_embeddings[idx]
            
            # Relevance score (cosine similarity with query)
            # Embeddings should be normalized, so dot product = cosine similarity
            relevance = np.dot(query_embedding, candidate)
            
            # Diversity score (max similarity with already selected)
            if len(selected_embeddings) > 0:
                similarities = np.dot(selected_embeddings, candidate)
                max_similarity = np.max(similarities)
            else:
                max_similarity = 0.0
            
            # MMR score: balance relevance and diversity
            mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_similarity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected_results.append(candidate_results[best_idx])
            selected_embeddings = np.vstack([
                selected_embeddings, 
                candidate_embeddings[best_idx:best_idx+1]
            ])
            remaining_indices.remove(best_idx)
            
            logger.debug(f"MMR round {selection_round + 1}: selected idx {best_idx}, score {best_score:.4f}")
    
    logger.debug(f"MMR selection completed: selected {len(selected_results)} documents")
    return selected_results


def extract_embeddings_from_results(results: List[Dict[str, Any]], embedder_func) -> np.ndarray:
    """Extract or compute embeddings for result texts.
    
    Args:
        results: List of search results with 'text' field
        embedder_func: Function to compute embeddings for texts
        
    Returns:
        2D numpy array of embeddings
    """
    if not results:
        return np.empty((0, 384))  # Default embedding dimension
    
    texts = [result.get('text', '') for result in results]
    embeddings = embedder_func(texts)
    return np.asarray(embeddings, dtype='float32')