"""Hybrid search utilities for combining dense and lexical search results."""

import numpy as np
from typing import List, Tuple, Dict, Any


def normalize_scores(scores: List[float], method: str = "min-max") -> List[float]:
    """Normalize scores to [0, 1] range with robust handling of edge cases."""
    if not scores:
        return []
    
    scores_array = np.array(scores)
    
    # Handle NaN and infinite values
    finite_mask = np.isfinite(scores_array)
    if not np.any(finite_mask):
        # All values are non-finite, return zeros
        return [0.0] * len(scores)
    
    if method == "min-max":
        finite_scores = scores_array[finite_mask]
        min_score = np.min(finite_scores)
        max_score = np.max(finite_scores)
        
        if max_score == min_score:
            return [1.0 if finite_mask[i] else 0.0 for i in range(len(scores))]
        
        normalized = np.zeros_like(scores_array)
        normalized[finite_mask] = (finite_scores - min_score) / (max_score - min_score)
        # Non-finite values become 0.0
        normalized[~finite_mask] = 0.0
        
        return normalized.tolist()
    
    elif method == "z-score":
        finite_scores = scores_array[finite_mask]
        mean_score = np.mean(finite_scores)
        std_score = np.std(finite_scores)
        
        if std_score == 0:
            return [0.0] * len(scores)
        
        normalized = np.zeros_like(scores_array)
        normalized[finite_mask] = (finite_scores - mean_score) / std_score
        # Non-finite values become 0.0
        normalized[~finite_mask] = 0.0
        
        return normalized.tolist()
    
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


def classify_query_intent(query: str) -> Dict[str, Any]:
    """
    Classify query intent to determine search strategy.
    
    Args:
        query: The search query string
        
    Returns:
        Dictionary with classification results:
        - strategy: "hybrid", "semantic_primary", or "lexical_primary"
        - confidence: confidence score (0-1)
        - proper_nouns: list of detected proper nouns
        - keywords: list of detected keywords
    """
    import re
    
    query_lower = query.lower().strip()
    
    # Detect proper nouns (capitalized words) but exclude common question/sentence starters
    question_words = {'what', 'who', 'where', 'when', 'why', 'how', 'which', 'the', 'a', 'an', 
                     'explain', 'describe', 'tell', 'show', 'give', 'provide', 'list', 'find'}
    proper_nouns = re.findall(r'\b[A-Z][A-Z0-9]*\b', query)  # All caps words
    title_case_words = re.findall(r'\b[A-Z][a-z]+\b', query)  # Title case words
    
    # Filter out question words and common sentence starters
    proper_nouns += [word for word in title_case_words if word.lower() not in question_words]
    
    # Detect common keywords that suggest lexical search
    lexical_phrases = [
        'containing', 'includes', 'mentions', 'keyword', 'exact', 'phrase',
        'find files', 'search for', 'documents with', 'files containing'
    ]
    
    # Individual words that suggest lexical search
    lexical_words = ['find', 'locate', 'search', 'contains']
    
    # Detect semantic indicators
    semantic_keywords = [
        'about', 'regarding', 'related to', 'similar to', 'like',
        'explain', 'what is', 'how to', 'why', 'concept'
    ]
    
    # Count indicators
    lexical_score = sum(1 for kw in lexical_phrases if kw in query_lower)
    lexical_score += sum(1 for word in lexical_words if word in query_lower.split())
    
    # Extra weight for "find" at the beginning of query
    if query_lower.startswith('find '):
        lexical_score += 1
    
    semantic_score = sum(1 for kw in semantic_keywords if kw in query_lower)
    
    # Query length analysis
    word_count = len(query.split())
    
    # Classification logic
    if lexical_score > semantic_score:
        strategy = "lexical_primary"
        confidence = min(0.9, 0.6 + lexical_score * 0.1)
    elif proper_nouns:
        # Queries with proper nouns benefit from hybrid search for exact matching
        strategy = "hybrid"
        confidence = min(0.8, 0.5 + len(proper_nouns) * 0.15)
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