"""Cross-encoder reranking utilities for Phase 4 advanced features.

This module provides cross-encoder reranking functionality to improve precision
in document retrieval by reranking initial results using more sophisticated
semantic understanding.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker for improving precision of search results.
    
    Uses a cross-encoder model to rerank search results by computing
    direct query-document relevance scores instead of relying solely
    on embedding similarity.
    """
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2"):
        """Initialize the cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self._model = None
        self._model_available = False
        
        # Check if sentence-transformers is available
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder_class = CrossEncoder
            self._model_available = True
        except ImportError:
            logger.warning(
                "sentence-transformers not available. Cross-encoder reranking will be disabled. "
                "Install with: pip install sentence-transformers"
            )
            self._cross_encoder_class = None
    
    def _get_model(self):
        """Lazy loading of the cross-encoder model."""
        if not self._model_available:
            return None
            
        if self._model is None:
            try:
                self._model = self._cross_encoder_class(self.model_name)
                logger.info(f"Loaded cross-encoder model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model {self.model_name}: {e}")
                self._model_available = False
                return None
        
        return self._model
    
    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
        """Rerank search results using cross-encoder model.
        
        Args:
            query: The search query
            results: List of search results to rerank
            top_k: Number of top results to return after reranking
            
        Returns:
            Reranked list of results with cross-encoder scores
        """
        if not self._model_available or not results:
            logger.debug("Cross-encoder reranking skipped (model not available or no results)")
            return results[:top_k]
        
        model = self._get_model()
        if model is None:
            logger.debug("Cross-encoder model not available, returning original results")
            return results[:top_k]
        
        try:
            # Prepare query-document pairs for cross-encoder
            query_doc_pairs = []
            for result in results:
                doc_text = result.get('text', '')
                # Truncate very long documents to avoid model limits
                if len(doc_text) > 512:
                    doc_text = doc_text[:512] + "..."
                query_doc_pairs.append([query, doc_text])
            
            # Get cross-encoder scores
            if not query_doc_pairs:
                return results[:top_k]
                
            cross_encoder_scores = model.predict(query_doc_pairs)
            
            # Add cross-encoder scores to results and sort
            reranked_results = []
            for i, result in enumerate(results):
                enhanced_result = result.copy()
                enhanced_result['cross_encoder_score'] = float(cross_encoder_scores[i])
                enhanced_result['original_rank'] = i
                reranked_results.append(enhanced_result)
            
            # Sort by cross-encoder score (higher is better)
            reranked_results.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
            
            # Take top k results
            final_results = reranked_results[:top_k]
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Cross-encoder reranking: {len(results)} -> {len(final_results)} results")
                if final_results:
                    logger.debug(f"Top cross-encoder score: {final_results[0]['cross_encoder_score']:.3f}")
                    logger.debug(f"Bottom cross-encoder score: {final_results[-1]['cross_encoder_score']:.3f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
            # Fallback to original results if reranking fails
            return results[:top_k]
    
    def is_available(self) -> bool:
        """Check if cross-encoder reranking is available."""
        return self._model_available


def create_query_variants(query: str, entities: List[str]) -> List[str]:
    """Create query variants for entity expansion.
    
    Args:
        query: Original query
        entities: List of detected entities
        
    Returns:
        List of query variants including synonyms and related terms
    """
    variants = [query]  # Always include original query
    
    if not entities:
        return variants
    
    # Simple synonym expansion (can be enhanced with NLP libraries)
    entity_synonyms = {
        # Common company name variations
        "Apple": ["Apple Inc", "Apple Computer"],
        "Google": ["Alphabet", "Google Inc"],
        "Microsoft": ["MSFT", "Microsoft Corporation"],
        "Tesla": ["Tesla Motors", "Tesla Inc"],
        
        # Common technology terms
        "AI": ["Artificial Intelligence", "Machine Learning"],
        "ML": ["Machine Learning", "Artificial Intelligence"],
        "COVID": ["COVID-19", "Coronavirus", "SARS-CoV-2"],
        "USA": ["United States", "America", "US"],
        "UK": ["United Kingdom", "Britain", "Great Britain"],
    }
    
    # Generate variants by replacing entities with synonyms
    for entity in entities:
        if entity in entity_synonyms:
            for synonym in entity_synonyms[entity]:
                # Create variant by replacing entity with synonym
                variant = query.replace(entity, synonym)
                if variant != query and variant not in variants:
                    variants.append(variant)
        
        # Also try different cases
        if entity.isupper() and len(entity) > 2:
            # Try title case for acronyms
            title_case = entity.title()
            variant = query.replace(entity, title_case)
            if variant != query and variant not in variants:
                variants.append(variant)
    
    # Limit number of variants to avoid too many queries
    return variants[:5]


def expand_entity_query(query: str, detected_entities: List[str]) -> List[str]:
    """Expand queries containing entities with variants and synonyms.
    
    Args:
        query: The original query
        detected_entities: List of entities detected in the query
        
    Returns:
        List of expanded query variants
    """
    if not detected_entities:
        return [query]
    
    variants = create_query_variants(query, detected_entities)
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Query expansion for entities {detected_entities}:")
        for i, variant in enumerate(variants):
            logger.debug(f"  {i+1}. {variant}")
    
    return variants