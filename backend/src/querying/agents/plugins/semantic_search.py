"""Semantic search plugin for DocQuest agent framework.

This plugin wraps the existing vector search functionality to provide
semantic document search capabilities through the agent framework.
"""

import logging
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from ..plugin import Plugin, PluginInfo

# Try different import approaches for the dependencies
try:
    from ingestion.processors.embedder import embed_texts
    from ingestion.storage.vector_store import VectorStore
    from shared.config import settings
    from shared.mmr import mmr_selection, extract_embeddings_from_results
    from shared.hybrid_search import merge_search_results, classify_query_intent
    from shared.reranking import CrossEncoderReranker, expand_entity_query
except ImportError:
    # Fallback to absolute imports
    try:
        from src.ingestion.processors.embedder import embed_texts
        from src.ingestion.storage.vector_store import VectorStore
        from src.shared.config import settings
        from src.shared.mmr import mmr_selection, extract_embeddings_from_results
        from src.shared.hybrid_search import merge_search_results, classify_query_intent
        from src.shared.reranking import CrossEncoderReranker, expand_entity_query
    except ImportError:
        # Last resort - relative imports from backend root
        import os
        backend_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(backend_root))
        from src.ingestion.processors.embedder import embed_texts
        from src.ingestion.storage.vector_store import VectorStore
        from src.shared.config import settings
        from src.shared.mmr import mmr_selection, extract_embeddings_from_results
        from src.shared.hybrid_search import merge_search_results, classify_query_intent
        from src.shared.reranking import CrossEncoderReranker, expand_entity_query

from openai import OpenAI

logger = logging.getLogger(__name__)


class SemanticSearchPlugin(Plugin):
    """Plugin for semantic document search using vector embeddings with MMR and robustness improvements.
    
    This plugin provides enhanced semantic search functionality with:
    - Cosine similarity with normalized embeddings
    - Maximal Marginal Relevance (MMR) for diverse results
    - Proper noun and entity name query heuristics
    - Configurable retrieval parameters
    - Detailed structured logging
    """
    
    def __init__(self):
        """Initialize the semantic search plugin."""
        self._vector_store = None
        self._openai_client = None
        self._lexical_plugin = None
        self._cross_encoder = None
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced semantic search with Phase 4 advanced features.
        
        Implements multi-stage search process with optional advanced features:
        1. Query Analysis - Detect entity/proper noun queries and classify intent
        2. Query Expansion - Expand entity queries with variants (Phase 4)
        3. Strategy Selection - Choose semantic-only, lexical-only, or hybrid approach
        4. Search Execution - Run appropriate search strategy
        5. Cross-encoder Reranking - Rerank results for better precision (Phase 4)
        6. Result Merging - Combine and deduplicate results if hybrid
        7. MMR Selection - Select diverse relevant chunks
        8. Response Generation - Synthesize answer with attribution
        
        Args:
            params: Dictionary containing:
                - question: The question to search for
                - k: Number of initial chunks to retrieve (default: from config)
                - mmr_lambda: MMR balance parameter (default: from config)
                - mmr_k: Final number of chunks after MMR (default: from config)
                - enable_mmr: Whether to use MMR selection (default: True)
                - include_metadata_boost: Include metadata/filename boosting (default: auto-detect)
                - force_semantic_only: Force semantic search only (default: False)
                - force_hybrid: Force hybrid search (default: False)
                - enable_cross_encoder: Override config for cross-encoder reranking (Phase 4)
                - enable_query_expansion: Override config for query expansion (Phase 4)
                
        Returns:
            Dictionary containing:
                - response: The synthesized answer with source attribution
                - sources: List of source documents with detailed attribution
                - metadata: Search metadata and pipeline decisions
        """
        question = params.get("question", "")
        k = params.get("k", settings.retrieval_k)
        mmr_lambda = params.get("mmr_lambda", settings.mmr_lambda)
        mmr_k = params.get("mmr_k", settings.mmr_k)
        enable_mmr = params.get("enable_mmr", True)
        include_metadata_boost = params.get("include_metadata_boost", None)  # Auto-detect if None
        force_semantic_only = params.get("force_semantic_only", False)
        force_hybrid = params.get("force_hybrid", False)
        
        # Phase 4: Advanced features
        enable_cross_encoder = params.get("enable_cross_encoder", getattr(settings, 'enable_cross_encoder_reranking', False))
        enable_query_expansion = params.get("enable_query_expansion", getattr(settings, 'enable_query_expansion', False))
        
        if not question.strip():
            return {
                "response": "No question provided.",
                "sources": [],
                "metadata": {"error": "empty_question"}
            }
        
        # Stage 1: Query Analysis and Strategy Selection
        query_analysis = self._analyze_query(question)
        
        # Determine search strategy
        use_hybrid = False
        search_strategy = "semantic_only"
        
        if force_semantic_only:
            search_strategy = "semantic_only"
        elif force_hybrid:
            use_hybrid = True
            search_strategy = "hybrid"
        elif getattr(settings, 'enable_hybrid_search', True) and getattr(settings, 'enable_query_classification', True):
            # Use query classification to determine strategy
            intent_analysis = classify_query_intent(question)
            if intent_analysis["strategy"] in ["hybrid", "lexical_primary"]:
                use_hybrid = True
                search_strategy = intent_analysis["strategy"]
                query_analysis.update(intent_analysis)
        
        if include_metadata_boost is None:
            include_metadata_boost = query_analysis.get("likely_entity_query", False)
            
        if settings.enable_debug_logging:
            logger.info(f"Query analysis: {query_analysis}")
            logger.info(f"Search strategy: {search_strategy}, use_hybrid: {use_hybrid}")
            logger.info(f"Using k={k}, mmr_lambda={mmr_lambda}, mmr_k={mmr_k}, metadata_boost={include_metadata_boost}")
        
        try:
            # Check OpenAI API key
            if not settings.openai_api_key or settings.openai_api_key == "your-openai-api-key-here":
                return {
                    "response": ("❌ OpenAI API key not configured.\n"
                               "Please set your API key in config/config.yaml:\n"
                               "1. Get your API key from: https://platform.openai.com/api-keys\n"
                               "2. Edit config/config.yaml and replace 'your-openai-api-key-here' with your actual key\n"
                               "3. Alternatively, set the OPENAI_API_KEY environment variable"),
                    "sources": [],
                    "metadata": {"error": "missing_api_key"}
                }
            
            # Initialize vector store if needed
            if self._vector_store is None:
                # Use 768 dimensions for E5 model (intfloat/e5-base-v2)
                # TODO: Auto-detect dimension from model or index
                vector_dim = 768 if settings.embed_model == "intfloat/e5-base-v2" else 384
                self._vector_store = VectorStore.load(
                    Path(settings.vector_path), 
                    Path(settings.db_path),
                    dim=vector_dim
                )
            
            # Get query embedding for MMR (using original question, not expanded variants)
            q_vec = embed_texts([question], settings.embed_model)[0]
            
            # Phase 4: Query expansion for entity queries
            queries_to_search = [question]
            if enable_query_expansion and query_analysis.get("likely_entity_query", False):
                expanded_queries = expand_entity_query(question, query_analysis.get("detected_entities", []))
                queries_to_search = expanded_queries
                if settings.enable_debug_logging:
                    logger.info(f"Query expansion enabled: {len(queries_to_search)} variants")
            
            # Stage 2: Execute search strategy (with optional expansion)
            all_results = []
            for search_query in queries_to_search:
                if use_hybrid:
                    query_results = self._execute_hybrid_search(search_query, k, query_analysis)
                else:
                    query_results = self._execute_semantic_search(search_query, k, query_analysis)
                
                # Mark results with query variant info
                for result in query_results:
                    result['search_query'] = search_query
                    result['is_expanded_query'] = search_query != question
                
                all_results.extend(query_results)
            
            # Deduplicate results by chunk_id while preserving best scores
            if len(queries_to_search) > 1:
                initial_results = self._deduplicate_results(all_results)
                if settings.enable_debug_logging:
                    logger.info(f"After deduplication: {len(all_results)} -> {len(initial_results)} results")
            else:
                initial_results = all_results
            
            if not initial_results:
                return {
                    "response": ("No relevant documents found in the knowledge base. "
                               "Try rephrasing your question or checking if documents have been properly ingested."),
                    "sources": [],
                    "metadata": {
                        "results_count": 0, 
                        "stage": "search_failed",
                        "search_strategy": search_strategy,
                        "query_analysis": query_analysis
                    }
                }
            
            # Phase 4: Cross-encoder reranking for improved precision
            cross_encoder_success = False
            if enable_cross_encoder and initial_results:
                try:
                    if self._cross_encoder is None:
                        cross_encoder_model = getattr(settings, 'cross_encoder_model', 'ms-marco-MiniLM-L-6-v2')
                        self._cross_encoder = CrossEncoderReranker(cross_encoder_model)
                    
                    if self._cross_encoder.is_available():
                        cross_encoder_top_k = getattr(settings, 'cross_encoder_top_k', 20)
                        # Limit reranking to reasonable number to avoid performance issues
                        rerank_input = initial_results[:100]  # Top 100 for reranking
                        reranked_results = self._cross_encoder.rerank(question, rerank_input, cross_encoder_top_k)
                        
                        # Validate reranking results before using them
                        if reranked_results and len(reranked_results) > 0:
                            initial_results = reranked_results
                            cross_encoder_success = True
                            if settings.enable_debug_logging:
                                logger.info(f"Cross-encoder reranking successful: {len(rerank_input)} -> {len(initial_results)} results")
                        else:
                            if settings.enable_debug_logging:
                                logger.warning("Cross-encoder reranking returned empty results, using original results")
                    else:
                        if settings.enable_debug_logging:
                            logger.info("Cross-encoder reranking requested but model not available, using original results")
                            
                except Exception as e:
                    logger.warning(f"Cross-encoder reranking failed: {e}. Falling back to original results.")
                    # Continue with original results - don't let reranking failure break the search
                
                # Add reranking metadata
                if not cross_encoder_success and settings.enable_debug_logging:
                    logger.info("Cross-encoder reranking was requested but failed or unavailable - results may be less precise")
            
            # Phase 4: Entity-aware boosting
            if getattr(settings, 'enable_entity_indexing', False) and query_analysis.get("detected_entities"):
                try:
                    initial_results = self._apply_entity_boosting(initial_results, query_analysis.get("detected_entities", []))
                    if settings.enable_debug_logging:
                        logger.info("Applied entity-aware boosting to results")
                except Exception as e:
                    if settings.enable_debug_logging:
                        logger.warning(f"Entity boosting failed: {e}")
            
            # Check for low-quality results - if all results have very low similarity, provide helpful response
            if initial_results:
                max_similarity = max(result.get('similarity', 0.0) for result in initial_results)
                min_similarity_threshold = getattr(settings, 'min_similarity_threshold', 0.1)
                
                if max_similarity < min_similarity_threshold:
                    if settings.enable_debug_logging:
                        logger.info(f"Low quality results detected. Max similarity: {max_similarity:.3f} < threshold: {min_similarity_threshold}")
                    
                    return {
                        "response": (
                            f"Found documents but they don't seem highly relevant to your question. "
                            f"The most similar document has a relevance score of {max_similarity:.2%}. "
                            f"You might want to rephrase your question or try different keywords."
                        ),
                        "sources": [
                            {
                                "file": result.get("file", "unknown"),
                                "similarity": result.get("similarity", 0.0),
                                "preview": result.get("text", "")[:200] + "..." if len(result.get("text", "")) > 200 else result.get("text", "")
                            }
                            for result in initial_results[:3]  # Show top 3 low-quality matches
                        ],
                        "metadata": {
                            "results_count": len(initial_results),
                            "max_similarity": max_similarity,
                            "threshold": min_similarity_threshold,
                            "stage": "low_quality_results",
                            "search_strategy": search_strategy,
                            "query_analysis": query_analysis
                        }
                    }
            
            # Stage 3: Optional Metadata Boosting
            if include_metadata_boost:
                if settings.enable_debug_logging:
                    logger.info("Applying metadata boosting for entity query")
                initial_results = self._apply_metadata_boosting(initial_results, question, query_analysis)
            
            # Stage 4: MMR Selection for diverse results
            final_results = initial_results
            if enable_mmr and len(initial_results) > mmr_k:
                # Adjust MMR parameters for hybrid search to preserve lexical matches
                adjusted_lambda = mmr_lambda
                if use_hybrid:
                    # For hybrid search, favor relevance over diversity to preserve lexical matches
                    adjusted_lambda = min(0.9, mmr_lambda + 0.2)  # Increase toward relevance
                    if settings.enable_debug_logging:
                        logger.info(f"Adjusting MMR lambda for hybrid search: {mmr_lambda} -> {adjusted_lambda}")
                
                # For entity queries with hybrid search, preserve top lexical matches
                preserve_top_k = 0
                if use_hybrid and query_analysis.get("likely_entity_query", False):
                    # Preserve top 5 hybrid results (likely lexical matches) before MMR
                    preserve_top_k = min(5, mmr_k // 2)
                    if settings.enable_debug_logging:
                        logger.info(f"Preserving top {preserve_top_k} hybrid results for entity query")
                
                if preserve_top_k > 0:
                    # Split into preserved and MMR-selected results
                    preserved_results = initial_results[:preserve_top_k]
                    remaining_results = initial_results[preserve_top_k:]
                    remaining_k = mmr_k - preserve_top_k
                    
                    if remaining_k > 0 and remaining_results:
                        # Apply MMR to remaining results
                        candidate_embeddings = extract_embeddings_from_results(
                            remaining_results, 
                            lambda texts: embed_texts(texts, settings.embed_model)
                        )
                        
                        mmr_selected = mmr_selection(
                            query_embedding=q_vec,
                            candidate_embeddings=candidate_embeddings,
                            candidate_results=remaining_results,
                            mmr_lambda=adjusted_lambda,
                            k=remaining_k
                        )
                        
                        final_results = preserved_results + mmr_selected
                    else:
                        final_results = preserved_results
                else:
                    # Standard MMR selection
                    if settings.enable_debug_logging:
                        logger.info(f"Starting MMR selection: {len(initial_results)} -> {mmr_k} with lambda={adjusted_lambda}")
                    
                    # Get embeddings for MMR selection
                    candidate_embeddings = extract_embeddings_from_results(
                        initial_results, 
                        lambda texts: embed_texts(texts, settings.embed_model)
                    )
                    
                    # Apply MMR selection
                    final_results = mmr_selection(
                        query_embedding=q_vec,
                        candidate_embeddings=candidate_embeddings,
                        candidate_results=initial_results,
                        mmr_lambda=adjusted_lambda,
                        k=mmr_k
                    )
                
                if settings.enable_debug_logging:
                    logger.info(f"MMR selection completed: {len(final_results)} results selected")
            
            # Stage 5: Build context and generate response
            enhanced_context = self._build_enhanced_context(final_results)
            
            prompt = f"""Answer the question using the context below from multiple documents. Always cite the specific source documents and sections when providing information.

Context from Documents:
{enhanced_context}

Question: {question}

Instructions:
- Provide a comprehensive answer using information from the relevant documents
- Always cite specific source documents when making claims
- If information comes from multiple documents, acknowledge this
- Include document titles and section references where helpful
- If the context doesn't contain sufficient information, clearly state this

Answer:"""
            
            # Initialize OpenAI client if needed
            if self._openai_client is None:
                self._openai_client = OpenAI(api_key=settings.openai_api_key)
            
            # Get response from OpenAI
            resp = self._openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=1024,
            )
            
            answer = resp.choices[0].message.content
            
            # Build source documents list
            source_documents = []
            for result in final_results:
                source_documents.append({
                    "file": result.get("file", "unknown"),
                    "unit": result.get("unit", "unknown"),
                    "distance": result.get("distance", 0.0),
                    "similarity": result.get("similarity", 0.0),
                    "document_path": result.get("document_path"),
                    "document_title": result.get("document_title"),
                    "document_id": result.get("document_id"),
                    "chunk_index": result.get("chunk_index"),
                    "metadata_boosted": result.get("metadata_boosted", False)
                })
            
            search_metadata = {
                "initial_results_count": len(initial_results),
                "final_results_count": len(final_results),
                "mmr_enabled": enable_mmr,
                "mmr_lambda": mmr_lambda if enable_mmr else None,
                "metadata_boost_enabled": include_metadata_boost,
                "query_analysis": query_analysis,
                "search_strategy": search_strategy,
                "model_used": "gpt-4o-mini",
                "embedding_model": settings.embed_model,
                "context_length": len(enhanced_context),
                "retrieval_k": k,
                # Phase 4 metadata
                "cross_encoder_enabled": enable_cross_encoder,
                "cross_encoder_available": self._cross_encoder.is_available() if self._cross_encoder else False,
                "query_expansion_enabled": enable_query_expansion,
                "expanded_queries_count": len(queries_to_search),
                "phase4_features": {
                    "cross_encoder_reranking": enable_cross_encoder,
                    "query_expansion": enable_query_expansion and query_analysis.get("likely_entity_query", False),
                    "entity_indexing": getattr(settings, 'enable_entity_indexing', False)
                }
            }
            
            return {
                "response": answer,
                "sources": source_documents,
                "metadata": search_metadata
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced semantic search: {e}")
            return {
                "response": f"❌ Error calling OpenAI API: {e}\nPlease check your API key in config/config.yaml",
                "sources": [],
                "metadata": {"error": str(e), "query_analysis": query_analysis, "search_strategy": search_strategy}
            }
    
    def _execute_semantic_search(self, question: str, k: int, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute semantic search only.
        
        Args:
            question: The search question
            k: Number of chunks to retrieve
            query_analysis: Query analysis results
            
        Returns:
            List of search results
        """
        if settings.enable_debug_logging:
            logger.info(f"Starting semantic search with k={k}")
            
        q_vec = embed_texts([question], settings.embed_model)[0]
        initial_results = self._vector_store.query(q_vec, k=k)
        
        if settings.enable_debug_logging:
            logger.info(f"Semantic search returned {len(initial_results)} results")
            if initial_results:
                logger.info(f"Top result similarity: {initial_results[0].get('similarity', 'N/A')}")
        
        if not initial_results:
            # Try with expanded retrieval as fallback
            if settings.enable_debug_logging:
                logger.info("No results found, attempting expanded retrieval with higher k")
            
            # Try with double the k to catch more marginal matches
            fallback_k = min(k * 2, 200)  # Cap at reasonable limit
            initial_results = self._vector_store.query(q_vec, k=fallback_k)
            
            if settings.enable_debug_logging:
                logger.info(f"Fallback retrieval with k={fallback_k} returned {len(initial_results)} results")
        
        return initial_results
    
    def _execute_hybrid_search(self, question: str, k: int, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute hybrid semantic + lexical search.
        
        Args:
            question: The search question
            k: Number of chunks to retrieve from each method
            query_analysis: Query analysis results
            
        Returns:
            List of merged and deduplicated search results
        """
        if settings.enable_debug_logging:
            logger.info(f"Starting hybrid search with k={k}")
        
        # Execute semantic search
        q_vec = embed_texts([question], settings.embed_model)[0]
        semantic_results = self._vector_store.query(q_vec, k=k)
        
        if settings.enable_debug_logging:
            logger.info(f"Semantic component returned {len(semantic_results)} results")
        
        # Execute lexical search
        lexical_results = []
        try:
            if self._lexical_plugin is None:
                from .lexical_search import LexicalSearchPlugin
                self._lexical_plugin = LexicalSearchPlugin()
            
            # For lexical search, use detected entities instead of full question
            # This improves recall by searching for key terms rather than full questions
            lexical_query = question
            if query_analysis.get("detected_entities"):
                # Use detected entities for better lexical search results
                lexical_query = " ".join(query_analysis["detected_entities"])
                if settings.enable_debug_logging:
                    logger.info(f"Using entities for lexical search: '{lexical_query}' (from: '{question}')")
            
            lexical_raw_results = self._lexical_plugin.search_raw(lexical_query, k)
            
            if settings.enable_debug_logging:
                logger.info(f"Lexical component returned {len(lexical_raw_results)} results")
            
            # Convert lexical results to semantic result format for merging
            if lexical_raw_results:
                # Get full chunk data for lexical results
                lexical_results = self._convert_lexical_to_semantic_format(lexical_raw_results)
            
        except Exception as e:
            logger.warning(f"Lexical search failed, continuing with semantic only: {e}")
            lexical_results = []
        
        # Merge results if we have both types
        if semantic_results and lexical_results:
            merged_results = self._merge_semantic_lexical_results(
                semantic_results, lexical_results, query_analysis
            )
            if settings.enable_debug_logging:
                logger.info(f"Merged results: {len(merged_results)} total")
            return merged_results
        elif semantic_results:
            if settings.enable_debug_logging:
                logger.info("Using semantic results only (lexical failed or empty)")
            return semantic_results
        else:
            if settings.enable_debug_logging:
                logger.info("Using lexical results only (semantic failed or empty)")
            return lexical_results
    
    def _convert_lexical_to_semantic_format(self, lexical_results: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """Convert lexical search results to semantic result format.
        
        Args:
            lexical_results: List of (chunk_id, score) from lexical search
            
        Returns:
            List of results in semantic search format
        """
        # Query database to get full chunk information
        import sqlite3
        
        converted_results = []
        try:
            conn = sqlite3.connect(self._vector_store.db_path)
            cursor = conn.cursor()
            
            for chunk_id, lexical_score in lexical_results:
                cursor.execute("""
                    SELECT c.text, c.file, c.chunk_index, c.id,
                           f.file_name, c.document_title, c.document_id
                    FROM chunks c
                    LEFT JOIN files f ON c.file = f.file_path
                    WHERE c.id = ? AND c.current = 1
                """, (chunk_id,))
                
                row = cursor.fetchone()
                if row:
                    text_content, file_path, chunk_index, chunk_id, file_name, doc_title, doc_id = row
                    
                    converted_results.append({
                        "id": chunk_id,
                        "text": text_content,
                        "file": file_path,
                        "file_path": file_path,
                        "chunk_index": chunk_index,
                        "similarity": lexical_score,  # Use lexical score as similarity
                        "distance": 1.0 - lexical_score,  # Convert to distance
                        "document_path": file_path,
                        "document_title": doc_title or file_name,
                        "document_id": doc_id,
                        "unit": f"chunk_{chunk_index}",
                        "search_type": "lexical"
                    })
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to convert lexical results: {e}")
        
        return converted_results
    
    def _merge_semantic_lexical_results(self, semantic_results: List[Dict[str, Any]], 
                                      lexical_results: List[Dict[str, Any]], 
                                      query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Merge semantic and lexical search results using hybrid scoring.
        
        Args:
            semantic_results: Results from semantic search
            lexical_results: Results from lexical search
            query_analysis: Query analysis for weighting decisions
            
        Returns:
            Merged and deduplicated results sorted by hybrid score
        """
        # Convert to format expected by merge_search_results
        semantic_tuples = [(r.get("id", ""), r.get("similarity", 0.0)) for r in semantic_results]
        lexical_tuples = [(r.get("id", ""), r.get("similarity", 0.0)) for r in lexical_results]
        
        # Get weights from config
        dense_weight = getattr(settings, 'hybrid_dense_weight', 0.6)
        lexical_weight = getattr(settings, 'hybrid_lexical_weight', 0.4)
        normalize_method = getattr(settings, 'hybrid_score_normalize', 'min-max')
        
        # Adjust weights based on query analysis
        strategy = query_analysis.get("strategy", "hybrid")
        if strategy == "lexical_primary":
            dense_weight = 0.3
            lexical_weight = 0.7
        elif strategy == "semantic_primary":
            dense_weight = 0.8
            lexical_weight = 0.2
        elif query_analysis.get("likely_entity_query", False):
            # For entity queries, favor lexical search since entities are exact matches
            dense_weight = 0.3
            lexical_weight = 0.7
            if settings.enable_debug_logging:
                logger.info(f"Adjusting weights for entity query: dense={dense_weight}, lexical={lexical_weight}")
        
        # Merge scores
        merged_scores = merge_search_results(
            semantic_tuples, lexical_tuples,
            dense_weight, lexical_weight, normalize_method
        )
        
        # Create lookup maps for full result data
        semantic_map = {r.get("id", ""): r for r in semantic_results}
        lexical_map = {r.get("id", ""): r for r in lexical_results}
        
        # Build final merged results with full data
        merged_results = []
        for chunk_id, hybrid_score in merged_scores:
            # Prefer semantic result format, fallback to lexical
            if chunk_id in semantic_map:
                result = semantic_map[chunk_id].copy()
                result["hybrid_score"] = hybrid_score
                result["similarity"] = hybrid_score  # Update similarity to hybrid score
                result["distance"] = 1.0 - hybrid_score
                if chunk_id in lexical_map:
                    result["lexical_score"] = lexical_map[chunk_id].get("similarity", 0.0)
                    result["search_type"] = "hybrid"
                else:
                    result["search_type"] = "semantic"
            elif chunk_id in lexical_map:
                result = lexical_map[chunk_id].copy()
                result["hybrid_score"] = hybrid_score
                result["similarity"] = hybrid_score
                result["distance"] = 1.0 - hybrid_score
                result["search_type"] = "lexical"
            else:
                continue  # Skip if chunk_id not found in either
                
            merged_results.append(result)
        
        if settings.enable_debug_logging:
            logger.info(f"Hybrid merging: {len(semantic_results)} semantic + {len(lexical_results)} lexical = {len(merged_results)} merged")
            logger.info(f"Weights used: dense={dense_weight}, lexical={lexical_weight}")
        
        return merged_results
    
    def _analyze_query(self, question: str) -> Dict[str, Any]:
        """Analyze query to detect entity/proper noun queries and other characteristics.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with query analysis results
        """
        analysis = {
            "likely_entity_query": False,
            "detected_entities": [],
            "query_type": "general",
            "short_query": len(question.split()) <= 5,
            "question_words": []
        }
        
        # Common stop words to exclude from entity detection
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "about", "from", "up", "out", "if", "then", "than", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "also", "just", "very", "what", "who", "tell", "find", "show", "locate", "give", "get", "make", "take", "come", "go", "see", "know", "think", "say", "use", "work", "call", "try", "ask", "need", "feel", "become", "leave", "put", "mean", "keep", "let", "begin", "seem", "help", "talk", "turn", "start", "might", "move", "live", "believe", "hold", "bring", "happen", "write", "provide", "sit", "stand", "lose", "pay", "meet", "include", "continue", "set", "learn", "change", "lead", "understand", "watch", "follow", "stop", "create", "speak", "read", "allow", "add", "spend", "grow", "open", "walk", "win", "offer", "remember", "love", "consider", "appear", "buy", "wait", "serve", "die", "send", "expect", "build", "stay", "fall", "cut", "reach", "kill", "remain"}
        
        # Enhanced entity patterns for better multi-word detection
        entity_patterns = [
            r"what is ([A-Z][\w\s\-\.]{1,30}?)(?:\s+(?:and|or|\?|$))",  # "what is Tesla Motors"
            r"who is ([A-Z][\w\s\-\.]{1,30}?)(?:\s+(?:and|or|\?|$))",   # "who is J.K. Rowling"
            r"tell me about ([A-Z][\w\s\-\.]{1,30}?)(?:\s+(?:and|or|\?|$))", # "tell me about New York City"
            r"find (?:information about |documents about |)?([A-Z][\w\s\-\.]{1,30}?)(?:\s+(?:and|or|\?|$))", # "find Tesla" or "find documents about COVID-19"
            r"show me (?:information about |documents about |)?([A-Z][\w\s\-\.]{1,30}?)(?:\s+(?:and|or|\?|$))",
            r"locate (?:information about |documents about |)?([A-Z][\w\s\-\.]{1,30}?)(?:\s+(?:and|or|\?|$))",
            r"information about ([A-Z][\w\s\-\.]{1,30}?)(?:\s+(?:and|or|\?|$))"
        ]
        
        # First pass: Extract entities using enhanced patterns
        for pattern in entity_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                # Clean up the match and normalize
                entity = match.strip()
                entity = re.sub(r'\s+', ' ', entity)  # Normalize whitespace
                if entity and len(entity) > 1:  # Avoid single characters
                    analysis["likely_entity_query"] = True
                    analysis["detected_entities"].append(entity)
                    analysis["query_type"] = "entity_lookup"
        
        # Second pass: Detect proper noun phrases (consecutive capitalized words)
        words = question.split()
        current_phrase = []
        
        for i, word in enumerate(words):
            # Clean word of punctuation for analysis
            clean_word = re.sub(r'[^\w\-]', '', word)
            
            # Check if this could be part of a proper noun phrase
            if (clean_word and 
                clean_word[0].isupper() and 
                clean_word.lower() not in stop_words and
                len(clean_word) > 1):  # Avoid single letters unless they're known patterns
                
                # Handle special cases like "J.K." or "U.S."
                if re.match(r'^[A-Z]\.([A-Z]\.?)*$', word):  # Initials like "J.K."
                    current_phrase.append(word)
                elif clean_word.isalpha() or '-' in clean_word:  # Regular words or hyphenated
                    current_phrase.append(clean_word)
                else:
                    # End current phrase if we hit something that doesn't fit
                    if len(current_phrase) >= 1:
                        entity = ' '.join(current_phrase)
                        if entity not in analysis["detected_entities"]:
                            analysis["detected_entities"].append(entity)
                            analysis["likely_entity_query"] = True
                    current_phrase = []
            else:
                # End current phrase
                if len(current_phrase) >= 1:
                    entity = ' '.join(current_phrase)
                    if entity not in analysis["detected_entities"]:
                        analysis["detected_entities"].append(entity)
                        analysis["likely_entity_query"] = True
                current_phrase = []
        
        # Handle phrase at end of sentence
        if len(current_phrase) >= 1:
            entity = ' '.join(current_phrase)
            if entity not in analysis["detected_entities"]:
                analysis["detected_entities"].append(entity)
                analysis["likely_entity_query"] = True
        
        # Third pass: Detect common entity patterns (acronyms, product names, etc.)
        # Look for ALL CAPS words (likely acronyms) that aren't stop words
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if (clean_word.isupper() and 
                len(clean_word) >= 2 and 
                clean_word.lower() not in stop_words and
                clean_word not in analysis["detected_entities"]):
                analysis["detected_entities"].append(clean_word)
                analysis["likely_entity_query"] = True
        
        # Fourth pass: Detect hyphenated or special format entities
        special_entity_patterns = [
            r'\b[A-Z][\w]*-\d+\b',  # COVID-19, iPhone-15, etc.
            r'\b[A-Z]+/[A-Z]+\b',   # AI/ML, US/UK, etc.
            r'\b\w+\s+\d+(?:\s+\w+)*\b'  # iPhone 15 Pro Max, Windows 11, etc.
        ]
        
        for pattern in special_entity_patterns:
            matches = re.findall(pattern, question)
            for match in matches:
                if match not in analysis["detected_entities"]:
                    analysis["detected_entities"].append(match)
                    analysis["likely_entity_query"] = True
        
        # Clean up detected entities - remove duplicates and very short ones
        unique_entities = []
        for entity in analysis["detected_entities"]:
            entity = entity.strip()
            if (entity and 
                len(entity) > 1 and 
                entity.lower() not in stop_words and
                entity not in unique_entities):
                unique_entities.append(entity)
        
        analysis["detected_entities"] = unique_entities
        
        # Detect question words
        question_words = ["what", "who", "where", "when", "why", "how", "which"]
        analysis["question_words"] = [w for w in words if w.lower() in question_words]
        
        if settings.enable_debug_logging:
            logger.debug(f"Query analysis for '{question}': {analysis}")
            
        return analysis
    
    def _apply_metadata_boosting(self, results: List[Dict[str, Any]], question: str, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply metadata boosting for entity/proper noun queries.
        
        Args:
            results: List of search results
            question: The user's question
            query_analysis: Results from query analysis
            
        Returns:
            Results with metadata boosting applied
        """
        if not query_analysis.get("detected_entities"):
            return results
        
        entities = query_analysis["detected_entities"]
        boost_factor = settings.proper_noun_boost
        
        if settings.enable_debug_logging:
            logger.debug(f"Applying metadata boosting for entities: {entities}")
        
        boosted_results = []
        for result in results:
            original_result = result.copy()
            
            # Check if any detected entities appear in document metadata
            doc_path = result.get("document_path", "").lower()
            doc_title = result.get("document_title", "").lower()
            file_name = result.get("file", "").lower()
            
            match_score = 0
            for entity in entities:
                entity_lower = entity.lower()
                if (entity_lower in doc_path or 
                    entity_lower in doc_title or 
                    entity_lower in file_name):
                    match_score += 1
            
            if match_score > 0:
                # Boost similarity score (decrease distance)
                current_distance = result.get("distance", 1.0)
                current_similarity = result.get("similarity", 0.0)
                
                # Apply boost to similarity and adjust distance accordingly
                boosted_similarity = min(1.0, current_similarity + (boost_factor * match_score))
                boosted_distance = 1.0 - boosted_similarity
                
                original_result["similarity"] = boosted_similarity
                original_result["distance"] = boosted_distance
                original_result["metadata_boosted"] = True
                original_result["metadata_boost_score"] = match_score
                
                if settings.enable_debug_logging:
                    logger.debug(f"Boosted result for entities {entities}: similarity {current_similarity:.3f} -> {boosted_similarity:.3f}")
            
            boosted_results.append(original_result)
        
        # Re-sort by boosted similarity (higher is better) / distance (lower is better)
        boosted_results.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        
        return boosted_results
    
    def _apply_entity_boosting(self, results: List[Dict[str, Any]], detected_entities: List[str]) -> List[Dict[str, Any]]:
        """Apply Phase 4 entity-aware boosting using entity-document mappings.
        
        Args:
            results: List of search results
            detected_entities: Entities detected in the query
            
        Returns:
            Results with entity boosting applied based on stored entity mappings
        """
        if not detected_entities or not results:
            return results
        
        try:
            # Get chunk IDs from results
            chunk_ids = [result.get('chunk_id') for result in results if result.get('chunk_id')]
            if not chunk_ids:
                return results
            
            # Get entity mappings from vector store
            entity_mappings = self._vector_store.get_entity_mappings(chunk_ids=chunk_ids)
            
            # Create a mapping from chunk_id to its entities for quick lookup
            chunk_entities = {}
            for mapping in entity_mappings:
                chunk_id = mapping['chunk_id']
                if chunk_id not in chunk_entities:
                    chunk_entities[chunk_id] = []
                chunk_entities[chunk_id].append(mapping)
            
            # Apply entity boosting
            entity_boost_factor = getattr(settings, 'entity_boost_factor', 0.2)
            boosted_results = []
            
            for result in results:
                boosted_result = result.copy()
                chunk_id = result.get('chunk_id')
                
                if chunk_id and chunk_id in chunk_entities:
                    # Get entity boost score using the stored entity mappings
                    from shared.entity_indexing import get_entity_boost_score
                    
                    chunk_entity_list = chunk_entities[chunk_id]
                    boost_score = get_entity_boost_score(
                        detected_entities,
                        chunk_entity_list,
                        entity_boost_factor
                    )
                    
                    if boost_score > 0:
                        # Apply entity boost to similarity
                        current_similarity = result.get("similarity", 0.0)
                        boosted_similarity = min(1.0, current_similarity + boost_score)
                        boosted_distance = 1.0 - boosted_similarity
                        
                        boosted_result["similarity"] = boosted_similarity
                        boosted_result["distance"] = boosted_distance
                        boosted_result["entity_boosted"] = True
                        boosted_result["entity_boost_score"] = boost_score
                        boosted_result["matched_entities"] = len([
                            ent for ent in detected_entities 
                            if any(ent.lower() == mapping['entity_text'] for mapping in chunk_entity_list)
                        ])
                        
                        if settings.enable_debug_logging:
                            logger.debug(f"Entity boosted chunk {chunk_id}: similarity {current_similarity:.3f} -> {boosted_similarity:.3f}")
                
                boosted_results.append(boosted_result)
            
            # Re-sort by boosted similarity (higher is better)
            boosted_results.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
            
            return boosted_results
            
        except Exception as e:
            if settings.enable_debug_logging:
                logger.error(f"Error in entity boosting: {e}")
            return results
    
    def _build_enhanced_context(self, results: List[Dict[str, Any]]) -> str:
        """Build enhanced context from search results with proper attribution.
        
        Args:
            results: List of search results
            
        Returns:
            Enhanced context string with document attribution
        """
        if not results:
            return ""
        
        context_parts = []
        current_doc = None
        
        for result in results:
            # Add document header when switching documents
            doc_id = result.get('document_id')
            if doc_id != current_doc:
                current_doc = doc_id
                doc_title = result.get('document_title', 'Unknown Document')
                doc_path = result.get('document_path', 'Unknown Path')
                context_parts.append(f"\n--- SOURCE: {doc_title} ({doc_path}) ---")
            
            # Add chunk content with section info if available
            text = result.get('text', '')
            unit = result.get('unit', '')
            section_id = result.get('section_id', '')
            chunk_index = result.get('chunk_index')
            
            section_info = ""
            if section_id:
                section_info = f" [{section_id}]"
            elif unit:
                section_info = f" [{unit}]"
            elif chunk_index is not None:
                section_info = f" [chunk {chunk_index}]"
            
            context_parts.append(f"{text}{section_info}")
        
        return "\n\n".join(context_parts)
    
    def get_info(self) -> PluginInfo:
        """Return plugin metadata and capabilities."""
        return PluginInfo(
            name="semantic_search",
            description="Enhanced semantic document search with Phase 4 advanced features: cross-encoder reranking, query expansion, and entity-aware indexing",
            version="2.2.0",
            capabilities=[
                "semantic_search",
                "document_query",
                "content_analysis", 
                "vector_search",
                "mmr_selection",
                "entity_queries",
                "proper_noun_detection",
                "metadata_boosting",
                "cosine_similarity",
                "diverse_retrieval",
                "query_analysis",
                "structured_logging",
                # Phase 4 capabilities
                "cross_encoder_reranking",
                "query_expansion",
                "entity_indexing",
                "precision_optimization"
            ],
            parameters={
                "question": "str - The question to search for",
                "k": "int - Number of initial chunks to retrieve (default: from config)",
                "mmr_lambda": "float - MMR balance: 1.0=pure relevance, 0.0=pure diversity (default: from config)",
                "mmr_k": "int - Final number of chunks after MMR selection (default: from config)",
                "enable_mmr": "bool - Whether to use MMR selection (default: True)",
                "include_metadata_boost": "bool - Include metadata/filename boosting (default: auto-detect)",
                "force_semantic_only": "bool - Force semantic search only (default: False)",
                "force_hybrid": "bool - Force hybrid search (default: False)",
                # Phase 4 parameters
                "enable_cross_encoder": "bool - Override config for cross-encoder reranking (default: from config)",
                "enable_query_expansion": "bool - Override config for query expansion (default: from config)"
            }
        )
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters before execution."""
        if "question" not in params:
            return False
        
        question = params.get("question")
        if not isinstance(question, str) or not question.strip():
            return False
        
        # Validate optional parameters
        k = params.get("k", settings.retrieval_k)
        if not isinstance(k, int) or k < 1:
            return False
            
        mmr_lambda = params.get("mmr_lambda", settings.mmr_lambda)
        if not isinstance(mmr_lambda, (int, float)) or not 0.0 <= mmr_lambda <= 1.0:
            return False
            
        mmr_k = params.get("mmr_k", settings.mmr_k)
        if not isinstance(mmr_k, int) or mmr_k < 1:
            return False
            
        enable_mmr = params.get("enable_mmr", True)
        if not isinstance(enable_mmr, bool):
            return False
        
        include_metadata_boost = params.get("include_metadata_boost")
        if include_metadata_boost is not None and not isinstance(include_metadata_boost, bool):
            return False
        
        return True
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate search results by chunk_id, keeping the best score for each.
        
        Args:
            results: List of search results potentially containing duplicates
            
        Returns:
            Deduplicated list of results sorted by similarity
        """
        chunk_map = {}
        
        for result in results:
            chunk_id = result.get('chunk_id', '')
            if not chunk_id:
                continue  # Skip results without chunk_id
            
            similarity = result.get('similarity', 0.0)
            
            if chunk_id not in chunk_map or similarity > chunk_map[chunk_id].get('similarity', 0.0):
                # Keep this result if it's the first or has better similarity
                chunk_map[chunk_id] = result
        
        # Convert back to list and sort by similarity
        deduplicated = list(chunk_map.values())
        deduplicated.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
        
        return deduplicated
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        # Vector store and OpenAI client don't need explicit cleanup
        # They will be garbage collected
        self._vector_store = None
        self._openai_client = None
        self._cross_encoder = None
        logger.info("Semantic search plugin cleaned up")