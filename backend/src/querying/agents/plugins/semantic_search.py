"""Semantic search plugin for DocQuest agent framework.

This plugin wraps the existing vector search functionality to provide
semantic document search capabilities through the agent framework.
"""

import logging
import sys
import re
from pathlib import Path
from typing import Dict, Any, List

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from ..plugin import Plugin, PluginInfo

# Try different import approaches for the dependencies
try:
    from ingestion.processors.embedder import embed_texts
    from ingestion.storage.vector_store import VectorStore
    from shared.config import settings
    from shared.mmr import mmr_selection, extract_embeddings_from_results
except ImportError:
    # Fallback to absolute imports
    try:
        from src.ingestion.processors.embedder import embed_texts
        from src.ingestion.storage.vector_store import VectorStore
        from src.shared.config import settings
        from src.shared.mmr import mmr_selection, extract_embeddings_from_results
    except ImportError:
        # Last resort - relative imports from backend root
        import os
        backend_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(backend_root))
        from src.ingestion.processors.embedder import embed_texts
        from src.ingestion.storage.vector_store import VectorStore
        from src.shared.config import settings
        from src.shared.mmr import mmr_selection, extract_embeddings_from_results

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
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced semantic search with robustness improvements.
        
        Implements multi-stage search process:
        1. Query Analysis - Detect entity/proper noun queries
        2. Initial Retrieval - Vector search for top-k chunks
        3. MMR Selection - Select diverse relevant chunks
        4. Optional Metadata Boosting - Boost document-name matches
        5. Response Generation - Synthesize answer with attribution
        
        Args:
            params: Dictionary containing:
                - question: The question to search for
                - k: Number of initial chunks to retrieve (default: from config)
                - mmr_lambda: MMR balance parameter (default: from config)
                - mmr_k: Final number of chunks after MMR (default: from config)
                - enable_mmr: Whether to use MMR selection (default: True)
                - include_metadata_boost: Include metadata/filename boosting (default: auto-detect)
                
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
        
        if not question.strip():
            return {
                "response": "No question provided.",
                "sources": [],
                "metadata": {"error": "empty_question"}
            }
        
        # Stage 1: Query Analysis
        query_analysis = self._analyze_query(question)
        if include_metadata_boost is None:
            include_metadata_boost = query_analysis.get("likely_entity_query", False)
            
        if settings.enable_debug_logging:
            logger.info(f"Query analysis: {query_analysis}")
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
                self._vector_store = VectorStore.load(
                    Path(settings.vector_path), 
                    Path(settings.db_path)
                )
            
            # Stage 2: Initial Retrieval - Vector search for top-k chunks
            if settings.enable_debug_logging:
                logger.info(f"Starting initial retrieval with k={k}")
                
            q_vec = embed_texts([question], settings.embed_model)[0]
            initial_results = self._vector_store.query(q_vec, k=k)
            
            if settings.enable_debug_logging:
                logger.info(f"Initial retrieval returned {len(initial_results)} results")
                if initial_results:
                    logger.info(f"Top result similarity: {initial_results[0].get('similarity', 'N/A')}")
            
            if not initial_results:
                return {
                    "response": "No relevant information found.",
                    "sources": [],
                    "metadata": {
                        "results_count": 0, 
                        "stage": "initial_retrieval",
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
                if settings.enable_debug_logging:
                    logger.info(f"Starting MMR selection: {len(initial_results)} -> {mmr_k} with lambda={mmr_lambda}")
                
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
                    mmr_lambda=mmr_lambda,
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
                "model_used": "gpt-4o-mini",
                "embedding_model": settings.embed_model,
                "context_length": len(enhanced_context),
                "retrieval_k": k
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
                "metadata": {"error": str(e), "query_analysis": query_analysis}
            }
    
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
        
        # Detect question patterns that suggest entity queries
        entity_patterns = [
            r"what is (\w+)",
            r"who is (\w+)",
            r"tell me about (\w+)",
            r"find (\w+)",
            r"show me (\w+)",
            r"locate (\w+)",
            r"information about (\w+)"
        ]
        
        question_lower = question.lower()
        for pattern in entity_patterns:
            matches = re.findall(pattern, question_lower)
            if matches:
                analysis["likely_entity_query"] = True
                analysis["detected_entities"].extend(matches)
                analysis["query_type"] = "entity_lookup"
        
        # Detect proper nouns (capitalized words that aren't at sentence start)
        words = question.split()
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and word.isalpha():
                analysis["detected_entities"].append(word)
                analysis["likely_entity_query"] = True
        
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
            description="Enhanced semantic document search with cosine similarity, MMR selection, and entity query robustness",
            version="2.1.0",
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
                "structured_logging"
            ],
            parameters={
                "question": "str - The question to search for",
                "k": "int - Number of initial chunks to retrieve (default: from config)",
                "mmr_lambda": "float - MMR balance: 1.0=pure relevance, 0.0=pure diversity (default: from config)",
                "mmr_k": "int - Final number of chunks after MMR selection (default: from config)",
                "enable_mmr": "bool - Whether to use MMR selection (default: True)",
                "include_metadata_boost": "bool - Include metadata/filename boosting (default: auto-detect)"
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
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        # Vector store and OpenAI client don't need explicit cleanup
        # They will be garbage collected
        self._vector_store = None
        self._openai_client = None
        logger.info("Semantic search plugin cleaned up")