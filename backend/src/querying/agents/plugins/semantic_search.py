"""Semantic search plugin for DocQuest agent framework.

This plugin wraps the existing vector search functionality to provide
semantic document search capabilities through the agent framework.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from querying.agents.plugin import Plugin, PluginInfo
from ingestion.processors.embedder import embed_texts
from ingestion.storage.enhanced_vector_store import EnhancedVectorStore
from shared.config import settings
from openai import OpenAI

logger = logging.getLogger(__name__)


class SemanticSearchPlugin(Plugin):
    """Plugin for semantic document search using vector embeddings.
    
    This plugin provides the core semantic search functionality that was
    previously in the CLI, now wrapped in the plugin architecture.
    """
    
    def __init__(self):
        """Initialize the semantic search plugin."""
        self._vector_store = None
        self._openai_client = None
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced semantic search with document-level retrieval.
        
        Implements multi-stage search process:
        1. Initial Discovery - Vector search for top-k chunks
        2. Document Selection - Group and rank documents 
        3. Context Expansion - Retrieve full document sections
        
        Args:
            params: Dictionary containing:
                - question: The question to search for
                - k: Number of initial chunks to retrieve (default: 50)
                - max_documents: Maximum documents to analyze (default: 5)
                - context_window: Context expansion window (default: 3)
                - use_document_level: Enable document-level retrieval (default: True)
                
        Returns:
            Dictionary containing:
                - response: The synthesized answer with source attribution
                - sources: List of source documents with detailed attribution
                - metadata: Search metadata and document analysis
        """
        question = params.get("question", "")
        k = params.get("k", 50)  # Increased for document-level analysis
        max_documents = params.get("max_documents", 5)
        context_window = params.get("context_window", 3)
        use_document_level = params.get("use_document_level", True)
        
        if not question.strip():
            return {
                "response": "No question provided.",
                "sources": [],
                "metadata": {"error": "empty_question"}
            }
        
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
                self._vector_store = EnhancedVectorStore.load(
                    Path(settings.vector_path), 
                    Path(settings.db_path)
                )
            
            # Stage 1: Initial Discovery - Vector search for top-k chunks
            q_vec = embed_texts([question], settings.embed_model)[0]
            initial_results = self._vector_store.query(q_vec, k=k)
            
            if not initial_results:
                return {
                    "response": "No relevant information found.",
                    "sources": [],
                    "metadata": {"results_count": 0, "stage": "initial_discovery"}
                }
            
            if use_document_level:
                # Stage 2: Document Selection - Group and rank documents
                ranked_documents = self._vector_store.rank_documents_by_relevance(initial_results)
                selected_documents = ranked_documents[:max_documents]
                
                # Stage 3: Context Expansion - Retrieve full document sections
                context_chunks = []
                source_documents = []
                
                for doc_info in selected_documents:
                    doc_id = doc_info['document_id']
                    
                    # Get expanded context for this document's chunks
                    chunk_ids = [chunk['id'] for chunk in doc_info['chunks']]
                    expanded_chunks = self._vector_store.get_document_context(
                        chunk_ids, window_size=context_window
                    )
                    
                    context_chunks.extend(expanded_chunks)
                    
                    # Prepare document source info
                    source_documents.append({
                        "document_id": doc_id,
                        "document_path": doc_info['document_path'],
                        "document_title": doc_info['document_title'],
                        "document_type": doc_info['document_type'],
                        "relevance_score": doc_info['avg_relevance'],
                        "chunk_count": doc_info['chunk_count'],
                        "sections_retrieved": len(expanded_chunks)
                    })
                
                # Build enhanced context with document attribution
                context_parts = []
                current_doc = None
                
                for chunk in context_chunks:
                    # Add document header when switching documents
                    if chunk['document_id'] != current_doc:
                        current_doc = chunk['document_id']
                        doc_title = chunk.get('document_title', 'Unknown Document')
                        doc_path = chunk.get('document_path', 'Unknown Path')
                        context_parts.append(f"\n--- SOURCE: {doc_title} ({doc_path}) ---")
                    
                    # Add chunk content with section info if available
                    section_info = f" [{chunk.get('section_id', chunk.get('unit', 'section'))}]" if chunk.get('section_id') or chunk.get('unit') else ""
                    context_parts.append(f"{chunk.get('text', '')}{section_info}")
                
                enhanced_context = "\n\n".join(context_parts)
                
                # Enhanced prompt with document-level instructions
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
                
                search_metadata = {
                    "results_count": len(initial_results),
                    "documents_analyzed": len(selected_documents),
                    "total_chunks_used": len(context_chunks),
                    "context_expansion_used": True,
                    "max_documents": max_documents,
                    "context_window": context_window
                }
                
            else:
                # Legacy mode - use original approach
                top_results = initial_results[:8]  # Use traditional k=8
                enhanced_context = "\n\n".join([row.get('text', '') for row in top_results])
                prompt = f"Answer the question using only the context below.\nContext:\n\"\"\"\n{enhanced_context}\n\"\"\"\nQ: {question}\nA:"
                
                source_documents = []
                for result in top_results:
                    source_documents.append({
                        "file": result.get("file", "unknown"),
                        "unit": result.get("unit", "unknown"),
                        "distance": result.get("distance", 0.0),
                        "document_path": result.get("document_path"),
                        "document_title": result.get("document_title")
                    })
                
                search_metadata = {
                    "results_count": len(top_results),
                    "context_expansion_used": False,
                    "legacy_mode": True
                }
            
            # Initialize OpenAI client if needed
            if self._openai_client is None:
                self._openai_client = OpenAI(api_key=settings.openai_api_key)
            
            # Get response from OpenAI
            resp = self._openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=1024,  # Increased for document-level responses
            )
            
            answer = resp.choices[0].message.content
            
            return {
                "response": answer,
                "sources": source_documents,
                "metadata": {
                    **search_metadata,
                    "model_used": "gpt-4o-mini",
                    "embedding_model": settings.embed_model,
                    "context_length": len(enhanced_context)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced semantic search: {e}")
            return {
                "response": f"❌ Error calling OpenAI API: {e}\nPlease check your API key in config/config.yaml",
                "sources": [],
                "metadata": {"error": str(e)}
            }
    
    def get_info(self) -> PluginInfo:
        """Return plugin metadata and capabilities."""
        return PluginInfo(
            name="semantic_search",
            description="Enhanced semantic document search with document-level retrieval, context expansion, and comprehensive source attribution",
            version="2.0.0",
            capabilities=[
                "semantic_search",
                "document_query",
                "content_analysis",
                "vector_search",
                "document_level_retrieval",
                "context_expansion",
                "source_attribution",
                "multi_stage_search",
                "document_ranking"
            ],
            parameters={
                "question": "str - The question to search for",
                "k": "int - Number of initial chunks to retrieve (default: 50)",
                "max_documents": "int - Maximum documents to analyze (default: 5)", 
                "context_window": "int - Context expansion window size (default: 3)",
                "use_document_level": "bool - Enable document-level retrieval (default: True)"
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
        k = params.get("k", 50)
        if not isinstance(k, int) or k < 1:
            return False
            
        max_documents = params.get("max_documents", 5)
        if not isinstance(max_documents, int) or max_documents < 1:
            return False
            
        context_window = params.get("context_window", 3)
        if not isinstance(context_window, int) or context_window < 0:
            return False
            
        use_document_level = params.get("use_document_level", True)
        if not isinstance(use_document_level, bool):
            return False
        
        return True
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        # Vector store and OpenAI client don't need explicit cleanup
        # They will be garbage collected
        self._vector_store = None
        self._openai_client = None
        logger.info("Semantic search plugin cleaned up")