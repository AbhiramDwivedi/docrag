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

from docquest.querying.agents.plugin import Plugin, PluginInfo
from docquest.ingestion.processors.embedder import embed_texts
from docquest.ingestion.storage.enhanced_vector_store import EnhancedVectorStore
from docquest.shared.config import settings
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
        """Execute semantic search for the given query.
        
        Args:
            params: Dictionary containing:
                - question: The question to search for
                - k: Number of results to return (default: 8)
                
        Returns:
            Dictionary containing:
                - response: The synthesized answer
                - sources: List of source documents used
                - metadata: Additional metadata about the search
        """
        question = params.get("question", "")
        k = params.get("k", 8)
        
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
            
            # Embed the question
            q_vec = embed_texts([question], settings.embed_model)[0]
            
            # Query vector store
            top_results = self._vector_store.query(q_vec, k=k)
            
            if not top_results:
                return {
                    "response": "No relevant information found.",
                    "sources": [],
                    "metadata": {"results_count": 0}
                }
            
            # Prepare context for OpenAI
            context = "\n\n".join([row.get('text', '') for row in top_results])
            prompt = f"Answer the question using only the context below.\nContext:\n\"\"\"\n{context}\n\"\"\"\nQ: {question}\nA:"
            
            # Initialize OpenAI client if needed
            if self._openai_client is None:
                self._openai_client = OpenAI(api_key=settings.openai_api_key)
            
            # Get response from OpenAI
            resp = self._openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=512,
            )
            
            answer = resp.choices[0].message.content
            
            # Prepare sources list
            sources = []
            for result in top_results:
                sources.append({
                    "file": result.get("file", "unknown"),
                    "unit": result.get("unit", "unknown"),
                    "distance": result.get("distance", 0.0)
                })
            
            return {
                "response": answer,
                "sources": sources,
                "metadata": {
                    "results_count": len(top_results),
                    "model_used": "gpt-4o-mini",
                    "embedding_model": settings.embed_model
                }
            }
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return {
                "response": f"❌ Error calling OpenAI API: {e}\nPlease check your API key in config/config.yaml",
                "sources": [],
                "metadata": {"error": str(e)}
            }
    
    def get_info(self) -> PluginInfo:
        """Return plugin metadata and capabilities."""
        return PluginInfo(
            name="semantic_search",
            description="Semantic document search using vector embeddings and OpenAI GPT synthesis",
            version="1.0.0",
            capabilities=[
                "semantic_search",
                "document_query",
                "content_analysis",
                "vector_search"
            ],
            parameters={
                "question": "str - The question to search for",
                "k": "int - Number of results to return (default: 8)"
            }
        )
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters before execution."""
        if "question" not in params:
            return False
        
        question = params.get("question")
        if not isinstance(question, str) or not question.strip():
            return False
        
        k = params.get("k", 8)
        if not isinstance(k, int) or k < 1:
            return False
        
        return True
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        # Vector store and OpenAI client don't need explicit cleanup
        # They will be garbage collected
        self._vector_store = None
        self._openai_client = None
        logger.info("Semantic search plugin cleaned up")