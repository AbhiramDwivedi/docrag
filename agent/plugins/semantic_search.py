"""Semantic search plugin that wraps the existing vector search functionality."""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directories to path to import from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.plugin import Plugin, PluginInfo
from ingest.embed import embed_texts
from ingest.vector_store import VectorStore
from config.config import settings
from openai import OpenAI


logger = logging.getLogger(__name__)


class SemanticSearchPlugin(Plugin):
    """
    Plugin that provides semantic search capabilities using vector embeddings.
    
    This plugin wraps the existing vector search functionality from cli/ask.py
    to maintain identical behavior while providing it through the plugin interface.
    """
    
    def __init__(self):
        """Initialize the semantic search plugin."""
        self._openai_client = None
        
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute semantic search on the document collection.
        
        Args:
            params: Must contain 'question' key with user's query
            
        Returns:
            Dictionary with:
            - 'result': The answer string
            - 'metadata': Information about sources and processing
            - 'confidence': Confidence score if available
        """
        question = params['question']
        
        # Check if OpenAI API key is configured
        if not settings.openai_api_key or settings.openai_api_key == "your-openai-api-key-here":
            return {
                'result': ("❌ OpenAI API key not configured.\n"
                          "Please set your API key in config/config.yaml:\n"
                          "1. Get your API key from: https://platform.openai.com/api-keys\n"
                          "2. Edit config/config.yaml and replace 'your-openai-api-key-here' with your actual key\n"
                          "3. Alternatively, set the OPENAI_API_KEY environment variable"),
                'metadata': {'error': 'missing_api_key'},
                'confidence': 0.0
            }
        
        try:
            # Initialize vector store and perform search
            store = VectorStore(settings.vector_path, settings.db_path, dim=384)
            q_vec = embed_texts([question], settings.embed_model)[0]
            top = store.query(q_vec, k=8)
            
            if not top:
                return {
                    'result': "No relevant information found.",
                    'metadata': {'sources_count': 0},
                    'confidence': 0.0
                }
            
            # Prepare context for GPT
            context = "\n\n".join([row.get('text', '') for row in top])
            prompt = f"Answer the question using only the context below.\nContext:\n\"\"\"\n{context}\n\"\"\"\nQ: {question}\nA:"
            
            # Query OpenAI
            if self._openai_client is None:
                self._openai_client = OpenAI(api_key=settings.openai_api_key)
            
            resp = self._openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=512,
            )
            
            answer = resp.choices[0].message.content
            
            # Prepare sources information
            sources = []
            for row in top:
                sources.append({
                    'file': row.get('file', ''),
                    'unit': row.get('unit', ''),
                    'distance': row.get('distance', 0.0)
                })
            
            return {
                'result': answer,
                'metadata': {
                    'sources_count': len(top),
                    'sources': sources,
                    'model': 'gpt-4o-mini',
                    'embedding_model': settings.embed_model
                },
                'confidence': 0.8  # Default confidence for successful search
            }
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return {
                'result': f"❌ Error calling OpenAI API: {e}\nPlease check your API key in config/config.yaml",
                'metadata': {'error': str(e)},
                'confidence': 0.0
            }
    
    def get_info(self) -> PluginInfo:
        """Return plugin metadata and capabilities."""
        return PluginInfo(
            name="semantic_search",
            description="Performs semantic search through document collection using vector embeddings and GPT synthesis",
            version="1.0.0",
            capabilities=[
                "Answer questions based on document content",
                "Semantic similarity search across all indexed documents",
                "Context-aware response synthesis using GPT",
                "File and unit source attribution"
            ],
            parameters={
                "question": "The natural language question to answer"
            }
        )
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameters before execution.
        
        Args:
            params: Parameters to validate
            
        Returns:
            True if parameters are valid
        """
        return isinstance(params, dict) and 'question' in params and isinstance(params['question'], str) and len(params['question'].strip()) > 0
    
    def can_handle(self, query: str, query_type: str = None) -> bool:
        """
        Determine if this plugin can handle the given query.
        
        Args:
            query: The user's query
            query_type: Optional hint about query type
            
        Returns:
            True if plugin can handle the query
        """
        # Handle all semantic/content queries, and anything not clearly metadata
        if query_type == 'semantic':
            return True
        
        # Also handle if no specific type or unknown type
        if query_type is None or query_type not in ['metadata']:
            return True
        
        return False