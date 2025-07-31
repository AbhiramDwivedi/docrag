"""Core Agent class for intelligent document querying."""

import logging
import re
from typing import List, Optional, Dict, Any
from .plugin import Plugin, PluginInfo
from .registry import PluginRegistry


logger = logging.getLogger(__name__)


class Agent:
    """
    Intelligent document analysis agent with plugin-based architecture.
    
    The agent classifies queries and routes them to appropriate plugins,
    then synthesizes responses from multiple sources when needed.
    """
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        """
        Initialize the agent.
        
        Args:
            registry: Plugin registry to use. If None, creates a new empty registry.
        """
        self.registry = registry or PluginRegistry()
        self._last_execution_trace: List[Dict[str, Any]] = []
    
    def process_query(self, question: str) -> str:
        """
        Process natural language query and return response.
        
        Args:
            question: User's natural language question
            
        Returns:
            String response to the question
        """
        self._last_execution_trace = []
        
        try:
            # Classify the query to determine type and routing
            query_type = self._classify_query(question)
            
            # Find plugins that can handle this query
            capable_plugins = self.registry.get_plugins_for_query(question, query_type)
            
            if not capable_plugins:
                return "No plugins available to handle this query."
            
            # For Phase 1, use simple single-plugin routing
            # Later phases will implement multi-plugin coordination
            plugin = self._select_plugin(capable_plugins, question, query_type)
            
            # Execute the plugin
            params = self._prepare_plugin_params(plugin, question, query_type)
            
            if not plugin.validate_params(params):
                return f"Invalid parameters for {plugin.get_info().name} plugin."
            
            # Track execution
            execution_step = {
                'plugin': plugin.get_info().name,
                'query_type': query_type,
                'params': params
            }
            
            result = plugin.execute(params)
            execution_step['result'] = result
            self._last_execution_trace.append(execution_step)
            
            # Extract and format the response
            return self._format_response(result)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {e}"
    
    def get_capabilities(self) -> List[str]:
        """
        Return list of agent capabilities for introspection.
        
        Returns:
            List of capability descriptions
        """
        capabilities = []
        plugin_info = self.registry.list_capabilities()
        
        for name, info in plugin_info.items():
            capabilities.extend([
                f"{name}: {desc}" for desc in info.capabilities
            ])
        
        return capabilities
    
    def explain_reasoning(self) -> Optional[str]:
        """
        Return explanation of last query processing steps.
        
        Returns:
            Explanation string or None if no query has been processed
        """
        if not self._last_execution_trace:
            return None
        
        explanation = "Query Processing Steps:\n"
        
        for i, step in enumerate(self._last_execution_trace, 1):
            explanation += f"{i}. Used {step['plugin']} plugin "
            explanation += f"(query type: {step['query_type']})\n"
            
            if 'result' in step and isinstance(step['result'], dict):
                if 'metadata' in step['result']:
                    meta = step['result']['metadata']
                    if 'sources_count' in meta:
                        explanation += f"   - Found {meta['sources_count']} relevant sources\n"
                    if 'confidence' in step['result']:
                        explanation += f"   - Confidence: {step['result']['confidence']:.2f}\n"
        
        return explanation
    
    def _classify_query(self, question: str) -> str:
        """
        Classify the query type for routing decisions.
        
        Args:
            question: User's question
            
        Returns:
            Query type string
        """
        question_lower = question.lower()
        
        # Metadata/statistics queries
        metadata_patterns = [
            r'\bhow many\b',
            r'\bcount\b',
            r'\btotal\b',
            r'\blist\b',
            r'\bshow me\b',
            r'\bwhat.*files?\b',
            r'\bfile types?\b',
            r'\bstatistics?\b'
        ]
        
        for pattern in metadata_patterns:
            if re.search(pattern, question_lower):
                return 'metadata'
        
        # Default to semantic search for content-based queries
        return 'semantic'
    
    def _select_plugin(self, plugins: List[Plugin], question: str, query_type: str) -> Plugin:
        """
        Select the best plugin for the query.
        
        Args:
            plugins: List of capable plugins
            question: User's question
            query_type: Classified query type
            
        Returns:
            Selected plugin
        """
        # Simple selection logic for Phase 1
        # Priority: metadata plugins for metadata queries, semantic for others
        
        if query_type == 'metadata':
            # Prefer metadata plugins
            for plugin in plugins:
                info = plugin.get_info()
                if 'metadata' in info.name.lower():
                    return plugin
        
        # Default to first semantic plugin or first available
        for plugin in plugins:
            info = plugin.get_info()
            if 'semantic' in info.name.lower():
                return plugin
        
        return plugins[0]
    
    def _prepare_plugin_params(self, plugin: Plugin, question: str, query_type: str) -> Dict[str, Any]:
        """
        Prepare parameters for plugin execution.
        
        Args:
            plugin: Selected plugin
            question: User's question
            query_type: Query type
            
        Returns:
            Parameters dictionary
        """
        return {
            'question': question,
            'query_type': query_type
        }
    
    def _format_response(self, result: Dict[str, Any]) -> str:
        """
        Format plugin result into user-friendly response.
        
        Args:
            result: Plugin execution result
            
        Returns:
            Formatted response string
        """
        if 'result' not in result:
            return "No result returned from plugin."
        
        response = result['result']
        
        # If result is already a string, return it
        if isinstance(response, str):
            return response
        
        # Handle structured results
        if isinstance(response, dict):
            if 'answer' in response:
                return response['answer']
            elif 'message' in response:
                return response['message']
        
        # Fallback to string representation
        return str(response)