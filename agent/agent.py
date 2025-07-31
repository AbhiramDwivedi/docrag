"""Core agent implementation for DocQuest intelligent document analysis."""

import logging
import time
from typing import List, Optional, Dict, Any
from .registry import PluginRegistry
from .plugin import Plugin

logger = logging.getLogger(__name__)


class Agent:
    """Core agent for intelligent document analysis.
    
    The agent coordinates multiple plugins to process natural language queries
    and provide comprehensive responses about document collections.
    """
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        """Initialize the agent with a plugin registry.
        
        Args:
            registry: Plugin registry instance. If None, creates a new one.
        """
        self.registry = registry or PluginRegistry()
        self._last_query = ""
        self._last_plugins_used = []
        self._last_execution_time = 0.0
        self._reasoning_trace = []
    
    def process_query(self, question: str) -> str:
        """Process natural language query and return response.
        
        Args:
            question: Natural language question to process
            
        Returns:
            Formatted response string
        """
        start_time = time.time()
        self._last_query = question
        self._last_plugins_used = []
        self._reasoning_trace = []
        
        try:
            # Classify query and determine appropriate plugins
            plugins_to_use = self._classify_query(question)
            
            if not plugins_to_use:
                return "No relevant information found."
            
            # Execute plugins and collect results
            results = []
            for plugin_name in plugins_to_use:
                plugin = self.registry.get_plugin(plugin_name)
                if plugin:
                    try:
                        self._reasoning_trace.append(f"Executing plugin: {plugin_name}")
                        params = self._prepare_params(plugin_name, question)
                        
                        if plugin.validate_params(params):
                            result = plugin.execute(params)
                            results.append((plugin_name, result))
                            self._last_plugins_used.append(plugin_name)
                        else:
                            logger.warning(f"Invalid parameters for plugin {plugin_name}")
                    except Exception as e:
                        logger.error(f"Error executing plugin {plugin_name}: {e}")
                        self._reasoning_trace.append(f"Plugin {plugin_name} failed: {e}")
            
            # Synthesize final response
            response = self._synthesize_response(question, results)
            
            self._last_execution_time = time.time() - start_time
            return response
            
        except Exception as e:
            logger.error(f"Error processing query '{question}': {e}")
            return f"❌ Error processing query: {e}"
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities for introspection.
        
        Returns:
            List of capabilities provided by all registered plugins
        """
        capabilities = []
        for plugin_info in self.registry._plugin_info.values():
            capabilities.extend(plugin_info.capabilities)
        return list(set(capabilities))  # Remove duplicates
    
    def explain_reasoning(self) -> Optional[str]:
        """Return explanation of last query processing steps.
        
        Returns:
            Human-readable explanation of reasoning process, or None if no query processed
        """
        if not self._last_query:
            return None
        
        explanation = [
            f"Query: {self._last_query}",
            f"Execution time: {self._last_execution_time:.2f}s",
            f"Plugins used: {', '.join(self._last_plugins_used) if self._last_plugins_used else 'None'}",
            "",
            "Reasoning trace:"
        ]
        
        if self._reasoning_trace:
            for i, step in enumerate(self._reasoning_trace, 1):
                explanation.append(f"  {i}. {step}")
        else:
            explanation.append("  No reasoning steps recorded")
        
        return "\n".join(explanation)
    
    def _classify_query(self, question: str) -> List[str]:
        """Classify query and determine which plugins should handle it.
        
        Args:
            question: User question to classify
            
        Returns:
            List of plugin names to execute for this query
        """
        question_lower = question.lower()
        plugins_to_use = []
        
        # Check for metadata queries
        metadata_keywords = [
            "how many", "count", "number of", "total", "list", "show me",
            "what files", "file types", "recently", "latest", "newest"
        ]
        
        if any(keyword in question_lower for keyword in metadata_keywords):
            if self.registry.get_plugin("metadata"):
                plugins_to_use.append("metadata")
                self._reasoning_trace.append("Detected metadata query keywords")
        
        # Check for semantic search queries
        # If it's not clearly a metadata query, or if it contains content-related terms,
        # use semantic search
        content_keywords = [
            "what is", "explain", "about", "describe", "compliance", "policy",
            "procedure", "requirements", "contains", "mentions", "discusses"
        ]
        
        is_content_query = (
            any(keyword in question_lower for keyword in content_keywords) or
            not plugins_to_use  # Default to semantic search if no metadata keywords
        )
        
        if is_content_query and self.registry.get_plugin("semantic_search"):
            plugins_to_use.append("semantic_search")
            self._reasoning_trace.append("Using semantic search for content analysis")
        
        return plugins_to_use
    
    def _prepare_params(self, plugin_name: str, question: str) -> Dict[str, Any]:
        """Prepare parameters for plugin execution.
        
        Args:
            plugin_name: Name of the plugin to prepare parameters for
            question: Original user question
            
        Returns:
            Dictionary of parameters for the plugin
        """
        # Basic parameters that all plugins might need
        params = {
            "question": question,
            "query": question,  # Alias for compatibility
        }
        
        # Plugin-specific parameter preparation could be added here
        # For now, we use the same basic parameters for all plugins
        
        return params
    
    def _synthesize_response(self, question: str, results: List[tuple]) -> str:
        """Synthesize final response from plugin results.
        
        Args:
            question: Original user question
            results: List of (plugin_name, result_dict) tuples
            
        Returns:
            Synthesized response string
        """
        if not results:
            return "No relevant information found."
        
        # If only one plugin was used, return its result directly
        if len(results) == 1:
            _, result = results[0]
            return result.get("response", "No response from plugin")
        
        # For multiple plugins, combine results
        # This is a simple implementation - could be enhanced with LLM synthesis
        response_parts = []
        for plugin_name, result in results:
            plugin_response = result.get("response", "")
            if plugin_response:
                response_parts.append(plugin_response)
        
        if response_parts:
            return "\n\n".join(response_parts)
        else:
            return "No relevant information found."