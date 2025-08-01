"""Core agent implementation for DocQuest intelligent document analysis."""

import logging
import time
import re
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
        """Enhanced query classification with multi-step planning support.
        
        Args:
            question: User question to classify
            
        Returns:
            List of plugin names to execute for this query
        """
        question_lower = question.lower()
        plugins_to_use = []
        
        # Enhanced email query detection
        email_indicators = [
            "email", "emails", "mail", "sender", "sent", "received",
            "from", "to", "subject", "message"
        ]
        
        # Enhanced metadata keywords
        metadata_keywords = [
            "how many", "count", "number of", "total", "list", "show me",
            "what files", "file types", "recently", "latest", "newest", 
            "recent files", "recent documents", "size", "larger", "smaller",
            "modified", "created", "last week", "last month", "yesterday"
        ]
        
        # Check for email-specific queries
        has_email_indicators = any(keyword in question_lower for keyword in email_indicators)
        
        # Check for metadata queries
        has_metadata_indicators = any(keyword in question_lower for keyword in metadata_keywords)
        
        # Multi-step query detection - queries that might need both plugins
        multi_step_patterns = [
            "latest email about",
            "recent email regarding", 
            "emails about .* and related files",
            "find .* and show",
            "emails .* plus"
        ]
        
        is_multi_step = any(re.search(pattern, question_lower) for pattern in multi_step_patterns)
        
        # Enhanced classification logic
        if has_email_indicators and has_metadata_indicators:
            # Queries like "emails from John last week" - primarily metadata
            if self.registry.get_plugin("metadata"):
                plugins_to_use.append("metadata")
                self._reasoning_trace.append("Detected email metadata query")
        
        elif has_email_indicators:
            # Pure email queries - route to metadata for Phase 2
            if self.registry.get_plugin("metadata"):
                plugins_to_use.append("metadata")
                self._reasoning_trace.append("Detected email-specific query")
        
        elif has_metadata_indicators:
            # Pure metadata queries
            if self.registry.get_plugin("metadata"):
                plugins_to_use.append("metadata")
                self._reasoning_trace.append("Detected metadata query keywords")
        
        # Check for semantic search queries
        content_keywords = [
            "what is", "explain", "about", "describe", "compliance", "policy",
            "procedure", "requirements", "contains", "mentions", "discusses",
            "content", "text", "information"
        ]
        
        is_content_query = (
            any(keyword in question_lower for keyword in content_keywords) or
            (not plugins_to_use and not any(word in question_lower for word in [
                "files", "documents", "count", "list", "show", "how many", "types", "email"
            ]))  # Default to semantic search if no clear metadata/email indicators
        )
        
        # Add semantic search for content queries or multi-step queries
        if (is_content_query or is_multi_step) and self.registry.get_plugin("semantic_search"):
            if "semantic_search" not in plugins_to_use:
                plugins_to_use.append("semantic_search")
                self._reasoning_trace.append("Using semantic search for content analysis")
        
        # Special handling for complex queries that benefit from both plugins
        complex_patterns = [
            "about .* files",  # "about budget files" - content + metadata
            "documents .* recent",  # "documents about X recent" - content + time filter
            "find .* and list"  # "find X and list files" - content + metadata
        ]
        
        is_complex = any(re.search(pattern, question_lower) for pattern in complex_patterns)
        
        if is_complex:
            # Add both plugins for complex queries
            for plugin_name in ["metadata", "semantic_search"]:
                if self.registry.get_plugin(plugin_name) and plugin_name not in plugins_to_use:
                    plugins_to_use.append(plugin_name)
                    self._reasoning_trace.append(f"Added {plugin_name} for complex query")
        
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
        """Enhanced response synthesis with multi-step query support.
        
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
        
        # For multiple plugins, we need intelligent synthesis
        self._reasoning_trace.append(f"Synthesizing responses from {len(results)} plugins")
        
        # Separate metadata and content results
        metadata_results = []
        content_results = []
        
        for plugin_name, result in results:
            if plugin_name == "metadata":
                metadata_results.append(result)
            elif plugin_name == "semantic_search":
                content_results.append(result)
        
        # Build combined response
        response_parts = []
        
        # Start with metadata if it provides context
        if metadata_results:
            for metadata_result in metadata_results:
                metadata_response = metadata_result.get("response", "")
                if metadata_response and not metadata_response.startswith("❌"):
                    # Check if this is providing context for content search
                    if content_results and any(word in question.lower() for word in ["about", "regarding", "contains"]):
                        response_parts.append(f"📊 {metadata_response}")
                    else:
                        response_parts.append(metadata_response)
        
        # Add content results with context
        if content_results:
            for content_result in content_results:
                content_response = content_result.get("response", "")
                if content_response and not content_response.startswith("❌"):
                    if metadata_results:
                        # If we have metadata context, introduce content section
                        response_parts.append(f"\n📄 Content Analysis:\n{content_response}")
                    else:
                        response_parts.append(content_response)
        
        # Combine results intelligently
        if response_parts:
            # For queries that ask for both metadata and content, structure the response
            if len(response_parts) > 1 and any(word in question.lower() for word in ["and", "plus", "also"]):
                return "\n\n".join(response_parts)
            else:
                return "\n\n".join(response_parts)
        else:
            return "No relevant information found."