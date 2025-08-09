"""Core agent implementation for DocQuest intelligent document analysis."""

import logging
import time
import re
from typing import List, Optional, Dict, Any
from .registry import PluginRegistry
from .plugin import Plugin

logger = logging.getLogger(__name__)

# Specialized loggers for verbose output
classification_logger = logging.getLogger('agent.classification')
execution_logger = logging.getLogger('agent.execution')
synthesis_logger = logging.getLogger('agent.synthesis')
timing_logger = logging.getLogger('timing')


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
            classification_logger.info(f"Agent classification: analyzing query type")
            plugins_to_use = self._classify_query(question)
            
            if not plugins_to_use:
                classification_logger.info("No suitable plugins found for query")
                return "No relevant information found."
            
            classification_logger.info(f"Selected plugins: {', '.join(plugins_to_use)}")
            
            # Execute plugins and collect results
            results = []
            for plugin_name in plugins_to_use:
                plugin = self.registry.get_plugin(plugin_name)
                if plugin:
                    try:
                        execution_logger.info(f"Executing plugin: {plugin_name}")
                        self._reasoning_trace.append(f"Executing plugin: {plugin_name}")
                        params = self._prepare_params(plugin_name, question)
                        
                        if plugin.validate_params(params):
                            result = plugin.execute(params)
                            results.append((plugin_name, result))
                            self._last_plugins_used.append(plugin_name)
                            execution_logger.info(f"Plugin {plugin_name} completed successfully")
                        else:
                            logger.warning(f"Invalid parameters for plugin {plugin_name}")
                            execution_logger.info(f"Plugin {plugin_name} failed: invalid parameters")
                    except Exception as e:
                        logger.error(f"Error executing plugin {plugin_name}: {e}")
                        execution_logger.info(f"Plugin {plugin_name} failed: {e}")
                        self._reasoning_trace.append(f"Plugin {plugin_name} failed: {e}")
            
            # Synthesize final response
            synthesis_logger.info(f"Synthesizing response from {len(results)} plugin results")
            response = self._synthesize_response(question, results)
            
            self._last_execution_time = time.time() - start_time
            timing_logger.info(f"Total execution time: {self._last_execution_time:.2f}s")
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
        """Enhanced query classification with Phase III multi-step planning support.
        
        Args:
            question: User question to classify
            
        Returns:
            List of plugin names to execute for this query
        """
        question_lower = question.lower()
        plugins_to_use = []
        
        # Enhanced email query detection - use word boundaries to avoid false matches
        email_indicators = [
            "email", "emails", "mail", "sender", "sent", "received",
            "from", "to", "subject", "message"
        ]
        
        # Enhanced metadata keywords
        metadata_keywords = [
            "how many", "count", "number of", "total", "list", "show me",
            "what files", "file types", "recently", "latest", "newest", 
            "recent files", "recent documents", "size", "larger", "smaller",
            "modified", "created", "last week", "last month", "yesterday",
            "find", "search", "get", "show", "all files", "files", "documents",
            "pdf", "pptx", "docx", "xlsx", "msg", "txt"
        ]
        
        # Document discovery keywords - queries looking for specific documents by name/title
        document_discovery_keywords = [
            "find the", "get the", "show me the", "where is the", "locate the",
            "find document", "find file", "get document", "get file", 
            "document called", "file called", "document named", "file named",
            "document about", "file about", "document titled", "file titled",
            "the document", "the file", "which document", "which file"
        ]
        
        # Document-level query keywords (NEW)
        document_level_keywords = [
            "document", "documents", "file", "files", "source", "sources",
            "from the", "according to", "in the document", "the document says",
            "as mentioned in", "referenced in", "cited in", "from document",
            "which document", "what document", "document contains", "document about",
            "full document", "entire document", "complete document", "whole document"
        ]
        
        # Cross-document analysis keywords (NEW)  
        cross_document_keywords = [
            "across documents", "multiple documents", "different documents", 
            "all documents", "various documents", "between documents",
            "compare documents", "contrast documents", "documents mention",
            "throughout", "across all", "in various", "different sources"
        ]
        
        # Comprehensive analysis keywords (NEW)
        comprehensive_keywords = [
            "comprehensive", "detailed", "thorough", "complete", "full analysis",
            "in depth", "extensive", "elaborate", "exhaustive", "all information",
            "everything about", "all details", "complete overview", "full picture"
        ]
        
        # Phase III relationship analysis keywords
        relationship_keywords = [
            "similar", "related", "relationship", "connection", "linked",
            "cross-reference", "references", "mentions", "clusters", "groups",
            "theme", "themes", "patterns", "evolution", "changes over time",
            "citations", "network", "graph"
        ]
        
        # Phase III reporting keywords
        reporting_keywords = [
            "report", "summary", "analysis", "overview", "statistics", "stats",
            "trends", "insights", "dashboard", "metrics", "analytics",
            "health", "quality", "performance", "usage", "activity"
        ]
        
        # Check for email-specific queries using word boundaries
        has_email_indicators = any(re.search(r'\b' + re.escape(keyword) + r'\b', question_lower) 
                                  for keyword in email_indicators)
        
        # Check for metadata queries
        has_metadata_indicators = any(keyword in question_lower for keyword in metadata_keywords)
        
        # Check for document discovery queries (NEW) - prioritize these for metadata
        has_document_discovery_indicators = any(keyword in question_lower for keyword in document_discovery_keywords)
        
        # Check for relationship analysis queries
        has_relationship_indicators = any(keyword in question_lower for keyword in relationship_keywords)
        
        # Check for reporting queries
        has_reporting_indicators = any(keyword in question_lower for keyword in reporting_keywords)
        
        # Check for document-level queries (NEW)
        has_document_level_indicators = any(keyword in question_lower for keyword in document_level_keywords)
        
        # Check for cross-document analysis queries (NEW)
        has_cross_document_indicators = any(keyword in question_lower for keyword in cross_document_keywords)
        
        # Check for comprehensive analysis queries (NEW)
        has_comprehensive_indicators = any(keyword in question_lower for keyword in comprehensive_keywords)
        
        # Multi-step query detection - queries that might need multiple plugins
        multi_step_patterns = [
            "latest email about",
            "recent email regarding", 
            "emails about .* and related files",
            "find .* and show",
            "emails .* plus",
            "analyze .* and report",
            "summarize .* relationships",
            "report on .* activity"
        ]
        
        is_multi_step = any(re.search(pattern, question_lower) for pattern in multi_step_patterns)
        
        # Log classification analysis
        classification_logger.debug(f"Query analysis - email: {has_email_indicators}, metadata: {has_metadata_indicators}, "
                                   f"document_discovery: {has_document_discovery_indicators}, relationships: {has_relationship_indicators}, "
                                   f"reporting: {has_reporting_indicators}, document_level: {has_document_level_indicators}, "
                                   f"cross_document: {has_cross_document_indicators}, comprehensive: {has_comprehensive_indicators}, "
                                   f"multi-step: {is_multi_step}")
        
        # Phase III enhanced classification logic
        
        # Document discovery queries - PRIORITY: Route primarily to metadata with semantic as backup
        if has_document_discovery_indicators:
            # Document discovery queries should prioritize metadata search
            if self.registry.get_plugin("metadata"):
                plugins_to_use.append("metadata")
                self._reasoning_trace.append("Detected document discovery query - prioritizing metadata search")
                classification_logger.info("Classified as document discovery query - using metadata priority")
            
            # Add semantic search as secondary for document discovery
            if self.registry.get_plugin("semantic_search"):
                plugins_to_use.append("semantic_search")
                self._reasoning_trace.append("Adding semantic search as backup for document discovery")
                classification_logger.info("Added semantic search as backup for document discovery")
        
        elif has_reporting_indicators:
            # Queries asking for reports, summaries, or analytics
            if self.registry.get_plugin("comprehensive_reporting"):
                plugins_to_use.append("comprehensive_reporting")
                self._reasoning_trace.append("Detected reporting/analytics query")
                classification_logger.info("Classified as reporting query")
        
        if has_relationship_indicators:
            # Queries asking about document relationships, similarities, or patterns
            if self.registry.get_plugin("document_relationships"):
                plugins_to_use.append("document_relationships")
                self._reasoning_trace.append("Detected relationship analysis query")
                classification_logger.info("Classified as relationship analysis query")
        
        if has_email_indicators and has_metadata_indicators:
            # Queries like "emails from John last week" - primarily metadata
            if self.registry.get_plugin("metadata"):
                plugins_to_use.append("metadata")
                self._reasoning_trace.append("Detected email metadata query")
                classification_logger.info("Classified as email metadata query")
        
        elif has_email_indicators:
            # Pure email queries - route to metadata commands for enhanced processing
            if self.registry.get_plugin("metadata"):
                plugins_to_use.append("metadata")
                self._reasoning_trace.append("Detected email-specific query")
                classification_logger.info("Classified as email-specific query")
        
        elif has_metadata_indicators:
            # Pure metadata queries - prefer enhanced metadata commands
            if self.registry.get_plugin("metadata"):
                plugins_to_use.append("metadata")
                self._reasoning_trace.append("Detected metadata query keywords")
                classification_logger.info("Classified as metadata query")
        
        # Enhanced document-level and cross-document classification (NEW)
        if has_cross_document_indicators or has_comprehensive_indicators:
            # Cross-document or comprehensive queries benefit from multiple plugins
            if has_relationship_indicators and self.registry.get_plugin("document_relationships"):
                if "document_relationships" not in plugins_to_use:
                    plugins_to_use.append("document_relationships")
                    self._reasoning_trace.append("Added document relationships for cross-document analysis")
                    classification_logger.info("Added document relationships for cross-document query")
            
            # Always use semantic search for cross-document queries with enhanced parameters
            if self.registry.get_plugin("semantic_search"):
                if "semantic_search" not in plugins_to_use:
                    plugins_to_use.append("semantic_search")
                    self._reasoning_trace.append("Added semantic search for cross-document analysis")
                    classification_logger.info("Added semantic search for cross-document query")
            
            # Add metadata for document discovery if relevant
            if any(word in question.lower() for word in ["which documents", "what files", "list documents"]):
                if self.registry.get_plugin("metadata") and "metadata" not in plugins_to_use:
                    plugins_to_use.append("metadata")
                    self._reasoning_trace.append("Added metadata for document discovery")
                    classification_logger.info("Added metadata for document discovery")
        
        elif has_document_level_indicators:
            # Document-level queries that ask about specific documents or source attribution
            if self.registry.get_plugin("semantic_search"):
                if "semantic_search" not in plugins_to_use:
                    plugins_to_use.append("semantic_search")
                    self._reasoning_trace.append("Added semantic search for document-level query")
                    classification_logger.info("Added semantic search for document-level query")
        
        # Check for semantic search queries
        content_keywords = [
            "what is", "explain", "about", "describe", "compliance", "policy",
            "procedure", "requirements", "contains", "mentions", "discusses",
            "content", "text", "information"
        ]
        
        # Check for Knowledge Graph queries - NEW
        entity_keywords = [
            "who", "who is", "who works", "who manages", "who leads",
            "what company", "what organization", "what org", 
            "people", "person", "employee", "staff", "team member",
            "organization", "company", "org", "corporation", "corp",
            "works at", "works for", "employed by", "manages", "leads",
            "related to", "connected to", "associated with",
            "relationships between", "how are", "connection between"
        ]
        
        has_entity_indicators = any(keyword in question_lower for keyword in entity_keywords)
        
        # Knowledge Graph routing logic
        if has_entity_indicators and self.registry.get_plugin("knowledge_graph"):
            plugins_to_use.append("knowledge_graph")
            self._reasoning_trace.append("Detected entity/relationship query - using knowledge graph")
            classification_logger.info("Classified as knowledge graph query")
            
            # For entity queries, also use semantic search for hybrid results
            if self.registry.get_plugin("semantic_search"):
                plugins_to_use.append("semantic_search")
                self._reasoning_trace.append("Adding semantic search for hybrid entity search")
                classification_logger.info("Added semantic search for hybrid entity results")
        
        is_content_query = (
            any(keyword in question_lower for keyword in content_keywords) or
            (not plugins_to_use and not any(word in question_lower for word in [
                "files", "documents", "count", "list", "show", "how many", "types", "email", "report", "summary"
            ]))  # Default to semantic search if no clear indicators
        )
        
        # Add semantic search for content queries or multi-step queries
        if (is_content_query or is_multi_step) and self.registry.get_plugin("semantic_search"):
            if "semantic_search" not in plugins_to_use:
                plugins_to_use.append("semantic_search")
                self._reasoning_trace.append("Using semantic search for content analysis")
                classification_logger.info("Added semantic search for content analysis")
        
        # Special handling for complex queries that benefit from multiple plugins
        complex_patterns = [
            "about .* files",  # "about budget files" - content + metadata
            "documents .* recent",  # "documents about X recent" - content + time filter
            "find .* and list",  # "find X and list files" - content + metadata
            "analyze .* and summarize",  # "analyze X and summarize" - relationships + reporting
            "report on .* relationships",  # reporting + relationships
            "trends in .* documents"  # reporting + relationships + content
        ]
        
        is_complex = any(re.search(pattern, question_lower) for pattern in complex_patterns)
        
        if is_complex:
            # Add complementary plugins for complex queries
            for plugin_name in ["metadata", "semantic_search", "document_relationships"]:
                if self.registry.get_plugin(plugin_name) and plugin_name not in plugins_to_use:
                    plugins_to_use.append(plugin_name)
                    self._reasoning_trace.append(f"Added {plugin_name} for complex query")
                    classification_logger.info(f"Added {plugin_name} for complex query analysis")
        
        # If no plugins selected, default to semantic search
        if not plugins_to_use and self.registry.get_plugin("semantic_search"):
            plugins_to_use.append("semantic_search")
            self._reasoning_trace.append("Defaulting to semantic search")
            classification_logger.info("No specific indicators found - defaulting to semantic search")
        
        classification_logger.info(f"Final plugin selection: {plugins_to_use}")
        return plugins_to_use
    
    def _prepare_params(self, plugin_name: str, question: str) -> Dict[str, Any]:
        """Prepare parameters for plugin execution with enhanced document-level support.
        
        Args:
            plugin_name: Name of the plugin to prepare parameters for
            question: Original user question
            
        Returns:
            Dictionary of parameters for the plugin
        """
        # Plugin-specific parameter preparation
        if plugin_name == "metadata":
            # Use LLM to generate structured metadata commands
            return self._generate_metadata_params_with_llm(question)
        
        elif plugin_name == "document_relationships":
            # Generate parameters for relationship analysis
            return self._generate_relationship_params(question)
        
        elif plugin_name == "comprehensive_reporting":
            # Generate parameters for reporting
            return self._generate_reporting_params(question)
        
        elif plugin_name == "knowledge_graph":
            # Generate parameters for knowledge graph operations
            return self._generate_kg_params(question)
        
        elif plugin_name == "semantic_search":
            # Enhanced parameters for semantic search with document-level analysis
            return self._generate_semantic_search_params(question)
        
        # Basic parameters for other plugins
        params = {
            "question": question,
            "query": question,  # Alias for compatibility
        }
        
        return params
    
    def _generate_semantic_search_params(self, question: str) -> Dict[str, Any]:
        """Generate enhanced parameters for semantic search plugin.
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary of parameters optimized for the query type
        """
        question_lower = question.lower()
        
        # Detect document discovery queries
        document_discovery_patterns = [
            "find the", "get the", "show me the", "where is the", "locate the",
            "find document", "find file", "get document", "get file",
            "document called", "file called", "document named", "file named",
            "document about", "file about", "document titled", "file titled"
        ]
        
        is_document_discovery = any(pattern in question_lower for pattern in document_discovery_patterns)
        
        # Base parameters
        params = {
            "question": question,
            "use_document_level": True,  # Enable document-level retrieval by default
            "k": 50,  # Increased for document-level analysis
            "max_documents": 5,
            "context_window": 3,
            "include_metadata_search": is_document_discovery  # NEW: Include metadata in search for document discovery
        }
        
        # Adjust parameters based on query characteristics
        
        # Complex analytical queries need more documents and context
        if any(word in question_lower for word in [
            "analyze", "compare", "relationship", "across", "between", "comprehensive",
            "detailed", "thorough", "explain", "describe", "overview"
        ]):
            params["max_documents"] = 7
            params["context_window"] = 5
            params["k"] = 75
            self._reasoning_trace.append("Using expanded search parameters for analytical query")
        
        # Simple factual queries can use focused search
        elif any(word in question_lower for word in [
            "what is", "when", "where", "who", "quick", "brief", "simple"
        ]):
            params["max_documents"] = 3
            params["context_window"] = 2
            params["k"] = 30
            self._reasoning_trace.append("Using focused search parameters for factual query")
        
        # Document discovery queries need metadata-aware search
        elif is_document_discovery:
            params["max_documents"] = 10  # More documents to find the right one
            params["context_window"] = 2  # Less context needed for discovery
            params["k"] = 100  # Cast a wider net
            params["include_metadata_search"] = True
            self._reasoning_trace.append("Using metadata-aware search parameters for document discovery")
        
        # Multi-document queries need broader search
        elif any(word in question_lower for word in [
            "all", "every", "across documents", "in all", "throughout", "multiple",
            "various", "different documents", "all files"
        ]):
            params["max_documents"] = 10
            params["context_window"] = 4
            params["k"] = 100
            self._reasoning_trace.append("Using broad search parameters for multi-document query")
        
        # Technical/detailed queries need more context
        elif any(word in question_lower for word in [
            "implementation", "technical", "specifications", "requirements", "details",
            "procedure", "process", "methodology", "architecture", "design"
        ]):
            params["max_documents"] = 6
            params["context_window"] = 4
            params["k"] = 60
            self._reasoning_trace.append("Using detailed search parameters for technical query")
        
        # Recent/temporal queries might benefit from more documents to find latest info
        elif any(word in question_lower for word in [
            "recent", "latest", "new", "updated", "current", "now", "today",
            "this week", "this month", "recently"
        ]):
            params["max_documents"] = 8
            params["context_window"] = 3
            params["k"] = 70
            self._reasoning_trace.append("Using expanded search for temporal query")
        
        # Check if user is asking for legacy-style response
        if any(phrase in question_lower for phrase in [
            "quick answer", "just tell me", "simple answer", "briefly"
        ]):
            params["use_document_level"] = False
            self._reasoning_trace.append("Using legacy mode for simple query")
        
        return params
    
    def _generate_relationship_params(self, question: str) -> Dict[str, Any]:
        """Generate parameters for document relationship analysis.
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary of parameters for DocumentRelationshipPlugin
        """
        question_lower = question.lower()
        
        # Determine the operation type based on question
        if any(word in question_lower for word in ["similar", "related", "like"]):
            operation = "find_similar_documents"
        elif any(word in question_lower for word in ["cluster", "group", "categorize"]):
            operation = "cluster_documents"
        elif any(word in question_lower for word in ["reference", "citation", "mention"]):
            operation = "detect_cross_references"
        elif any(word in question_lower for word in ["theme", "topic", "subject"]):
            operation = "analyze_themes"
        elif any(word in question_lower for word in ["evolution", "change", "history"]):
            operation = "track_content_evolution"
        elif any(word in question_lower for word in ["cite", "citation"]):
            operation = "find_citations"
        else:
            operation = "analyze_relationships"  # Comprehensive analysis
        
        # Extract additional parameters
        params = {
            "operation": operation,
            "query": question
        }
        
        # Add specific parameters based on operation
        if operation == "find_similar_documents":
            # Look for document path or use query
            params["query_text"] = question
            params["threshold"] = 0.7
            params["max_results"] = 10
        
        elif operation == "cluster_documents":
            # Extract number of clusters if mentioned
            import re
            # Look for patterns like "3 clusters", "into 3 groups", "cluster into 3"
            cluster_match = re.search(r'(?:into\s+)?(\d+)\s*(?:cluster|group)', question_lower) or \
                           re.search(r'(\d+)\s*cluster', question_lower)
            params["num_clusters"] = int(cluster_match.group(1)) if cluster_match else 5
        
        elif operation == "track_content_evolution":
            # Extract time window
            if "week" in question_lower:
                params["time_window"] = "1_week"
            elif "month" in question_lower:
                params["time_window"] = "1_month"
            else:
                params["time_window"] = "3_months"
        
        return params
    
    def _generate_reporting_params(self, question: str) -> Dict[str, Any]:
        """Generate parameters for comprehensive reporting.
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary of parameters for ComprehensiveReportingPlugin
        """
        question_lower = question.lower()
        
        # Determine the report type based on question
        if any(word in question_lower for word in ["summary", "overview", "collection"]):
            operation = "generate_collection_summary"
        elif any(word in question_lower for word in ["theme", "topic", "subject", "thematic"]):
            operation = "generate_thematic_analysis"
        elif any(word in question_lower for word in ["activity", "recent", "changes", "new"]):
            operation = "generate_activity_report"
        elif any(word in question_lower for word in ["insight", "connection", "relationship"]):
            operation = "generate_insights_report"
        elif any(word in question_lower for word in ["trend", "pattern", "evolution"]):
            operation = "generate_trend_analysis"
        elif any(word in question_lower for word in ["usage", "analytics", "performance"]):
            operation = "generate_usage_analytics"
        elif any(word in question_lower for word in ["health", "quality", "status"]):
            operation = "generate_health_report"
        elif any(word in question_lower for word in ["custom", "specific"]):
            operation = "generate_custom_report"
        else:
            operation = "generate_collection_summary"  # Default
        
        # Base parameters
        params = {
            "operation": operation,
            "query": question
        }
        
        # Add specific parameters based on operation
        if operation == "generate_activity_report":
            # Extract time window
            if "day" in question_lower or "daily" in question_lower:
                params["time_window"] = "1_day"
            elif "week" in question_lower or "weekly" in question_lower:
                params["time_window"] = "1_week"
            elif "month" in question_lower or "monthly" in question_lower:
                params["time_window"] = "1_month"
            else:
                params["time_window"] = "1_week"  # Default
        
        elif operation == "generate_trend_analysis":
            # Set time periods for trend analysis
            params["time_periods"] = ["1_week", "1_month", "3_months"]
        
        elif operation == "generate_custom_report":
            # Extract any specific criteria mentioned
            params["criteria"] = {"query": question}
            params["title"] = "Custom Analysis Report"
        
        return params
    
    def _generate_metadata_params_with_llm(self, question: str) -> Dict[str, Any]:
        """Use LLM to generate structured metadata command parameters.
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary of structured parameters for MetadataCommandsPlugin
        """
        llm_logger = logging.getLogger('llm.generation')
        
        try:
            from openai import OpenAI
            from shared.config import get_settings
            
            settings = get_settings()
            if not settings.openai_api_key:
                self._reasoning_trace.append("No OpenAI API key - using fallback parsing")
                llm_logger.info("No OpenAI API key - using fallback parsing")
                return self._fallback_metadata_params(question)
            
            # Initialize OpenAI client
            client = OpenAI(api_key=settings.openai_api_key)
            
            # Create a prompt for the LLM to understand the query and generate structured parameters
            prompt = f"""You are a document metadata query parser. Parse the user's natural language question and convert it to structured parameters for a metadata plugin.

Available operations:
- find_files: Universal file finder with comprehensive filtering (RECOMMENDED)
- get_latest_files: Get most recently modified files (legacy)
- find_files_by_content: Search within file content (legacy)
- get_file_stats: Get statistics about files (legacy)
- get_file_count: Count files with filters (legacy)
- get_file_types: List file types in collection (legacy)

Available file types: PDF, DOCX, PPTX, XLSX, MSG, TXT

Available time filters: recent, last_week, last_month, yesterday, today

User question: "{question}"

Response must be valid JSON in this exact format:
{{
    "operation": "find_files",
    "file_type": "TYPE or null",
    "count": number_or_null,
    "time_filter": "filter_or_null",
    "keywords": ["keyword1", "keyword2"] or null
}}

Examples:
"list all pdf files" -> {{"operation": "find_files", "file_type": "PDF", "count": null, "time_filter": null, "keywords": null}}
"show me 5 latest emails" -> {{"operation": "find_files", "file_type": "MSG", "count": 5, "time_filter": "recent", "keywords": null}}
"how many documents from last week" -> {{"operation": "find_files", "file_type": null, "count": null, "time_filter": "last_week", "keywords": null}}

Respond with only the JSON, no other text:"""

            llm_logger.debug(f"LLM prompt: {prompt[:100]}...")
            
            # Get response from LLM
            llm_logger.info("LLM parsing with GPT-4o-mini...")
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=200,
                temperature=0.1  # Low temperature for consistent parsing
            )
            
            # Parse the JSON response
            import json
            params_text = response.choices[0].message.content
            if params_text is None:
                self._reasoning_trace.append("LLM returned empty response")
                llm_logger.info("LLM returned empty response")
                return self._fallback_metadata_params(question)
            
            params_text = params_text.strip()
            llm_logger.debug(f"LLM response: {params_text}")
            
            # Clean up the response if needed
            if params_text.startswith('```json'):
                params_text = params_text.replace('```json', '').replace('```', '').strip()
            
            params = json.loads(params_text)
            
            # Validate the parsed parameters
            required_keys = ['operation']
            for key in required_keys:
                if key not in params:
                    self._reasoning_trace.append(f"LLM parsing missing required key: {key}")
                    llm_logger.info(f"LLM parsing missing required key: {key}")
                    return self._fallback_metadata_params(question)
            
            self._reasoning_trace.append(f"LLM parsed query to operation: {params.get('operation')}")
            llm_logger.info(f"LLM parsed query to operation: {params.get('operation')}")
            
            # Create detailed log entry for debug level
            llm_logger.debug("", extra={
                'llm_model': 'GPT-4o-mini',
                'llm_prompt': prompt[:200],
                'msg': f"Response: {params_text}"
            })
            
            return params
            
        except Exception as e:
            self._reasoning_trace.append(f"LLM parsing failed: {e}")
            llm_logger.info(f"LLM parsing failed: {e}")
            return self._fallback_metadata_params(question)
    
    def _fallback_metadata_params(self, question: str) -> Dict[str, Any]:
        """Fallback parameter generation using simple keyword matching.
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary of basic parameters
        """
        question_lower = question.lower()
        
        # Simple operation detection
        if any(word in question_lower for word in ['latest', 'recent', 'newest']):
            operation = 'find_files'
        elif any(word in question_lower for word in ['how many', 'count', 'number']):
            operation = 'find_files'
        elif any(word in question_lower for word in ['list', 'show', 'find']):
            operation = 'find_files'
        else:
            operation = 'find_files'
        
        # Simple file type detection
        file_type = None
        if any(word in question_lower for word in ['email', 'mail', 'msg']):
            file_type = 'MSG'
        elif any(word in question_lower for word in ['pdf']):
            file_type = 'PDF'
        elif any(word in question_lower for word in ['doc', 'docx', 'document']):
            file_type = 'DOCX'
        elif any(word in question_lower for word in ['ppt', 'pptx', 'presentation']):
            file_type = 'PPTX'
        elif any(word in question_lower for word in ['xls', 'xlsx', 'spreadsheet']):
            file_type = 'XLSX'
        
        # Simple count detection
        count = None
        import re
        count_match = re.search(r'\b(\d+)\b', question)
        if count_match:
            count = int(count_match.group(1))
        
        # Simple time filter detection
        time_filter = None
        if 'last week' in question_lower:
            time_filter = 'last_week'
        elif 'last month' in question_lower:
            time_filter = 'last_month'
        elif any(word in question_lower for word in ['recent', 'latest']):
            time_filter = 'recent'
        
        return {
            'operation': operation,
            'file_type': file_type,
            'count': count,
            'time_filter': time_filter,
            'keywords': None
        }
    
    def _generate_kg_params(self, question: str) -> Dict[str, Any]:
        """Generate parameters for knowledge graph operations.
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary of parameters for KnowledgeGraphPlugin
        """
        question_lower = question.lower()
        
        # Determine the operation type based on question patterns
        if any(word in question_lower for word in ["who works", "who is", "employee", "staff"]):
            operation = "find_entities"
            entity_type = "person"
        elif any(word in question_lower for word in ["what company", "what organization", "organization", "corp"]):
            operation = "find_entities"
            entity_type = "organization"
        elif any(word in question_lower for word in ["project", "initiative", "program"]):
            operation = "find_entities"
            entity_type = "topic"
        elif any(word in question_lower for word in ["where", "location", "city", "office"]):
            operation = "find_entities"
            entity_type = "location"
        elif any(word in question_lower for word in ["related to", "connected", "relationship", "how are"]):
            operation = "hybrid_search"  # Use hybrid search for relationship queries
        elif any(word in question_lower for word in ["statistics", "stats", "overview"]):
            operation = "get_statistics"
        else:
            # Default to hybrid search for general entity questions
            operation = "hybrid_search"
            
        params = {
            "operation": operation,
            "question": question,
            "max_depth": 2,
            "max_entities": 10
        }
        
        # Add entity type if we determined one
        if operation == "find_entities" and 'entity_type' in locals():
            params["entity_type"] = entity_type
            
        return params
    
    def _synthesize_response(self, question: str, results: List[tuple]) -> str:
        """Enhanced response synthesis with intelligent document discovery support.
        
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
        knowledge_graph_results = []
        other_results = []
        
        for plugin_name, result in results:
            if plugin_name == "metadata":
                metadata_results.append(result)
            elif plugin_name == "semantic_search":
                content_results.append(result)
            elif plugin_name == "knowledge_graph":
                knowledge_graph_results.append(result)
            else:
                other_results.append((plugin_name, result))
        
        # Detect document discovery queries - queries looking for specific documents by name/title
        question_lower = question.lower()
        document_discovery_patterns = [
            "find the", "get the", "show me the", "where is the", "locate the",
            "find document", "find file", "get document", "get file",
            "document called", "file called", "document named", "file named",
            "document about", "file about", "document titled", "file titled"
        ]
        
        is_document_discovery = any(pattern in question_lower for pattern in document_discovery_patterns)
        
        # Also detect by document/file keywords combined with specific terms
        has_document_keywords = any(word in question_lower for word in ["document", "file", "the"])
        has_specific_terms = any(word in question_lower for word in ["find", "get", "show", "where", "locate"])
        is_document_discovery = is_document_discovery or (has_document_keywords and has_specific_terms)
        
        synthesis_logger.info(f"Document discovery query detected: {is_document_discovery}")
        
        # Enhanced synthesis logic with document discovery prioritization
        response_parts = []
        metadata_found_documents = False
        semantic_found_content = False
        
        # Analyze what each plugin found
        if metadata_results:
            for metadata_result in metadata_results:
                metadata_response = metadata_result.get("response", "")
                if metadata_response and not metadata_response.startswith("❌") and not "No files found" in metadata_response:
                    metadata_found_documents = True
                    self._reasoning_trace.append("Metadata plugin found matching documents")
                    
        if content_results:
            for content_result in content_results:
                content_response = content_result.get("response", "")
                if content_response and not content_response.startswith("❌") and "No relevant information found" not in content_response:
                    semantic_found_content = True
                    self._reasoning_trace.append("Semantic search found matching content")
        
        # Smart prioritization based on query type and results
        if is_document_discovery:
            synthesis_logger.info("Applying document discovery synthesis strategy")
            
            if metadata_found_documents:
                # For document discovery queries, prioritize metadata results
                synthesis_logger.info("Prioritizing metadata results for document discovery")
                for metadata_result in metadata_results:
                    metadata_response = metadata_result.get("response", "")
                    if metadata_response and not metadata_response.startswith("❌"):
                        response_parts.append(metadata_response)
                        
                # Add semantic content if it provides additional value
                if semantic_found_content:
                    for content_result in content_results:
                        content_response = content_result.get("response", "")
                        if content_response and not content_response.startswith("❌") and "No relevant information found" not in content_response:
                            response_parts.append(f"\n📄 Related Content:\n{content_response}")
                            
            elif semantic_found_content:
                # Metadata didn't find documents, but semantic search did - use semantic results
                synthesis_logger.info("Using semantic results as fallback for document discovery")
                for content_result in content_results:
                    content_response = content_result.get("response", "")
                    if content_response and not content_response.startswith("❌"):
                        response_parts.append(content_response)
            else:
                # Neither found anything useful for document discovery
                return "No relevant documents found matching your query."
                
        else:
            # Standard content-focused synthesis for non-document-discovery queries
            synthesis_logger.info("Applying standard content synthesis strategy")
            
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
                        if metadata_results and metadata_found_documents:
                            # If we have metadata context, introduce content section
                            response_parts.append(f"\n📄 Content Analysis:\n{content_response}")
                        else:
                            response_parts.append(content_response)
        
        # Add knowledge graph and other results
        for kg_result in knowledge_graph_results:
            kg_response = kg_result.get("response", "")
            if kg_response and not kg_response.startswith("❌"):
                response_parts.append(f"\n🔗 Knowledge Graph:\n{kg_response}")
                
        for plugin_name, other_result in other_results:
            other_response = other_result.get("response", "")
            if other_response and not other_response.startswith("❌"):
                response_parts.append(f"\n📋 {plugin_name.title()}:\n{other_response}")
        
        # Combine results intelligently
        if response_parts:
            synthesis_logger.info(f"Successfully synthesized response from {len(response_parts)} parts")
            # For queries that ask for both metadata and content, structure the response
            if len(response_parts) > 1 and any(word in question.lower() for word in ["and", "plus", "also"]):
                return "\n\n".join(response_parts)
            else:
                return "\n\n".join(response_parts)
        else:
            synthesis_logger.info("No valid response parts found")
            return "No relevant information found."