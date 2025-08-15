"""Discovery agent for document and entity discovery operations."""

from typing import Dict, Any, List
import logging

from .base_agent import BaseAgent
from .execution_plan import ExecutionStep, StepResult, StepType
from .context import AgentContext
from ..registry import PluginRegistry

logger = logging.getLogger(__name__)


class DiscoveryAgent(BaseAgent):
    """Specialized agent for document and entity discovery operations.
    
    This agent focuses on finding documents, extracting metadata, and 
    identifying entities within the document collection.
    """
    
    def __init__(self, plugin_registry: PluginRegistry):
        """Initialize the discovery agent.
        
        Args:
            plugin_registry: Registry of available plugins
        """
        super().__init__("discovery")
        self.registry = plugin_registry
    
    def can_handle(self, step: ExecutionStep) -> bool:
        """Check if this agent can handle the given step type.
        
        Args:
            step: ExecutionStep to evaluate
            
        Returns:
            True if this agent can handle the step
        """
        return step.step_type in [
            StepType.DISCOVER_DOCUMENT,
            StepType.RETURN_METADATA,
            StepType.RETURN_FILE_PATH
        ]
    
    def get_capabilities(self) -> List[str]:
        """Get capabilities provided by this agent."""
        return [
            "document_discovery",
            "metadata_extraction", 
            "file_path_resolution",
            "entity_identification"
        ]
    
    def execute_step(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Execute a discovery step.
        
        Args:
            step: ExecutionStep to execute
            context: AgentContext for communication
            
        Returns:
            StepResult with discovery results
        """
        if step.step_type == StepType.DISCOVER_DOCUMENT:
            return self._discover_documents(step, context)
        elif step.step_type == StepType.RETURN_METADATA:
            return self._return_metadata(step, context)
        elif step.step_type == StepType.RETURN_FILE_PATH:
            return self._return_file_paths(step, context)
        else:
            return self._create_failure_result(
                step, f"Unsupported step type: {step.step_type.value}"
            )
    
    def _discover_documents(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Discover documents based on search criteria.
        
        Args:
            step: ExecutionStep with discovery parameters
            context: AgentContext for storing results
            
        Returns:
            StepResult with discovered documents
        """
        self.agent_logger.info("Executing document discovery")
        
        try:
            # Prepare parameters for search
            search_params = self._prepare_discovery_params(step, context)
            
            # Determine which plugin to use based on operation
            operation = search_params.get("operation", "metadata")
            if operation == "semantic_search":
                plugin = self.registry.get_plugin("semantic_search")
                plugin_name = "semantic search"
            else:
                plugin = self.registry.get_plugin("metadata")
                plugin_name = "metadata"
            
            if not plugin:
                return self._create_failure_result(
                    step, f"{plugin_name.title()} plugin not available for document discovery"
                )
            
            # Execute search
            if not plugin.validate_params(search_params):
                return self._create_failure_result(
                    step, f"Invalid parameters for {plugin_name} search"
                )
            
            result = plugin.execute(search_params)
            
            # Debug logging
            self.agent_logger.info(f"Plugin result keys: {result.keys() if result else 'None'}")
            if result and "sources" in result:
                self.agent_logger.info(f"Found {len(result['sources'])} sources in plugin result")
            
            # Process and store results
            discovered_docs = self._process_discovery_results(result, context)
            
            # Update context with discovered documents
            for doc in discovered_docs:
                context.add_discovered_document(doc)
            
            self.agent_logger.info(f"Discovered {len(discovered_docs)} documents")
            
            return self._create_success_result(
                step,
                {
                    "discovered_documents": discovered_docs,
                    "count": len(discovered_docs),
                    "search_params": search_params,
                    "raw_response": result.get("response", "")
                },
                confidence=0.9 if discovered_docs else 0.3
            )
            
        except Exception as e:
            self.agent_logger.error(f"Document discovery failed: {e}")
            return self._create_failure_result(step, f"Document discovery error: {e}")
    
    def _return_metadata(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Return metadata information from context or fresh search.
        
        Args:
            step: ExecutionStep requesting metadata
            context: AgentContext with stored information
            
        Returns:
            StepResult with metadata information
        """
        self.agent_logger.info("Returning metadata information")
        
        try:
            # Check if we have documents in context
            if context.has_documents():
                # Format existing document metadata
                metadata_response = self._format_document_metadata(context.discovered_documents)
                return self._create_success_result(
                    step,
                    {
                        "metadata": context.discovered_documents,
                        "formatted_response": metadata_response,
                        "source": "context"
                    },
                    confidence=0.9
                )
            
            # No documents in context and this is a metadata query, use metadata plugin directly
            intent = step.parameters.get("intent", "")
            if intent == "metadata_query":
                return self._execute_metadata_command(step, context)
            
            # Otherwise, perform fresh document discovery
            return self._discover_documents(step, context)
            
        except Exception as e:
            return self._create_failure_result(step, f"Metadata retrieval error: {e}")
    
    def _execute_metadata_command(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Execute metadata commands directly using the metadata plugin.
        
        Args:
            step: ExecutionStep requesting metadata
            context: AgentContext with query information
            
        Returns:
            StepResult with metadata command results
        """
        self.agent_logger.info("Executing metadata command")
        
        try:
            # Find the metadata plugin
            metadata_plugin = self.registry.get_plugin("metadata")
            
            if not metadata_plugin:
                return self._create_failure_result(step, "Metadata plugin not found")
            
            # Determine appropriate metadata operation based on query
            query = step.parameters.get("query", context.query).lower()
            
            # Extract constraint parameters from step
            count = step.parameters.get("count")
            file_types = step.parameters.get("file_types", [])
            recency = step.parameters.get("recency", False)
            
            # Build metadata operation parameters
            metadata_params = {"operation": "get_latest_files"}
            
            if any(term in query for term in ["how many", "count", "number of"]):
                metadata_params["operation"] = "get_file_count"
            elif any(term in query for term in ["recent", "latest", "newest"]) or recency:
                metadata_params["operation"] = "get_latest_files" 
                if count:
                    metadata_params["count"] = count
                if file_types:
                    metadata_params["file_type"] = file_types[0] if len(file_types) == 1 else file_types
            elif any(term in query for term in ["types", "file types", "extensions"]):
                metadata_params["operation"] = "get_file_types"
            elif any(term in query for term in ["stats", "statistics", "summary"]):
                metadata_params["operation"] = "get_file_stats"
            elif any(term in query for term in ["list", "show", "all files", "all documents"]):
                metadata_params["operation"] = "find_files"
                if count:
                    metadata_params["count"] = count
                if file_types:
                    metadata_params["file_type"] = file_types[0] if len(file_types) == 1 else file_types
            else:
                # Default to latest files for general metadata queries
                metadata_params["operation"] = "get_latest_files"
                if count:
                    metadata_params["count"] = count
                if file_types:
                    metadata_params["file_type"] = file_types[0] if len(file_types) == 1 else file_types
            
            self.agent_logger.debug(f"Metadata operation parameters: {metadata_params}")
            
            # Execute the metadata operation
            result = metadata_plugin.execute(metadata_params)
            
            # Extract response from plugin result
            if isinstance(result, dict) and "response" in result:
                formatted_response = result["response"]
            else:
                formatted_response = "Metadata operation completed"
            
            return self._create_success_result(
                step,
                {
                    "metadata_result": result,
                    "formatted_response": formatted_response,
                    "operation": metadata_params["operation"],
                    "source": "metadata_plugin",
                    "discovered_documents": result.get("files", []) if isinstance(result, dict) else []
                },
                confidence=0.8
            )
            
        except Exception as e:
            return self._create_failure_result(step, f"Metadata command error: {e}")
    
    def _return_file_paths(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Return file paths for discovered documents.
        
        Args:
            step: ExecutionStep requesting file paths
            context: AgentContext with document information
            
        Returns:
            StepResult with file paths
        """
        self.agent_logger.info("Returning file paths")
        
        try:
            if not context.has_documents():
                # Need to discover documents first
                discovery_result = self._discover_documents(step, context)
                if not discovery_result.is_successful():
                    return discovery_result
            
            # Extract paths from discovered documents
            paths = context.get_discovered_paths()
            
            if not paths:
                return self._create_failure_result(
                    step, "No document paths found"
                )
            
            # Format path response
            if len(paths) == 1:
                path_response = f"Full path: {paths[0]}"
            else:
                path_list = "\n".join(f"- {path}" for path in paths)
                path_response = f"Found {len(paths)} documents:\n{path_list}"
            
            return self._create_success_result(
                step,
                {
                    "paths": paths,
                    "formatted_response": path_response,
                    "count": len(paths)
                },
                confidence=0.95
            )
            
        except Exception as e:
            return self._create_failure_result(step, f"Path retrieval error: {e}")
    
    def _prepare_discovery_params(self, step: ExecutionStep, context: AgentContext) -> Dict[str, Any]:
        """Prepare parameters for metadata plugin discovery.
        
        Args:
            step: ExecutionStep with user parameters
            context: AgentContext with shared information
            
        Returns:
            Dictionary of parameters for metadata plugin
        """
        # Get base parameters from step
        base_params = step.parameters.copy()
        
        # Extract search terms from query if not specified
        query = base_params.get("query", context.query)
        
        # Determine search strategy based on query type
        if self._is_filename_query(query):
            # Extract actual search term from natural language query for filename matching
            search_term = self._extract_search_term(query)
            params = {
                "operation": "find_files",
                "filename_pattern": search_term
            }
        else:
            # Use semantic search for conceptual queries
            params = {
                "operation": "semantic_search",
                "question": query,
                "count": 50
            }
            
            # Enable hybrid search for entity queries (names, places, etc.)
            if self._should_use_hybrid_search(query):
                params["force_hybrid"] = True
                self.agent_logger.debug(f"Enabling hybrid search for entity query: {query}")
        
        # Merge with any specific parameters from the step
        params.update(base_params)
        
        # Add context-aware enhancements
        if context.intent and "document_discovery" in context.intent:
            params["prioritize_exact_matches"] = True
        
        self.agent_logger.debug(f"Prepared discovery params: {params}")
        return params
    
    def _is_filename_query(self, query: str) -> bool:
        """Determine if query is asking for a specific filename vs. conceptual content.
        
        Args:
            query: Natural language query
            
        Returns:
            True if query appears to be asking for a specific file by name
        """
        query_lower = query.lower()
        
        # Patterns that indicate filename queries
        filename_indicators = [
            "file with name",
            "file named",
            "file called",
            "find the file",
            "named omaha",  # specific examples
            "called omaha",
            "name omaha"
        ]
        
        # Check for filename indicator patterns
        for indicator in filename_indicators:
            if indicator in query_lower:
                return True
        
        # Check if query mentions specific file extensions
        file_extensions = ['.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.doc', '.xls', '.ppt']
        for ext in file_extensions:
            if ext in query_lower:
                return True
        
        # If none of these patterns match, assume it's a conceptual query
        return False
    
    def _should_use_hybrid_search(self, query: str) -> bool:
        """Determine if query should use hybrid search for better entity/name recall.
        
        Args:
            query: Natural language query
            
        Returns:
            True if query likely contains entities or proper names that would benefit from lexical search
        """
        query_lower = query.lower()
        
        # Patterns that suggest entity/proper noun queries
        entity_indicators = [
            # Question patterns about entities
            "what do we know about",
            "tell me about",
            "information about",
            "details about",
            "who is",
            "what is",
            # Proper noun patterns (capitalized words in queries)
            r'\b[A-Z][a-z]+\b',  # Capitalized words like "Hubli", "Amazon", etc.
        ]
        
        # Check for entity indicator patterns
        for indicator in entity_indicators:
            if isinstance(indicator, str):
                if indicator in query_lower:
                    return True
            else:
                # Regex pattern
                import re
                if re.search(indicator, query):
                    return True
        
        # Check if query contains capitalized words (likely proper nouns)
        import re
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        if len(capitalized_words) > 0:
            # Exclude common question words and general terms
            common_words = {'What', 'Who', 'Where', 'When', 'Why', 'How', 'The', 'This', 'That', 'These', 'Those'}
            proper_nouns = [word for word in capitalized_words if word not in common_words]
            if proper_nouns:
                return True
        
        return False
    
    def _extract_search_term(self, query: str) -> str:
        """Extract the actual search term from a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Extracted search term
        """
        # Convert to lowercase for pattern matching
        query_lower = query.lower()
        
        # Common patterns for file search queries
        patterns = [
            r'file.*name[d\s]*[\'"]*([^\'"\s]+)[\'"]*',  # "file named X" or "file name X"
            r'find.*file.*[\'"]*([^\'"\s]+)[\'"]*',      # "find file X"
            r'named.*[\'"]*([^\'"\s]+)[\'"]*',           # "named X"
            r'called.*[\'"]*([^\'"\s]+)[\'"]*',          # "called X"
            r'with.*name.*[\'"]*([^\'"\s]+)[\'"]*',      # "with name X"
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1)
        
        # Fallback: look for quoted terms
        quoted_match = re.search(r'[\'"]+([^\'"\s]+)[\'"]+', query)
        if quoted_match:
            return quoted_match.group(1)
        
        # Last resort: extract likely filename terms (avoid common words)
        words = query.split()
        stop_words = {'find', 'the', 'file', 'with', 'name', 'named', 'called', 'in', 'it', 'a', 'an', 'and', 'or'}
        for word in words:
            clean_word = word.strip('.,!?()[]{}"\';:')
            if clean_word.lower() not in stop_words and len(clean_word) > 2:
                return clean_word
        
        # If all else fails, return the original query
        return query
    
    def _process_discovery_results(self, result: Dict[str, Any], 
                                 context: AgentContext) -> List[Dict[str, Any]]:
        """Process plugin results into structured document information.
        
        Args:
            result: Raw result from semantic search or metadata plugin
            context: AgentContext for additional processing
            
        Returns:
            List of structured document information
        """
        discovered_docs = []
        
        # First check if this is a semantic search result (has 'sources' field)
        if "sources" in result and result["sources"]:
            # Process semantic search results
            for source in result["sources"]:
                discovered_docs.append({
                    "path": source.get("file", source.get("document_path", "")),
                    "name": source.get("document_title", self._extract_filename(source.get("file", ""))),
                    "source": "semantic_search",
                    "unit": source.get("unit", ""),
                    "similarity": source.get("similarity", 0.0),
                    "distance": source.get("distance", 1.0),
                    "chunk_index": source.get("chunk_index", 0)
                })
            return discovered_docs
        
        # Handle metadata plugin structured data response
        data = result.get("data", {})
        if data and "files" in data:
            # Use structured file data from metadata plugin
            files = data["files"]
            for file_info in files:
                discovered_docs.append({
                    "path": file_info.get("path", ""),
                    "name": file_info.get("name", ""),
                    "source": "metadata_discovery",
                    "modified": file_info.get("modified"),
                    "size": file_info.get("size"),
                    "type": file_info.get("type")
                })
            
            return discovered_docs
        
        # Fallback: Extract document information from text response
        response = result.get("response", "")
        
        # Parse response for file paths and metadata
        if "files found" in response.lower() or "documents:" in response.lower():
            # Extract file information from structured response
            lines = response.split('\n')
            current_doc = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for file paths
                if line.startswith(('/', 'C:', 'D:', '\\')) or '.docx' in line or '.pdf' in line:
                    if current_doc:
                        discovered_docs.append(current_doc)
                    
                    current_doc = {
                        "path": line,
                        "name": self._extract_filename(line),
                        "source": "metadata_discovery"
                    }
                elif current_doc and ("Size:" in line or "Modified:" in line or "Type:" in line):
                    # Add metadata to current document
                    if "Size:" in line:
                        current_doc["size"] = line.split("Size:")[1].strip()
                    elif "Modified:" in line:
                        current_doc["modified"] = line.split("Modified:")[1].strip()
                    elif "Type:" in line:
                        current_doc["type"] = line.split("Type:")[1].strip()
            
            # Add the last document
            if current_doc:
                discovered_docs.append(current_doc)
        
        # If no structured parsing worked, create basic entries
        if not discovered_docs and response and not response.startswith("❌"):
            # Try to extract any file references
            import re
            file_patterns = [
                r'([A-Z]:[\\\/][^\\\/\n]+\.(?:docx|pdf|xlsx|pptx|msg|txt))',
                r'([\/][^\/\n]+\.(?:docx|pdf|xlsx|pptx|msg|txt))',
                r'([^\\\/\s]+\.(?:docx|pdf|xlsx|pptx|msg|txt))'
            ]
            
            for pattern in file_patterns:
                matches = re.findall(pattern, response)
                for match in matches:
                    discovered_docs.append({
                        "path": match,
                        "name": self._extract_filename(match),
                        "source": "pattern_extraction"
                    })
                if matches:
                    break
        
        return discovered_docs
    
    def _extract_filename(self, file_path: str) -> str:
        """Extract filename from file path."""
        import os
        return os.path.basename(file_path)
    
    def _format_document_metadata(self, documents: List[Dict[str, Any]]) -> str:
        """Format document metadata for display.
        
        Args:
            documents: List of document information
            
        Returns:
            Formatted string representation
        """
        if not documents:
            return "No documents found."
        
        if len(documents) == 1:
            doc = documents[0]
            return f"Document: {doc.get('name', 'Unknown')}\nPath: {doc.get('path', 'Unknown')}"
        
        response_parts = [f"Found {len(documents)} documents:"]
        for i, doc in enumerate(documents, 1):
            name = doc.get('name', 'Unknown')
            path = doc.get('path', 'Unknown')
            response_parts.append(f"{i}. {name}")
            response_parts.append(f"   Path: {path}")
        
        return "\n".join(response_parts)