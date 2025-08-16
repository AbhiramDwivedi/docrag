"""Analysis agent for content analysis and extraction operations."""

from typing import Dict, Any, List, Union
import logging
import json
from collections import Counter

from .base_agent import BaseAgent
from .execution_plan import ExecutionStep, StepResult, StepType
from .context import AgentContext
from ..registry import PluginRegistry

# Try to import OpenAI for synthesis
try:
    from shared.config import settings
    from openai import OpenAI
except ImportError:
    try:
        from src.shared.config import settings
        from openai import OpenAI
    except ImportError:
        settings = None
        OpenAI = None

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    """Specialized agent for content analysis and extraction operations.
    
    This agent focuses on analyzing document content, extracting specific
    information, and performing deep content analysis.
    """
    
    def __init__(self, plugin_registry: PluginRegistry):
        """Initialize the analysis agent.
        
        Args:
            plugin_registry: Registry of available plugins
        """
        super().__init__("analysis")
        self.registry = plugin_registry
    
    def can_handle(self, step: ExecutionStep) -> bool:
        """Check if this agent can handle the given step type.
        
        Args:
            step: ExecutionStep to evaluate
            
        Returns:
            True if this agent can handle the step
        """
        return step.step_type in [
            StepType.EXTRACT_CONTENT,
            StepType.ANALYZE_CONTENT,
            StepType.ANALYZE_DECISIONS,
            StepType.COMPARE_ACROSS_DOCS,
            StepType.SYNTHESIZE_FINDINGS
        ]
    
    def get_capabilities(self) -> List[str]:
        """Get capabilities provided by this agent."""
        return [
            "content_extraction",
            "decision_analysis",
            "cross_document_comparison",
            "finding_synthesis",
            "content_summarization"
        ]
    
    def execute_step(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Execute an analysis step.
        
        Args:
            step: ExecutionStep to execute
            context: AgentContext for communication
            
        Returns:
            StepResult with analysis results
        """
        if step.step_type == StepType.EXTRACT_CONTENT:
            return self._extract_content(step, context)
        elif step.step_type == StepType.ANALYZE_CONTENT:
            return self._analyze_content(step, context)
        elif step.step_type == StepType.ANALYZE_DECISIONS:
            return self._analyze_decisions(step, context)
        elif step.step_type == StepType.COMPARE_ACROSS_DOCS:
            return self._compare_across_documents(step, context)
        elif step.step_type == StepType.SYNTHESIZE_FINDINGS:
            return self._synthesize_findings(step, context)
        else:
            return self._create_failure_result(
                step, f"Unsupported step type: {step.step_type.value}"
            )
    
    def _extract_content(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Extract content from documents using semantic search.
        
        Args:
            step: ExecutionStep with extraction parameters
            context: AgentContext with document information
            
        Returns:
            StepResult with extracted content
        """
        self.agent_logger.info("Executing content extraction")
        
        try:
            # Get semantic search plugin for content extraction
            semantic_plugin = self.registry.get_plugin("semantic_search")
            if not semantic_plugin:
                return self._create_failure_result(
                    step, "Semantic search plugin not available for content extraction"
                )
            
            # Prepare parameters for content extraction with validation
            try:
                search_params = self._prepare_extraction_params(step, context)
            except ValueError as e:
                return self._create_failure_result(step, f"Invalid extraction parameters: {e}")
            
            # Execute semantic search
            if not semantic_plugin.validate_params(search_params):
                return self._create_failure_result(
                    step, "Invalid parameters for content extraction"
                )
            
            result = semantic_plugin.execute(search_params)
            
            # Process and store results
            extracted_content = self._process_extraction_results(result, step, context)
            
            # Store in context
            source_key = f"content_extraction_{step.id}"
            context.add_extracted_content(source_key, extracted_content)
            
            self.agent_logger.info(f"Extracted content from {len(extracted_content.get('sources', []))} sources")
            
            return self._create_success_result(
                step,
                {
                    "extracted_content": extracted_content,
                    "search_params": search_params,
                    "raw_response": result.get("response", "")
                },
                confidence=0.8 if extracted_content.get("sources") else 0.3
            )
            
        except Exception as e:
            self.agent_logger.error(f"Content extraction failed: {e}")
            return self._create_failure_result(step, f"Content extraction error: {e}")
    
    def _analyze_content(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Analyze content with semantic search restricted to target documents.
        
        This method implements the content filtering step for complex queries,
        performing semantic search on a restricted set of documents.
        
        Args:
            step: ExecutionStep with content analysis parameters
            context: AgentContext with document information
            
        Returns:
            StepResult with filtered content
        """
        self.agent_logger.info("Executing content analysis with target document filtering")
        
        try:
            # Get semantic search plugin
            semantic_plugin = self.registry.get_plugin("semantic_search")
            if not semantic_plugin:
                return self._create_failure_result(
                    step, "Semantic search plugin not available for content analysis"
                )
            
            # Get target documents from previous step or parameters
            target_docs = self._get_target_documents(step, context)
            if not target_docs:
                return self._create_failure_result(
                    step, "No target documents available for content filtering"
                )
            
            # Validate and sanitize parameters
            try:
                query = self._validate_query_parameter(step.parameters.get("query", context.query))
                max_documents = self._validate_count_parameter(step.parameters.get("max_documents", 10))
                k = self._validate_count_parameter(step.parameters.get("k", 50))
            except ValueError as e:
                return self._create_failure_result(step, f"Invalid parameter: {e}")
            
            # Validate and sanitize target documents
            try:
                target_doc_paths = self._validate_target_documents(target_docs)
            except ValueError as e:
                return self._create_failure_result(step, f"Invalid target documents: {e}")
            
            search_params = {
                "question": query,
                "target_docs": target_doc_paths,
                "k": min(k, len(target_doc_paths)),
                "max_documents": max_documents
            }
            
            self.agent_logger.debug(f"Content analysis parameters: query='{query}', "
                                  f"target_docs={len(target_doc_paths)}, max_documents={max_documents}")
            
            # Execute semantic search with restricted document set
            if not semantic_plugin.validate_params(search_params):
                return self._create_failure_result(
                    step, "Invalid parameters for content analysis"
                )
            
            result = semantic_plugin.execute(search_params)
            
            # Process results with stable ordering (similarity desc, path asc)
            filtered_content = self._process_content_analysis_results(
                result, target_docs, max_documents
            )
            
            # Store in context
            source_key = f"content_analysis_{step.id}"
            context.add_extracted_content(source_key, filtered_content)
            
            self.agent_logger.info(f"Filtered content to {len(filtered_content.get('sources', []))} relevant documents")
            
            return self._create_success_result(
                step,
                {
                    "filtered_content": filtered_content,
                    "search_params": search_params,
                    "target_docs_count": len(target_doc_paths),
                    "final_count": len(filtered_content.get('sources', [])),
                    "raw_response": result.get("response", "")
                },
                confidence=0.9 if filtered_content.get("sources") else 0.2
            )
            
        except Exception as e:
            self.agent_logger.error(f"Content analysis failed: {e}")
            return self._create_failure_result(step, f"Content analysis error: {e}")
    
    def _analyze_decisions(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Analyze decisions within extracted or target content.
        
        Args:
            step: ExecutionStep with analysis parameters
            context: AgentContext with content information
            
        Returns:
            StepResult with decision analysis
        """
        self.agent_logger.info("Executing decision analysis")
        
        try:
            # First extract content if not already available
            if not context.has_content():
                extract_result = self._extract_content(step, context)
                if not extract_result.is_successful():
                    return extract_result
            
            # Analyze for decision-related content
            decision_params = self._prepare_decision_analysis_params(step, context)
            
            # Use semantic search to find decision-related content
            semantic_plugin = self.registry.get_plugin("semantic_search")
            if semantic_plugin and semantic_plugin.validate_params(decision_params):
                result = semantic_plugin.execute(decision_params)
                
                # Extract decision insights
                decisions = self._extract_decisions_from_content(result, context)
                
                # Store analysis results
                analysis_key = f"decision_analysis_{step.id}"
                context.add_extracted_content(analysis_key, {
                    "decisions": decisions,
                    "analysis_type": "decision_extraction"
                })
                
                return self._create_success_result(
                    step,
                    {
                        "decisions": decisions,
                        "analysis_summary": self._format_decision_summary(decisions),
                        "source_content": result.get("response", "")
                    },
                    confidence=0.7 if decisions else 0.4
                )
            
            return self._create_failure_result(step, "Unable to perform decision analysis")
            
        except Exception as e:
            self.agent_logger.error(f"Decision analysis failed: {e}")
            return self._create_failure_result(step, f"Decision analysis error: {e}")
    
    def _compare_across_documents(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Compare information across multiple documents.
        
        Args:
            step: ExecutionStep with comparison parameters
            context: AgentContext with document information
            
        Returns:
            StepResult with comparison analysis
        """
        self.agent_logger.info("Executing cross-document comparison")
        
        try:
            # Ensure we have multiple documents to compare
            if not context.has_documents() or len(context.discovered_documents) < 2:
                return self._create_failure_result(
                    step, "Need at least 2 documents for comparison"
                )
            
            # Extract content from each document
            comparison_results = []
            
            for doc in context.discovered_documents:
                # Create targeted search for this document
                doc_params = self._prepare_document_specific_params(step, doc, context)
                
                semantic_plugin = self.registry.get_plugin("semantic_search")
                if semantic_plugin and semantic_plugin.validate_params(doc_params):
                    result = semantic_plugin.execute(doc_params)
                    
                    comparison_results.append({
                        "document": doc,
                        "content": result.get("response", ""),
                        "analysis": self._analyze_document_content(result, step.parameters)
                    })
            
            # Perform cross-document analysis
            comparison_analysis = self._perform_cross_analysis(comparison_results, step.parameters)
            
            # Store comparison results
            comp_key = f"comparison_{step.id}"
            context.add_extracted_content(comp_key, {
                "comparison_results": comparison_results,
                "cross_analysis": comparison_analysis
            })
            
            return self._create_success_result(
                step,
                {
                    "comparison_analysis": comparison_analysis,
                    "document_results": comparison_results,
                    "summary": self._format_comparison_summary(comparison_analysis)
                },
                confidence=0.8
            )
            
        except Exception as e:
            self.agent_logger.error(f"Cross-document comparison failed: {e}")
            return self._create_failure_result(step, f"Comparison error: {e}")
    
    def _synthesize_findings(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Synthesize findings from all previous analysis steps.
        
        Args:
            step: ExecutionStep with synthesis parameters
            context: AgentContext with accumulated results
            
        Returns:
            StepResult with synthesized findings
        """
        self.agent_logger.info("Executing findings synthesis")
        
        try:
            # Gather all content from context
            all_content = context.get_extracted_content()
            
            if not all_content:
                return self._create_failure_result(
                    step, "No content available for synthesis"
                )
            
            # Synthesize findings based on query intent
            synthesis = self._create_synthesis(all_content, context.query, context.intent)
            
            # Store synthesis results
            synth_key = f"synthesis_{step.id}"
            context.add_extracted_content(synth_key, synthesis)
            
            return self._create_success_result(
                step,
                {
                    "synthesis": synthesis,
                    "final_response": synthesis.get("final_response", ""),
                    "confidence": synthesis.get("confidence", 0.7)
                },
                confidence=synthesis.get("confidence", 0.7)
            )
            
        except Exception as e:
            self.agent_logger.error(f"Findings synthesis failed: {e}")
            return self._create_failure_result(step, f"Synthesis error: {e}")
    
    # Helper methods for parameter preparation and result processing
    
    def _prepare_extraction_params(self, step: ExecutionStep, context: AgentContext) -> Dict[str, Any]:
        """Prepare parameters for content extraction with validation."""
        base_params = step.parameters.copy()
        
        # Validate query parameter
        query = self._validate_query_parameter(base_params.get("query", context.query))
        
        # Use target documents if specified
        target_docs = base_params.get("target_docs", context.get_discovered_paths())
        
        params = {
            "question": query,
            "use_document_level": True,
            "k": 50,
            "max_documents": len(target_docs) if target_docs else 5
        }
        
        # If we have specific documents, validate and focus search on them
        if target_docs:
            if isinstance(target_docs, list) and len(target_docs) > 0:
                # Validate target document paths
                validated_paths = []
                for doc in target_docs:
                    if isinstance(doc, str):
                        validated_paths.append(doc)
                    elif isinstance(doc, dict):
                        path = doc.get("path", doc.get("file_path", ""))
                        if path:
                            validated_paths.append(path)
                
                if validated_paths:
                    params["target_documents"] = validated_paths
        
        # Validate and update other parameters
        for key, value in base_params.items():
            if key in ["k", "max_documents"] and value is not None:
                params[key] = self._validate_count_parameter(value)
            elif key not in ["query", "target_docs"]:  # Already processed
                params[key] = value
        
        return params
    
    def _prepare_decision_analysis_params(self, step: ExecutionStep, context: AgentContext) -> Dict[str, Any]:
        """Prepare parameters for decision analysis."""
        base_params = step.parameters.copy()
        
        # Create decision-focused query
        decision_query = base_params.get("query", context.query)
        if "decision" not in decision_query.lower():
            decision_query = f"decisions and key choices in {decision_query}"
        
        return {
            "question": decision_query,
            "use_document_level": True,
            "k": 30,
            "max_documents": 3,
            "context_window": 4
        }
    
    def _prepare_document_specific_params(self, step: ExecutionStep, document: Dict[str, Any], 
                                        context: AgentContext) -> Dict[str, Any]:
        """Prepare parameters for document-specific search."""
        return {
            "question": step.parameters.get("query", context.query),
            "target_documents": [document.get("path", "")],
            "use_document_level": True,
            "k": 20,
            "max_documents": 1
        }
    
    def _process_extraction_results(self, result: Dict[str, Any], 
                                  step: ExecutionStep, context: AgentContext) -> Dict[str, Any]:
        """Process content extraction results with OpenAI synthesis."""
        raw_content = result.get("response", "")
        query = step.parameters.get("query", context.query)
        
        # If we have content and OpenAI is available, synthesize an answer
        if raw_content and OpenAI and settings and settings.openai_api_key:
            try:
                synthesized_response = self._synthesize_answer_with_openai(raw_content, query)
                if synthesized_response:
                    # Return synthesized answer instead of raw content
                    return {
                        "content": synthesized_response,
                        "raw_content": raw_content,
                        "sources": self._extract_sources_from_response(raw_content),
                        "extraction_type": step.parameters.get("extraction_type", "general"),
                        "query": query,
                        "synthesized": True
                    }
            except Exception as e:
                self.agent_logger.warning(f"OpenAI synthesis failed: {e}, falling back to raw content")
        
        # Fallback to raw content if synthesis fails or isn't available
        return {
            "content": raw_content,
            "sources": self._extract_sources_from_response(raw_content),
            "extraction_type": step.parameters.get("extraction_type", "general"),
            "query": query,
            "synthesized": False
        }
    
    def _synthesize_answer_with_openai(self, content: str, query: str) -> str:
        """Use OpenAI to synthesize a direct answer from the document content.
        
        Args:
            content: Raw document content from semantic search
            query: Original user query
            
        Returns:
            Synthesized answer or empty string if synthesis fails
        """
        try:
            # Initialize OpenAI client
            client = OpenAI(api_key=settings.openai_api_key)
            
            # Create synthesis prompt
            prompt = f"""Based on the following document content, please provide a direct answer to the user's question.

User Question: "{query}"

Document Content:
{content}

Instructions:
- Provide a direct, concise answer to the question
- Use information from the document content provided
- If the answer is not in the content, say "The answer is not found in the provided documents"
- Keep the response focused and relevant to the specific question asked
- Do not list documents or sources in your answer

Answer:"""

            # Get synthesis from OpenAI
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            synthesized_answer = response.choices[0].message.content.strip()
            self.agent_logger.info(f"Successfully synthesized answer with OpenAI: {synthesized_answer[:100]}...")
            
            return synthesized_answer
            
        except Exception as e:
            self.agent_logger.error(f"OpenAI synthesis error: {e}")
            return ""
    
    def _extract_decisions_from_content(self, result: Dict[str, Any], 
                                      context: AgentContext) -> List[Dict[str, Any]]:
        """Extract decision information from content."""
        content = result.get("response", "")
        decisions = []
        
        # Simple decision extraction logic
        decision_indicators = ["decided", "decision", "chose", "determined", "agreed", "resolved"]
        
        sentences = content.split('. ')
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in decision_indicators):
                decisions.append({
                    "text": sentence.strip(),
                    "type": "decision_statement",
                    "confidence": 0.7
                })
        
        return decisions
    
    def _extract_sources_from_response(self, response: str) -> List[str]:
        """Extract source information from response."""
        sources = []
        lines = response.split('\n')
        
        for line in lines:
            if 'Source:' in line or 'From:' in line:
                sources.append(line.strip())
        
        return sources
    
    def _analyze_document_content(self, result: Dict[str, Any], 
                                parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content from a single document."""
        content = result.get("response", "")
        
        return {
            "content_length": len(content),
            "has_content": bool(content.strip()),
            "analysis_focus": parameters.get("focus", "general"),
            "key_points": self._extract_key_points(content)
        }
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content."""
        # Simple key point extraction
        sentences = content.split('. ')
        key_points = []
        
        for sentence in sentences[:5]:  # Take first 5 sentences as key points
            if len(sentence.strip()) > 20:  # Only meaningful sentences
                key_points.append(sentence.strip())
        
        return key_points
    
    def _perform_cross_analysis(self, comparison_results: List[Dict[str, Any]], 
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-document analysis."""
        analysis = {
            "documents_analyzed": len(comparison_results),
            "common_themes": [],
            "differences": [],
            "summary": ""
        }
        
        # Extract common themes and differences
        all_content = [r["content"] for r in comparison_results]
        
        # Simple theme analysis
        common_words = self._find_common_terms(all_content)
        analysis["common_themes"] = common_words[:5]  # Top 5 common themes
        
        # Create summary
        analysis["summary"] = f"Analyzed {len(comparison_results)} documents. Found {len(common_words)} common themes."
        
        return analysis
    
    def _find_common_terms(self, content_list: List[str]) -> List[str]:
        """Find common terms across content."""
        from collections import Counter
        
        all_words = []
        for content in content_list:
            words = [word.lower().strip('.,!?') for word in content.split() 
                    if len(word) > 4]  # Only meaningful words
            all_words.extend(words)
        
        # Find most common terms
        counter = Counter(all_words)
        return [word for word, count in counter.most_common(10) if count > 1]
    
    def _create_synthesis(self, all_content: Dict[str, Any], query: str, intent: str) -> Dict[str, Any]:
        """Create synthesis of all findings."""
        synthesis = {
            "sources_count": len(all_content),
            "content_types": list(all_content.keys()),
            "final_response": "",
            "confidence": 0.7
        }
        
        # Create consolidated response
        response_parts = []
        
        for source, content in all_content.items():
            if isinstance(content, dict) and "content" in content:
                response_parts.append(content["content"])
            elif isinstance(content, str):
                response_parts.append(content)
        
        if response_parts:
            synthesis["final_response"] = "\n\n".join(response_parts[:3])  # Limit response
        else:
            synthesis["final_response"] = "No synthesizable content found."
            synthesis["confidence"] = 0.3
        
        return synthesis
    
    def _format_decision_summary(self, decisions: List[Dict[str, Any]]) -> str:
        """Format decision analysis summary."""
        if not decisions:
            return "No decisions identified in the content."
        
        summary_parts = [f"Found {len(decisions)} decision points:"]
        for i, decision in enumerate(decisions[:5], 1):  # Limit to 5 decisions
            summary_parts.append(f"{i}. {decision.get('text', 'Unknown decision')}")
        
        return "\n".join(summary_parts)
    
    def _format_comparison_summary(self, analysis: Dict[str, Any]) -> str:
        """Format comparison analysis summary."""
        doc_count = analysis.get("documents_analyzed", 0)
        themes = analysis.get("common_themes", [])
        
        summary = f"Compared {doc_count} documents. "
        if themes:
            summary += f"Common themes: {', '.join(themes[:3])}."
        else:
            summary += "No common themes identified."
        
        return summary
    
    def _get_target_documents(self, step: ExecutionStep, context: AgentContext) -> List[Dict[str, Any]]:
        """Get target documents from step parameters or context.
        
        Args:
            step: ExecutionStep with potential target documents reference
            context: AgentContext with stored information
            
        Returns:
            List of target document metadata
        """
        # Check if target documents are referenced from a previous step
        target_docs_step = step.parameters.get("target_docs_from_step")
        if target_docs_step:
            # Get documents from the referenced step result
            step_result = context.get_step_result(target_docs_step.id)
            if step_result and step_result.is_successful():
                result_data = step_result.get_result_data()
                if "discovered_documents" in result_data:
                    return result_data["discovered_documents"]
                elif "metadata_result" in result_data:
                    metadata_result = result_data["metadata_result"]
                    if isinstance(metadata_result, dict) and "files" in metadata_result:
                        return metadata_result["files"]
        
        # Fallback to direct target_docs parameter
        target_docs = step.parameters.get("target_docs", [])
        if target_docs:
            return [{"path": doc} for doc in target_docs]
        
        # Fallback to context discovered documents
        return context.discovered_documents or []
    
    def _process_content_analysis_results(self, result: Dict[str, Any], 
                                        target_docs: List[Dict[str, Any]], 
                                        max_documents: int) -> Dict[str, Any]:
        """Process semantic search results with stable ordering and clipping.
        
        Args:
            result: Raw semantic search results
            target_docs: Original target document list
            max_documents: Maximum documents to return
            
        Returns:
            Processed results with stable ordering
        """
        if not result or "sources" not in result:
            return {"sources": [], "response": result.get("response", "")}
        
        sources = result["sources"]
        
        # Ensure stable ordering with multiple sort keys for full determinism:
        # 1. Similarity score (desc) - primary relevance ranking
        # 2. File path (asc) - consistent ordering for ties 
        # 3. Content hash or chunk index (asc) - final tie-breaker for identical documents
        sorted_sources = sorted(sources, key=lambda x: (
            -x.get("similarity", 0),  # Negative for desc order
            x.get("file_path", x.get("path", "")),  # Path ascending for first tie-breaker
            x.get("chunk_index", x.get("id", str(hash(x.get("content", "")))))  # Final tie-breaker
        ))
        
        # Clip to requested count
        final_sources = sorted_sources[:max_documents]
        
        return {
            "sources": final_sources,
            "response": result.get("response", ""),
            "total_candidates": len(sources),
            "final_count": len(final_sources)
        }
    
    # Security validation helper methods
    
    def _validate_query_parameter(self, query: str) -> str:
        """Validate and sanitize query parameter to prevent injection attacks.
        
        Args:
            query: User-provided query string
            
        Returns:
            Sanitized query string
            
        Raises:
            ValueError: If query is invalid or dangerous
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        # Remove potential dangerous characters and limit length
        query = query.strip()
        if len(query) > 1000:  # Reasonable limit
            raise ValueError("Query too long (max 1000 characters)")
        
        # Check for potential SQL injection patterns (basic protection)
        dangerous_patterns = [
            '--', ';', 'DROP', 'DELETE', 'UPDATE', 'INSERT', 'EXEC',
            'UNION', 'SELECT', 'CREATE', 'ALTER', '<SCRIPT', 'JAVASCRIPT:'
        ]
        
        query_upper = query.upper()
        for pattern in dangerous_patterns:
            if pattern in query_upper:
                raise ValueError(f"Query contains potentially dangerous pattern: {pattern}")
        
        return query
    
    def _validate_count_parameter(self, count: Any) -> int:
        """Validate count parameter to prevent resource abuse.
        
        Args:
            count: Count parameter from user input
            
        Returns:
            Validated count as integer
            
        Raises:
            ValueError: If count is invalid
        """
        if not isinstance(count, (int, float, str)):
            raise ValueError("Count must be a number")
        
        try:
            count_int = int(count)
        except (ValueError, TypeError):
            raise ValueError("Count must be a valid integer")
        
        if count_int < 1:
            raise ValueError("Count must be positive")
        
        if count_int > 1000:  # Prevent excessive resource usage
            raise ValueError("Count too large (max 1000)")
        
        return count_int
    
    def _validate_target_documents(self, target_docs: List[Dict[str, Any]]) -> List[str]:
        """Validate and sanitize target document paths.
        
        Args:
            target_docs: List of document metadata dictionaries
            
        Returns:
            List of validated document paths
            
        Raises:
            ValueError: If document paths are invalid
        """
        if not target_docs or not isinstance(target_docs, list):
            raise ValueError("Target documents must be a non-empty list")
        
        if len(target_docs) > 1000:  # Prevent excessive processing
            raise ValueError("Too many target documents (max 1000)")
        
        validated_paths = []
        
        for i, doc in enumerate(target_docs):
            if not isinstance(doc, dict):
                raise ValueError(f"Document {i} must be a dictionary")
            
            # Extract path with fallback options
            path = doc.get("path") or doc.get("file_path") or ""
            
            if not path or not isinstance(path, str):
                raise ValueError(f"Document {i} missing valid path")
            
            path = path.strip()
            if not path:
                raise ValueError(f"Document {i} has empty path")
            
            # Basic path traversal protection
            if '..' in path or path.startswith('/'):
                # Allow absolute paths but log for security monitoring
                self.agent_logger.warning(f"Absolute path detected in document: {path}")
            
            # Prevent extremely long paths
            if len(path) > 500:
                raise ValueError(f"Document path too long: {path[:50]}...")
            
            validated_paths.append(path)
        
        return validated_paths