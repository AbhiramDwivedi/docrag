"""Analysis agent for content analysis and extraction operations."""

from typing import Dict, Any, List
import logging

from .base_agent import BaseAgent
from .execution_plan import ExecutionStep, StepResult, StepType
from .context import AgentContext
from ..registry import PluginRegistry

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
            
            # Prepare parameters for content extraction
            search_params = self._prepare_extraction_params(step, context)
            
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
        """Prepare parameters for content extraction."""
        base_params = step.parameters.copy()
        
        # Use target documents if specified
        target_docs = base_params.get("target_docs", context.get_discovered_paths())
        
        params = {
            "question": base_params.get("query", context.query),
            "use_document_level": True,
            "k": 50,
            "max_documents": len(target_docs) if target_docs else 5
        }
        
        # If we have specific documents, focus search on them
        if target_docs:
            params["target_documents"] = target_docs
        
        params.update(base_params)
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
        """Process content extraction results."""
        return {
            "content": result.get("response", ""),
            "sources": self._extract_sources_from_response(result.get("response", "")),
            "extraction_type": step.parameters.get("extraction_type", "general"),
            "query": step.parameters.get("query", context.query)
        }
    
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