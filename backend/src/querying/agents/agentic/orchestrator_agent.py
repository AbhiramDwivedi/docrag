"""Orchestrator agent for master planning and coordination."""

import time
import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from .execution_plan import ExecutionPlan, ExecutionStep, StepResult, StepStatus, StepType
from .context import AgentContext
from .llm_intent_analyzer import IntentAnalyzer, QueryIntent, QueryComplexity
from .discovery_agent import DiscoveryAgent
from .analysis_agent import AnalysisAgent
from .knowledge_graph_agent import KnowledgeGraphAgent
from ..registry import PluginRegistry

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """Master agent for high-level planning and coordination.
    
    The orchestrator analyzes user intent, creates execution plans,
    coordinates specialist agents, and synthesizes final responses.
    """
    
    def __init__(self, plugin_registry: PluginRegistry):
        """Initialize the orchestrator agent.
        
        Args:
            plugin_registry: Registry of available plugins
        """
        super().__init__("orchestrator")
        self.registry = plugin_registry
        self.intent_analyzer = IntentAnalyzer()
        
        # Initialize specialist agents
        self.discovery_agent = DiscoveryAgent(plugin_registry)
        self.analysis_agent = AnalysisAgent(plugin_registry)
        self.knowledge_graph_agent = KnowledgeGraphAgent(plugin_registry)
        
        # Map agents by name for routing
        self.agents = {
            "discovery": self.discovery_agent,
            "analysis": self.analysis_agent,
            "knowledge_graph": self.knowledge_graph_agent
        }
        
        self.orchestrator_logger = logging.getLogger('agent.orchestrator')
    
    def process_query(self, query: str) -> str:
        """Process a natural language query using agentic architecture.
        
        Args:
            query: Natural language question to process
            
        Returns:
            Formatted response string
        """
        start_time = time.time()
        self.orchestrator_logger.info(f"Processing query: {query}")
        
        try:
            # Step 1: Analyze intent
            intent_result = self.intent_analyzer.analyze_intent(query)
            self.orchestrator_logger.info(f"Intent analysis: {intent_result.primary_intent.value} "
                                        f"(confidence: {intent_result.confidence:.2f})")
            
            # Step 2: Create execution plan
            plan = self._create_execution_plan(query, intent_result)
            
            # Step 3: Create agent context
            context = AgentContext(
                session_id=plan.id,
                query=query,
                intent=intent_result.primary_intent.value
            )
            
            # Step 4: Execute the plan
            execution_result = self._execute_plan(plan, context)
            
            # Step 5: Synthesize response
            response = self._synthesize_response(plan, context, execution_result)
            
            execution_time = time.time() - start_time
            self.orchestrator_logger.info(f"Query processing completed in {execution_time:.2f}s")
            
            return response
            
        except Exception as e:
            self.orchestrator_logger.error(f"Query processing failed: {e}")
            return f"❌ Error processing query: {e}"
    
    def can_handle(self, step: ExecutionStep) -> bool:
        """Orchestrator can handle coordination steps."""
        return step.step_type == StepType.SYNTHESIZE_FINDINGS
    
    def execute_step(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Execute a coordination step."""
        if step.step_type == StepType.SYNTHESIZE_FINDINGS:
            return self._synthesize_findings_step(step, context)
        else:
            return self._create_failure_result(
                step, f"Orchestrator cannot handle step type: {step.step_type.value}"
            )
    
    def _create_execution_plan(self, query: str, intent_result) -> ExecutionPlan:
        """Create an execution plan based on intent analysis.
        
        Args:
            query: User query
            intent_result: IntentAnalysisResult from intent analysis
            
        Returns:
            ExecutionPlan with appropriate steps
        """
        plan = ExecutionPlan(query=query, intent=intent_result.primary_intent.value)
        
        self.orchestrator_logger.info(f"Creating execution plan for intent: {intent_result.primary_intent.value}")
        
        # Create plan based on primary intent
        if intent_result.primary_intent == QueryIntent.DOCUMENT_DISCOVERY:
            self._create_document_discovery_plan(plan, query, intent_result)
        
        elif intent_result.primary_intent == QueryIntent.CONTENT_ANALYSIS:
            self._create_content_analysis_plan(plan, query, intent_result)
        
        elif intent_result.primary_intent == QueryIntent.COMPARISON:
            self._create_comparison_plan(plan, query, intent_result)
        
        elif intent_result.primary_intent == QueryIntent.INVESTIGATION:
            self._create_investigation_plan(plan, query, intent_result)
        
        elif intent_result.primary_intent == QueryIntent.METADATA_QUERY:
            self._create_metadata_query_plan(plan, query, intent_result)
        
        elif intent_result.primary_intent == QueryIntent.RELATIONSHIP_ANALYSIS:
            self._create_relationship_analysis_plan(plan, query, intent_result)
        
        else:
            # Default plan for unknown intents
            self._create_default_plan(plan, query, intent_result)
        
        self.orchestrator_logger.info(f"Created plan with {len(plan.steps)} steps")
        return plan
    
    def _create_document_discovery_plan(self, plan: ExecutionPlan, query: str, intent_result) -> None:
        """Create plan for document discovery queries."""
        # Step 1: Discover documents
        discover_step = plan.add_step(
            StepType.DISCOVER_DOCUMENT,
            "discovery",
            {"query": query, "intent": "document_discovery"}
        )
        
        # Step 2: Return file paths or metadata
        if any(word in query.lower() for word in ["path", "location", "where"]):
            plan.add_step(
                StepType.RETURN_FILE_PATH,
                "discovery",
                {"query": query},
                dependencies=[discover_step]
            )
        else:
            plan.add_step(
                StepType.RETURN_METADATA,
                "discovery", 
                {"query": query},
                dependencies=[discover_step]
            )
    
    def _create_content_analysis_plan(self, plan: ExecutionPlan, query: str, intent_result) -> None:
        """Create plan for content analysis queries."""
        # Step 1: Discover relevant documents
        discover_step = plan.add_step(
            StepType.DISCOVER_DOCUMENT,
            "discovery",
            {"query": query, "intent": "content_analysis"}
        )
        
        # Step 2: Extract content
        extract_step = plan.add_step(
            StepType.EXTRACT_CONTENT,
            "analysis",
            {"query": query, "extraction_type": "content_analysis"},
            dependencies=[discover_step]
        )
        
        # Step 3: Analyze for specific patterns if needed
        if any(word in query.lower() for word in ["decision", "choice", "determine"]):
            plan.add_step(
                StepType.ANALYZE_DECISIONS,
                "analysis",
                {"query": query, "focus": "decisions"},
                dependencies=[extract_step]
            )
    
    def _create_comparison_plan(self, plan: ExecutionPlan, query: str, intent_result) -> None:
        """Create plan for comparison queries."""
        # Step 1: Discover documents to compare
        discover_step = plan.add_step(
            StepType.DISCOVER_DOCUMENT,
            "discovery",
            {"query": query, "intent": "comparison", "multiple_docs": True}
        )
        
        # Step 2: Compare across documents
        compare_step = plan.add_step(
            StepType.COMPARE_ACROSS_DOCS,
            "analysis",
            {"query": query, "comparison_type": "content"},
            dependencies=[discover_step]
        )
        
        # Step 3: Synthesize findings
        plan.add_step(
            StepType.SYNTHESIZE_FINDINGS,
            "analysis",
            {"query": query, "synthesis_type": "comparison"},
            dependencies=[compare_step]
        )
    
    def _create_investigation_plan(self, plan: ExecutionPlan, query: str, intent_result) -> None:
        """Create plan for investigative queries."""
        # Step 1: Discover relevant documents
        discover_step = plan.add_step(
            StepType.DISCOVER_DOCUMENT,
            "discovery",
            {"query": query, "intent": "investigation"}
        )
        
        # Step 2: Extract content from multiple sources
        extract_step = plan.add_step(
            StepType.EXTRACT_CONTENT,
            "analysis",
            {"query": query, "extraction_type": "investigation"},
            dependencies=[discover_step]
        )
        
        # Step 3: Find relationships if entity-focused
        if any(word in query.lower() for word in ["who", "relationship", "connected"]):
            rel_step = plan.add_step(
                StepType.FIND_RELATIONSHIPS,
                "knowledge_graph",
                {"query": query, "relationship_type": "investigation"},
                dependencies=[extract_step]
            )
            
            # Step 4: Synthesize all findings
            plan.add_step(
                StepType.SYNTHESIZE_FINDINGS,
                "analysis",
                {"query": query, "synthesis_type": "investigation"},
                dependencies=[rel_step]
            )
        else:
            # Step 4: Synthesize findings without relationships
            plan.add_step(
                StepType.SYNTHESIZE_FINDINGS,
                "analysis",
                {"query": query, "synthesis_type": "investigation"},
                dependencies=[extract_step]
            )
    
    def _create_metadata_query_plan(self, plan: ExecutionPlan, query: str, intent_result) -> None:
        """Create plan for metadata queries with constraint-aware processing."""
        # Import constraint extraction locally to avoid circular imports
        from shared.constraints import ConstraintExtractor, get_content_filtering_multiplier
        
        # Extract constraints from the query
        constraints = ConstraintExtractor.extract(query)
        
        self.orchestrator_logger.info(f"Extracted constraints: count={constraints.count}, "
                                    f"file_types={constraints.file_types}, "
                                    f"has_content_filter={constraints.has_content_filter}")
        
        if constraints.has_content_filter:
            # Two-step plan: metadata filtering + content filtering
            self._create_two_step_content_plan(plan, query, constraints)
        else:
            # Single-step plan: metadata-only query
            self._create_single_step_metadata_plan(plan, query, constraints)
    
    def _create_single_step_metadata_plan(self, plan: ExecutionPlan, query: str, constraints) -> None:
        """Create single-step plan for metadata-only queries."""
        # Prepare metadata parameters with extracted constraints
        metadata_params = {
            "query": query,
            "intent": "metadata_query",
            "operation": "get_latest_files"  # Use latest files operation for constraint support
        }
        
        # Add constraint parameters if present
        if constraints.count is not None:
            metadata_params["count"] = constraints.count
        
        if constraints.file_types:
            # Use first file type for the metadata plugin (it supports single file_type)
            # For multiple types, the metadata plugin's _get_latest_files can handle it
            if len(constraints.file_types) == 1:
                metadata_params["file_type"] = constraints.file_types[0]
            else:
                # Pass as list - the metadata plugin will need to handle this
                metadata_params["file_types"] = constraints.file_types
        
        plan.add_step(
            StepType.RETURN_METADATA,
            "discovery",
            metadata_params
        )
    
    def _create_two_step_content_plan(self, plan: ExecutionPlan, query: str, constraints) -> None:
        """Create two-step plan for queries with content filtering."""
        from shared.constraints import get_content_filtering_multiplier, get_default_count
        
        # Step 1: Metadata filtering with widened count
        base_count = constraints.count if constraints.count is not None else get_default_count()
        widened_count = base_count * get_content_filtering_multiplier()
        
        metadata_params = {
            "query": query,
            "intent": "metadata_query_for_content",
            "operation": "get_latest_files",
            "count": widened_count
        }
        
        # Add file type constraints
        if constraints.file_types:
            if len(constraints.file_types) == 1:
                metadata_params["file_type"] = constraints.file_types[0]
            else:
                metadata_params["file_types"] = constraints.file_types
        
        # Add metadata step
        metadata_step_id = plan.add_step(
            StepType.RETURN_METADATA,
            "discovery",
            metadata_params
        )
        
        # Step 2: Content filtering on metadata results
        content_params = {
            "query": query,
            "intent": "content_filtering",
            "extraction_type": "content_filtering",
            "content_terms": constraints.content_terms,
            "target_count": base_count,  # Final desired count
            "use_target_docs": True  # Signal to restrict search to metadata results
        }
        
        plan.add_step(
            StepType.EXTRACT_CONTENT,
            "analysis",
            content_params,
            dependencies=[metadata_step_id]
        )
    
    def _create_relationship_analysis_plan(self, plan: ExecutionPlan, query: str, intent_result) -> None:
        """Create plan for relationship analysis queries."""
        # Step 1: Find relationships
        rel_step = plan.add_step(
            StepType.FIND_RELATIONSHIPS,
            "knowledge_graph",
            {"query": query, "relationship_type": "analysis"}
        )
        
        # Step 2: Discover related documents if needed
        if any(word in query.lower() for word in ["document", "file", "source"]):
            discover_step = plan.add_step(
                StepType.DISCOVER_DOCUMENT,
                "discovery",
                {"query": query, "intent": "relationship_context"},
                dependencies=[rel_step]
            )
            
            # Step 3: Synthesize with document context
            plan.add_step(
                StepType.SYNTHESIZE_FINDINGS,
                "analysis",
                {"query": query, "synthesis_type": "relationship_with_docs"},
                dependencies=[discover_step]
            )
    
    def _create_default_plan(self, plan: ExecutionPlan, query: str, intent_result) -> None:
        """Create default plan for unclear intents."""
        # Try discovery first
        discover_step = plan.add_step(
            StepType.DISCOVER_DOCUMENT,
            "discovery",
            {"query": query, "intent": "general"}
        )
        
        # Then content extraction
        plan.add_step(
            StepType.EXTRACT_CONTENT,
            "analysis",
            {"query": query, "extraction_type": "general"},
            dependencies=[discover_step]
        )
    
    def _execute_plan(self, plan: ExecutionPlan, context: AgentContext) -> Dict[str, Any]:
        """Execute the execution plan step by step.
        
        Args:
            plan: ExecutionPlan to execute
            context: AgentContext for communication
            
        Returns:
            Dictionary with execution results
        """
        plan.status = "executing"
        self.orchestrator_logger.info(f"Executing plan with {len(plan.steps)} steps")
        
        max_iterations = 20  # Prevent infinite loops
        iteration = 0
        
        while not plan.is_complete() and iteration < max_iterations:
            iteration += 1
            
            # Get next executable steps
            next_steps = plan.get_next_steps()
            
            if not next_steps:
                # No more steps can execute - check if we're stuck
                if plan.has_failed_steps():
                    self.orchestrator_logger.warning("Plan execution stopped due to failed steps")
                    break
                else:
                    # Might be waiting for dependencies
                    self.orchestrator_logger.warning("No executable steps found - plan may be complete")
                    break
            
            # Execute each ready step
            for step in next_steps:
                result = self._execute_single_step(step, context)
                plan.update_step_status(step.id, result.status, result)
                
                # Log step completion
                status_msg = "SUCCESS" if result.is_successful() else f"FAILED: {result.error}"
                self.orchestrator_logger.info(f"Step {step.step_type.value} - {status_msg}")
        
        plan.status = "completed" if plan.is_complete() else "incomplete"
        
        execution_summary = plan.get_execution_summary()
        self.orchestrator_logger.info(f"Plan execution completed: {execution_summary['completed_steps']}/{execution_summary['total_steps']} steps")
        
        return {
            "plan_summary": execution_summary,
            "context_summary": context.get_summary(),
            "success": not plan.has_failed_steps()
        }
    
    def _execute_single_step(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Execute a single step using the appropriate agent.
        
        Args:
            step: ExecutionStep to execute
            context: AgentContext for communication
            
        Returns:
            StepResult from step execution
        """
        step.status = StepStatus.IN_PROGRESS
        
        # Find the appropriate agent
        agent = self.agents.get(step.agent_name)
        if not agent:
            return self._create_failure_result(
                step, f"No agent found for: {step.agent_name}"
            )
        
        # Check if agent can handle this step
        if not agent.can_handle(step):
            return self._create_failure_result(
                step, f"Agent {step.agent_name} cannot handle step type {step.step_type.value}"
            )
        
        # Execute the step using the agent's safe execution method
        return agent._safe_execute_step(step, context)
    
    def _synthesize_response(self, plan: ExecutionPlan, context: AgentContext, 
                           execution_result: Dict[str, Any]) -> str:
        """Synthesize final response from plan execution results.
        
        Args:
            plan: Executed ExecutionPlan
            context: AgentContext with accumulated results
            execution_result: Results from plan execution
            
        Returns:
            Final formatted response string
        """
        self.orchestrator_logger.info("Synthesizing final response")
        
        # Check if execution was successful
        if not execution_result["success"]:
            return "❌ Unable to process query due to execution failures."
        
        # Get all successful step results
        successful_results = plan.get_results()
        
        if not successful_results:
            return "No relevant information found."
        
        # Synthesize based on intent and available results
        return self._create_intent_based_response(plan, context, successful_results)
    
    def _create_intent_based_response(self, plan: ExecutionPlan, context: AgentContext,
                                    results: List[StepResult]) -> str:
        """Create response based on original intent and results.
        
        Args:
            plan: Executed ExecutionPlan
            context: AgentContext with results
            results: List of successful StepResults
            
        Returns:
            Formatted response string
        """
        intent = plan.intent
        
        # Handle different intent types
        if intent == "document_discovery":
            return self._format_discovery_response(context, results)
        
        elif intent == "content_analysis":
            return self._format_analysis_response(context, results)
        
        elif intent == "comparison":
            return self._format_comparison_response(context, results)
        
        elif intent == "investigation":
            return self._format_investigation_response(context, results)
        
        elif intent == "metadata_query":
            return self._format_metadata_response(context, results)
        
        elif intent == "relationship_analysis":
            return self._format_relationship_response(context, results)
        
        else:
            # Default response formatting
            return self._format_default_response(context, results)
    
    def _format_discovery_response(self, context: AgentContext, results: List[StepResult]) -> str:
        """Format response for document discovery."""
        if context.has_documents():
            # Prioritize file path results if available
            for result in results:
                if result.step_type == StepType.RETURN_FILE_PATH:
                    return result.get_value("formatted_response", "Documents found")
                elif result.step_type == StepType.RETURN_METADATA:
                    return result.get_value("formatted_response", "Documents found")
        
        return "No documents found matching your query."
    
    def _format_analysis_response(self, context: AgentContext, results: List[StepResult]) -> str:
        """Format response for content analysis with constraint validation."""
        response_parts = []
        
        # Check if this was a constraint-based content filtering query
        try:
            from shared.constraints import ConstraintExtractor, validate_constraint_results
            constraints = ConstraintExtractor.extract(context.query)
            
            content_results = []
            for result in results:
                if result.step_type == StepType.EXTRACT_CONTENT:
                    content = result.get_value("extracted_content", {})
                    if content.get("content"):
                        response_parts.append(content["content"])
                        
                    # Extract sources for constraint validation
                    sources = content.get("sources", [])
                    content_results.extend(sources)
                        
                elif result.step_type == StepType.ANALYZE_DECISIONS:
                    decisions = result.get_value("analysis_summary", "")
                    if decisions:
                        response_parts.append(f"\n📋 Decision Analysis:\n{decisions}")
            
            # Validate constraints for content filtering queries
            if constraints.has_content_filter and content_results:
                validation = validate_constraint_results(constraints, content_results)
                
                if not validation["meets_constraints"]:
                    # Add constraint validation feedback
                    feedback_parts = []
                    
                    if validation["issues"]:
                        feedback_parts.append("\n⚠️ Query constraints:")
                        for issue in validation["issues"]:
                            feedback_parts.append(f"  • {issue}")
                    
                    if validation["suggestions"]:
                        feedback_parts.append("\n💡 Suggestions:")
                        for suggestion in validation["suggestions"]:
                            feedback_parts.append(f"  • {suggestion}")
                    
                    if feedback_parts:
                        response_parts.extend(feedback_parts)
            
        except Exception as e:
            self.orchestrator_logger.warning(f"Content constraint validation failed: {e}")
        
        return "\n\n".join(response_parts) if response_parts else "No content analysis available."
    
    def _format_comparison_response(self, context: AgentContext, results: List[StepResult]) -> str:
        """Format response for comparison analysis."""
        for result in results:
            if result.step_type == StepType.COMPARE_ACROSS_DOCS:
                return result.get_value("summary", "Comparison completed")
            elif result.step_type == StepType.SYNTHESIZE_FINDINGS:
                synthesis = result.get_value("synthesis", {})
                return synthesis.get("final_response", "Comparison analysis completed")
        
        return "Comparison analysis completed."
    
    def _format_investigation_response(self, context: AgentContext, results: List[StepResult]) -> str:
        """Format response for investigation."""
        response_parts = []
        
        for result in results:
            if result.step_type == StepType.SYNTHESIZE_FINDINGS:
                synthesis = result.get_value("synthesis", {})
                final_response = synthesis.get("final_response", "")
                if final_response:
                    response_parts.append(final_response)
            elif result.step_type == StepType.FIND_RELATIONSHIPS:
                entities = result.get_value("entities", [])
                if entities:
                    response_parts.append(f"\n🔗 Key Entities: {', '.join(entities[:5])}")
        
        return "\n".join(response_parts) if response_parts else "Investigation completed."
    
    def _format_metadata_response(self, context: AgentContext, results: List[StepResult]) -> str:
        """Format response for metadata queries with constraint validation."""
        # Check if this was a constraint-based query
        try:
            from shared.constraints import ConstraintExtractor, validate_constraint_results
            constraints = ConstraintExtractor.extract(context.query)
            
            # Find metadata results
            metadata_result = None
            for result in results:
                if result.step_type == StepType.RETURN_METADATA:
                    metadata_result = result
                    break
            
            if metadata_result:
                # Extract results for validation
                result_data = metadata_result.get_value("data", {})
                files = result_data.get("files", []) if isinstance(result_data, dict) else []
                
                # Validate against constraints
                validation = validate_constraint_results(constraints, files)
                
                # Get formatted response
                base_response = metadata_result.get_value("formatted_response", "Metadata retrieved")
                
                # Add constraint validation feedback
                if not validation["meets_constraints"]:
                    feedback_parts = [base_response]
                    
                    # Add issues
                    if validation["issues"]:
                        feedback_parts.append("\n⚠️ Constraint validation:")
                        for issue in validation["issues"]:
                            feedback_parts.append(f"  • {issue}")
                    
                    # Add suggestions
                    if validation["suggestions"]:
                        feedback_parts.append("\n💡 Suggestions:")
                        for suggestion in validation["suggestions"]:
                            feedback_parts.append(f"  • {suggestion}")
                    
                    return "\n".join(feedback_parts)
                
                return base_response
            
        except Exception as e:
            self.orchestrator_logger.warning(f"Constraint validation failed: {e}")
        
        # Fallback to simple formatting
        for result in results:
            if result.step_type == StepType.RETURN_METADATA:
                return result.get_value("formatted_response", "Metadata retrieved")
        
        return "Metadata information retrieved."
    
    def _format_relationship_response(self, context: AgentContext, results: List[StepResult]) -> str:
        """Format response for relationship analysis."""
        for result in results:
            if result.step_type == StepType.FIND_RELATIONSHIPS:
                entities = result.get_value("entities", [])
                raw_response = result.get_value("raw_response", "")
                
                if raw_response and not raw_response.startswith("❌"):
                    return raw_response
                elif entities:
                    return f"🔗 Found entities and relationships: {', '.join(entities[:5])}"
        
        return "Relationship analysis completed."
    
    def _format_default_response(self, context: AgentContext, results: List[StepResult]) -> str:
        """Format default response."""
        if results:
            # Try to get the most relevant result
            for result in results:
                if result.step_type == StepType.EXTRACT_CONTENT:
                    content = result.get_value("extracted_content", {})
                    if content.get("content"):
                        return content["content"]
                elif "formatted_response" in result.data:
                    return result.get_value("formatted_response", "")
        
        return "Query processed successfully."
    
    def _synthesize_findings_step(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Execute synthesis step as orchestrator."""
        # This delegates to the analysis agent for actual synthesis
        return self.analysis_agent.execute_step(step, context)