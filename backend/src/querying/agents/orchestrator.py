"""OrchestratorAgent for coordinating multi-step agentic workflows in DocQuest.

This module provides the core orchestration logic that replaces the traditional
keyword-based routing with intelligent agentic workflow coordination.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .registry import PluginRegistry
from .plugin import Plugin

logger = logging.getLogger(__name__)

# Specialized loggers for orchestrator operations
orchestrator_logger = logging.getLogger('agent.orchestrator')
intent_logger = logging.getLogger('agent.intent')
execution_logger = logging.getLogger('agent.execution')
synthesis_logger = logging.getLogger('agent.synthesis')


class QueryIntent(Enum):
    """Types of user intents that the orchestrator can identify."""
    DOCUMENT_DISCOVERY = "document_discovery"
    CONTENT_ANALYSIS = "content_analysis" 
    METADATA_QUERY = "metadata_query"
    RELATIONSHIP_ANALYSIS = "relationship_analysis"
    COMPREHENSIVE_REPORTING = "comprehensive_reporting"
    KNOWLEDGE_GRAPH_QUERY = "knowledge_graph_query"
    MULTI_STEP_ANALYSIS = "multi_step_analysis"


@dataclass
class ExecutionStep:
    """Represents a single step in the execution plan."""
    plugin_name: str
    parameters: Dict[str, Any]
    description: str
    depends_on: Optional[List[str]] = None


@dataclass
class ExecutionPlan:
    """Represents the complete execution plan for a query."""
    intent: QueryIntent
    steps: List[ExecutionStep]
    description: str


class OrchestratorAgent:
    """Orchestrator agent that coordinates multi-step agentic workflows.
    
    The orchestrator replaces the traditional keyword-based routing with intelligent
    workflow coordination that can:
    1. Analyze user intent
    2. Create multi-step execution plans
    3. Coordinate specialist agents/plugins
    4. Synthesize results across multiple steps
    """
    
    def __init__(self, registry: PluginRegistry):
        """Initialize the orchestrator with a plugin registry.
        
        Args:
            registry: Plugin registry containing available specialist agents/plugins
        """
        self.registry = registry
        self._last_intent = None
        self._last_execution_plan = None
        self._last_execution_results = []
        self._reasoning_trace = []
    
    def process_query(self, question: str) -> str:
        """Process a natural language query using agentic workflow coordination.
        
        Args:
            question: Natural language question to process
            
        Returns:
            Synthesized response from the agentic workflow
        """
        start_time = time.time()
        self._reasoning_trace = []
        self._last_execution_results = []
        
        try:
            orchestrator_logger.info(f"Orchestrator processing query: \"{question}\"")
            
            # Step 1: Intent Analysis
            intent = self._analyze_intent(question)
            self._last_intent = intent
            intent_logger.info(f"Identified intent: {intent.value}")
            
            # Step 2: Create Execution Plan
            execution_plan = self._create_execution_plan(question, intent)
            self._last_execution_plan = execution_plan
            orchestrator_logger.info(f"Created execution plan with {len(execution_plan.steps)} steps")
            
            # Step 3: Execute Plan
            results = self._execute_plan(execution_plan)
            self._last_execution_results = results
            
            # Step 4: Synthesize Response
            response = self._synthesize_response(question, intent, results)
            
            execution_time = time.time() - start_time
            orchestrator_logger.info(f"Orchestrator completed in {execution_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Orchestrator error processing query '{question}': {e}")
            return f"❌ Error processing query: {e}"
    
    def _analyze_intent(self, question: str) -> QueryIntent:
        """Analyze user intent from natural language question.
        
        Args:
            question: Natural language question
            
        Returns:
            Identified query intent
        """
        question_lower = question.lower()
        
        # Document discovery patterns
        document_discovery_indicators = [
            "find the", "get the", "show me the", "where is the", "locate the",
            "find document", "find file", "get document", "get file",
            "document called", "file called", "document named", "file named"
        ]
        
        # Knowledge graph patterns
        knowledge_graph_indicators = [
            "who is", "who works", "what company", "what organization",
            "people", "person", "employee", "organization", "related to",
            "connected to", "relationships between"
        ]
        
        # Relationship analysis patterns
        relationship_indicators = [
            "similar", "related", "relationship", "connection", "linked",
            "cross-reference", "references", "clusters", "groups", "themes"
        ]
        
        # Reporting patterns
        reporting_indicators = [
            "report", "summary", "analysis", "overview", "statistics",
            "trends", "insights", "dashboard", "metrics"
        ]
        
        # Metadata query patterns
        metadata_indicators = [
            "how many", "count", "number of", "list", "show me",
            "what files", "file types", "recently", "latest"
        ]
        
        # Check for multi-step patterns
        multi_step_patterns = [
            "analyze .* and report", "find .* and show", "summarize .* relationships",
            "latest .* about", "emails .* plus", "documents .* and related"
        ]
        
        import re
        is_multi_step = any(re.search(pattern, question_lower) for pattern in multi_step_patterns)
        
        # Intent classification logic
        if is_multi_step:
            intent_logger.debug("Multi-step analysis pattern detected")
            return QueryIntent.MULTI_STEP_ANALYSIS
        elif any(pattern in question_lower for pattern in document_discovery_indicators):
            intent_logger.debug("Document discovery pattern detected")
            return QueryIntent.DOCUMENT_DISCOVERY
        elif any(pattern in question_lower for pattern in knowledge_graph_indicators):
            intent_logger.debug("Knowledge graph query pattern detected")
            return QueryIntent.KNOWLEDGE_GRAPH_QUERY
        elif any(pattern in question_lower for pattern in relationship_indicators):
            intent_logger.debug("Relationship analysis pattern detected")
            return QueryIntent.RELATIONSHIP_ANALYSIS
        elif any(pattern in question_lower for pattern in reporting_indicators):
            intent_logger.debug("Comprehensive reporting pattern detected")
            return QueryIntent.COMPREHENSIVE_REPORTING
        elif any(pattern in question_lower for pattern in metadata_indicators):
            intent_logger.debug("Metadata query pattern detected")
            return QueryIntent.METADATA_QUERY
        else:
            intent_logger.debug("Content analysis pattern detected (default)")
            return QueryIntent.CONTENT_ANALYSIS
    
    def _create_execution_plan(self, question: str, intent: QueryIntent) -> ExecutionPlan:
        """Create a multi-step execution plan based on the identified intent.
        
        Args:
            question: Natural language question
            intent: Identified query intent
            
        Returns:
            Execution plan with ordered steps
        """
        steps = []
        
        if intent == QueryIntent.DOCUMENT_DISCOVERY:
            # Document discovery: metadata first, then semantic search as backup
            steps.append(ExecutionStep(
                plugin_name="metadata",
                parameters=self._prepare_metadata_params(question),
                description="Search document metadata for matching files"
            ))
            steps.append(ExecutionStep(
                plugin_name="semantic_search",
                parameters=self._prepare_semantic_params(question, enhanced_discovery=True),
                description="Semantic search as backup for document discovery",
                depends_on=["metadata"]
            ))
            
        elif intent == QueryIntent.KNOWLEDGE_GRAPH_QUERY:
            # Knowledge graph with semantic search for hybrid results
            steps.append(ExecutionStep(
                plugin_name="knowledge_graph",
                parameters=self._prepare_kg_params(question),
                description="Query knowledge graph for entities and relationships"
            ))
            steps.append(ExecutionStep(
                plugin_name="semantic_search",
                parameters=self._prepare_semantic_params(question),
                description="Semantic search for content context"
            ))
            
        elif intent == QueryIntent.RELATIONSHIP_ANALYSIS:
            # Relationship analysis with supporting semantic search
            steps.append(ExecutionStep(
                plugin_name="document_relationships",
                parameters=self._prepare_relationship_params(question),
                description="Analyze document relationships and patterns"
            ))
            steps.append(ExecutionStep(
                plugin_name="semantic_search",
                parameters=self._prepare_semantic_params(question),
                description="Semantic search for relationship context"
            ))
            
        elif intent == QueryIntent.COMPREHENSIVE_REPORTING:
            # Comprehensive reporting
            steps.append(ExecutionStep(
                plugin_name="comprehensive_reporting",
                parameters=self._prepare_reporting_params(question),
                description="Generate comprehensive report"
            ))
            
        elif intent == QueryIntent.METADATA_QUERY:
            # Pure metadata query
            steps.append(ExecutionStep(
                plugin_name="metadata",
                parameters=self._prepare_metadata_params(question),
                description="Process metadata query"
            ))
            
        elif intent == QueryIntent.MULTI_STEP_ANALYSIS:
            # Complex multi-step analysis
            steps.append(ExecutionStep(
                plugin_name="metadata",
                parameters=self._prepare_metadata_params(question),
                description="Initial metadata analysis"
            ))
            steps.append(ExecutionStep(
                plugin_name="semantic_search",
                parameters=self._prepare_semantic_params(question),
                description="Semantic content analysis",
                depends_on=["metadata"]
            ))
            steps.append(ExecutionStep(
                plugin_name="document_relationships",
                parameters=self._prepare_relationship_params(question),
                description="Relationship analysis",
                depends_on=["semantic_search"]
            ))
            
        else:  # CONTENT_ANALYSIS (default)
            # Standard content analysis
            steps.append(ExecutionStep(
                plugin_name="semantic_search",
                parameters=self._prepare_semantic_params(question),
                description="Semantic content search and analysis"
            ))
        
        plan_description = f"Execute {len(steps)} steps for {intent.value}"
        return ExecutionPlan(intent=intent, steps=steps, description=plan_description)
    
    def _execute_plan(self, plan: ExecutionPlan) -> List[Tuple[str, Dict[str, Any]]]:
        """Execute the multi-step execution plan.
        
        Args:
            plan: Execution plan to execute
            
        Returns:
            List of (plugin_name, result) tuples
        """
        results = []
        executed_steps = set()
        
        execution_logger.info(f"Executing plan: {plan.description}")
        
        for step in plan.steps:
            # Check dependencies
            if step.depends_on:
                missing_deps = set(step.depends_on) - executed_steps
                if missing_deps:
                    logger.warning(f"Step {step.plugin_name} missing dependencies: {missing_deps}")
                    continue
            
            plugin = self.registry.get_plugin(step.plugin_name)
            if not plugin:
                logger.warning(f"Plugin {step.plugin_name} not available")
                continue
            
            try:
                execution_logger.info(f"Executing step: {step.description}")
                self._reasoning_trace.append(f"Executing: {step.description}")
                
                if plugin.validate_params(step.parameters):
                    result = plugin.execute(step.parameters)
                    results.append((step.plugin_name, result))
                    executed_steps.add(step.plugin_name)
                    execution_logger.info(f"Step {step.plugin_name} completed successfully")
                else:
                    logger.warning(f"Invalid parameters for step {step.plugin_name}")
                    
            except Exception as e:
                logger.error(f"Error executing step {step.plugin_name}: {e}")
                self._reasoning_trace.append(f"Step {step.plugin_name} failed: {e}")
        
        return results
    
    def _synthesize_response(self, question: str, intent: QueryIntent, results: List[Tuple[str, Dict[str, Any]]]) -> str:
        """Synthesize final response from multi-step execution results.
        
        Args:
            question: Original question
            intent: Query intent
            results: Results from execution steps
            
        Returns:
            Synthesized response
        """
        if not results:
            return "No relevant information found."
        
        synthesis_logger.info(f"Synthesizing response for {intent.value} with {len(results)} results")
        
        # Intent-specific synthesis strategies
        if intent == QueryIntent.DOCUMENT_DISCOVERY:
            return self._synthesize_document_discovery(results)
        elif intent == QueryIntent.KNOWLEDGE_GRAPH_QUERY:
            return self._synthesize_knowledge_graph(results)
        elif intent == QueryIntent.MULTI_STEP_ANALYSIS:
            return self._synthesize_multi_step(results)
        else:
            # Default synthesis for other intents
            return self._synthesize_default(results)
    
    def _synthesize_document_discovery(self, results: List[Tuple[str, Dict[str, Any]]]) -> str:
        """Synthesize response for document discovery queries."""
        metadata_results = []
        semantic_results = []
        
        for plugin_name, result in results:
            if plugin_name == "metadata":
                metadata_results.append(result)
            elif plugin_name == "semantic_search":
                semantic_results.append(result)
        
        # Prioritize metadata results for document discovery
        if metadata_results:
            for result in metadata_results:
                response = result.get("response", "")
                if response and not response.startswith("❌") and "No files found" not in response:
                    return response
        
        # Fall back to semantic results
        if semantic_results:
            for result in semantic_results:
                response = result.get("response", "")
                if response and not response.startswith("❌"):
                    return response
        
        return "No relevant documents found matching your query."
    
    def _synthesize_knowledge_graph(self, results: List[Tuple[str, Dict[str, Any]]]) -> str:
        """Synthesize response for knowledge graph queries."""
        response_parts = []
        
        for plugin_name, result in results:
            response = result.get("response", "")
            if response and not response.startswith("❌"):
                if plugin_name == "knowledge_graph":
                    response_parts.append(f"🔗 Knowledge Graph:\n{response}")
                elif plugin_name == "semantic_search":
                    response_parts.append(f"📄 Related Content:\n{response}")
        
        return "\n\n".join(response_parts) if response_parts else "No relevant information found."
    
    def _synthesize_multi_step(self, results: List[Tuple[str, Dict[str, Any]]]) -> str:
        """Synthesize response for multi-step analysis queries."""
        response_parts = []
        
        for plugin_name, result in results:
            response = result.get("response", "")
            if response and not response.startswith("❌"):
                if plugin_name == "metadata":
                    response_parts.append(f"📊 Document Overview:\n{response}")
                elif plugin_name == "semantic_search":
                    response_parts.append(f"📄 Content Analysis:\n{response}")
                elif plugin_name == "document_relationships":
                    response_parts.append(f"🔗 Relationship Analysis:\n{response}")
        
        return "\n\n".join(response_parts) if response_parts else "No relevant information found."
    
    def _synthesize_default(self, results: List[Tuple[str, Dict[str, Any]]]) -> str:
        """Default synthesis strategy for simple queries."""
        for plugin_name, result in results:
            response = result.get("response", "")
            if response and not response.startswith("❌"):
                return response
        
        return "No relevant information found."
    
    def _prepare_metadata_params(self, question: str) -> Dict[str, Any]:
        """Prepare parameters for metadata plugin."""
        # Simplified metadata parameter generation
        # In a real implementation, this could use LLM-based parameter generation
        return {
            "operation": "find_files",
            "question": question,
            "file_type": None,
            "count": None,
            "time_filter": None,
            "keywords": None
        }
    
    def _prepare_semantic_params(self, question: str, enhanced_discovery: bool = False) -> Dict[str, Any]:
        """Prepare parameters for semantic search plugin."""
        params = {
            "question": question,
            "use_document_level": True,
            "k": 50,
            "max_documents": 5,
            "context_window": 3
        }
        
        if enhanced_discovery:
            params.update({
                "include_metadata_search": True,
                "max_documents": 10,
                "k": 100
            })
        
        return params
    
    def _prepare_kg_params(self, question: str) -> Dict[str, Any]:
        """Prepare parameters for knowledge graph plugin."""
        return {
            "operation": "hybrid_search",
            "question": question,
            "max_depth": 2,
            "max_entities": 10
        }
    
    def _prepare_relationship_params(self, question: str) -> Dict[str, Any]:
        """Prepare parameters for relationship analysis plugin."""
        return {
            "operation": "analyze_relationships",
            "query": question
        }
    
    def _prepare_reporting_params(self, question: str) -> Dict[str, Any]:
        """Prepare parameters for reporting plugin."""
        return {
            "operation": "generate_collection_summary",
            "query": question
        }
    
    def get_last_reasoning_trace(self) -> List[str]:
        """Get the reasoning trace from the last query execution.
        
        Returns:
            List of reasoning steps
        """
        return self._reasoning_trace.copy()
    
    def get_last_execution_plan(self) -> Optional[ExecutionPlan]:
        """Get the execution plan from the last query.
        
        Returns:
            Last execution plan, or None if no query processed
        """
        return self._last_execution_plan