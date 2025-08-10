"""Tests for agentic architecture components."""

import pytest
import time
from unittest.mock import Mock, MagicMock

from backend.src.querying.agents.agentic.execution_plan import (
    ExecutionPlan, ExecutionStep, StepResult, StepType, StepStatus
)
from backend.src.querying.agents.agentic.context import AgentContext
from backend.src.querying.agents.agentic.intent_analyzer import (
    IntentAnalyzer, QueryIntent, QueryComplexity
)
from backend.src.querying.agents.agentic.orchestrator_agent import OrchestratorAgent


class TestExecutionPlan:
    """Test execution plan functionality."""
    
    def test_plan_creation(self):
        """Test basic plan creation."""
        plan = ExecutionPlan(query="test query", intent="document_discovery")
        
        assert plan.query == "test query"
        assert plan.intent == "document_discovery"
        assert len(plan.steps) == 0
        assert plan.status == "created"
    
    def test_add_step(self):
        """Test adding steps to plan."""
        plan = ExecutionPlan()
        
        step_id = plan.add_step(
            StepType.DISCOVER_DOCUMENT,
            "discovery",
            {"query": "test"},
            []
        )
        
        assert len(plan.steps) == 1
        assert plan.steps[0].id == step_id
        assert plan.steps[0].step_type == StepType.DISCOVER_DOCUMENT
        assert plan.steps[0].agent_name == "discovery"
    
    def test_step_dependencies(self):
        """Test step dependency management."""
        plan = ExecutionPlan()
        
        step1_id = plan.add_step(StepType.DISCOVER_DOCUMENT, "discovery", {})
        step2_id = plan.add_step(StepType.EXTRACT_CONTENT, "analysis", {}, [step1_id])
        
        # Initially, only step1 should be executable
        next_steps = plan.get_next_steps()
        assert len(next_steps) == 1
        assert next_steps[0].id == step1_id
        
        # Complete step1
        plan.update_step_status(step1_id, StepStatus.COMPLETED)
        
        # Now step2 should be executable
        next_steps = plan.get_next_steps()
        assert len(next_steps) == 1
        assert next_steps[0].id == step2_id
    
    def test_plan_completion(self):
        """Test plan completion tracking."""
        plan = ExecutionPlan()
        
        step1_id = plan.add_step(StepType.DISCOVER_DOCUMENT, "discovery", {})
        step2_id = plan.add_step(StepType.EXTRACT_CONTENT, "analysis", {})
        
        assert not plan.is_complete()
        
        # Complete both steps
        plan.update_step_status(step1_id, StepStatus.COMPLETED)
        plan.update_step_status(step2_id, StepStatus.COMPLETED)
        
        assert plan.is_complete()


class TestAgentContext:
    """Test agent context functionality."""
    
    def test_context_creation(self):
        """Test basic context creation."""
        context = AgentContext(session_id="test-session", query="test query")
        
        assert context.session_id == "test-session"
        assert context.query == "test query"
        assert len(context.discovered_documents) == 0
        assert not context.has_documents()
    
    def test_document_management(self):
        """Test document discovery management."""
        context = AgentContext(session_id="test")
        
        doc1 = {"path": "/test/doc1.pdf", "name": "doc1.pdf"}
        doc2 = {"path": "/test/doc2.pdf", "name": "doc2.pdf"}
        
        context.add_discovered_document(doc1)
        context.add_discovered_document(doc2)
        
        assert context.has_documents()
        assert len(context.discovered_documents) == 2
        
        paths = context.get_discovered_paths()
        assert "/test/doc1.pdf" in paths
        assert "/test/doc2.pdf" in paths
    
    def test_content_management(self):
        """Test content extraction management."""
        context = AgentContext(session_id="test")
        
        content = {"content": "test content", "source": "doc1.pdf"}
        context.add_extracted_content("step1", content)
        
        assert context.has_content()
        retrieved = context.get_extracted_content("step1")
        assert retrieved["content"] == "test content"
    
    def test_entity_management(self):
        """Test entity tracking."""
        context = AgentContext(session_id="test")
        
        context.add_entity("John Doe")
        context.add_entity("ACME Corp")
        context.add_entity("John Doe")  # Duplicate
        
        entities = context.get_entities()
        assert len(entities) == 2
        assert "John Doe" in entities
        assert "ACME Corp" in entities
    
    def test_shared_data(self):
        """Test shared data management."""
        context = AgentContext(session_id="test")
        
        context.set_shared_data("key1", "value1")
        context.set_shared_data("key2", {"nested": "data"})
        
        assert context.get_shared_data("key1") == "value1"
        assert context.get_shared_data("key2")["nested"] == "data"
        assert context.get_shared_data("nonexistent", "default") == "default"


class TestIntentAnalyzer:
    """Test intent analysis functionality."""
    
    def test_analyzer_creation(self):
        """Test analyzer initialization."""
        analyzer = IntentAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'patterns')
    
    def test_document_discovery_intent(self):
        """Test document discovery intent recognition."""
        analyzer = IntentAnalyzer()
        
        queries = [
            "find the quarterly report",
            "get the meeting document",
            "where is the budget file",
            "show me the quarterly meeting recap"
        ]
        
        for query in queries:
            result = analyzer.analyze_intent(query)
            assert result.primary_intent == QueryIntent.DOCUMENT_DISCOVERY
            assert result.confidence > 0.5
    
    def test_content_analysis_intent(self):
        """Test content analysis intent recognition."""
        analyzer = IntentAnalyzer()
        
        queries = [
            "what does the document say about revenue",
            "analyze the meeting decisions",
            "explain the budget requirements",
            "describe the project scope"
        ]
        
        for query in queries:
            result = analyzer.analyze_intent(query)
            assert result.primary_intent == QueryIntent.CONTENT_ANALYSIS
            assert result.confidence > 0.5
    
    def test_comparison_intent(self):
        """Test comparison intent recognition."""
        analyzer = IntentAnalyzer()
        
        queries = [
            "compare strategies across documents",
            "what do all files say about revenue",
            "difference between Q1 and Q2 reports",
            "analyze trends across multiple documents"
        ]
        
        for query in queries:
            result = analyzer.analyze_intent(query)
            assert result.primary_intent == QueryIntent.COMPARISON
            assert result.confidence > 0.5
    
    def test_metadata_query_intent(self):
        """Test metadata query intent recognition."""
        analyzer = IntentAnalyzer()
        
        queries = [
            "how many PDF files are there",
            "list all documents",
            "show recent files",
            "count the presentations"
        ]
        
        for query in queries:
            result = analyzer.analyze_intent(query)
            assert result.primary_intent == QueryIntent.METADATA_QUERY
            assert result.confidence > 0.5
    
    def test_relationship_analysis_intent(self):
        """Test relationship analysis intent recognition."""
        analyzer = IntentAnalyzer()
        
        queries = [
            "who works with John Doe",
            "what organization is mentioned",
            "find people related to this project",
            "show connections between entities"
        ]
        
        for query in queries:
            result = analyzer.analyze_intent(query)
            assert result.primary_intent == QueryIntent.RELATIONSHIP_ANALYSIS
            assert result.confidence > 0.5
    
    def test_complexity_assessment(self):
        """Test query complexity assessment."""
        analyzer = IntentAnalyzer()
        
        # Simple query
        simple_result = analyzer.analyze_intent("find the budget file")
        assert simple_result.complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]
        
        # Complex query
        complex_result = analyzer.analyze_intent(
            "analyze comprehensive relationships between stakeholders across all project documents and identify evolution patterns"
        )
        assert complex_result.complexity == QueryComplexity.COMPLEX
    
    def test_entity_extraction(self):
        """Test entity extraction from queries."""
        analyzer = IntentAnalyzer()
        
        query = "find documents about John Doe and ACME Corporation"
        result = analyzer.analyze_intent(query)
        
        # Should extract proper nouns as entities
        assert "John" in result.key_entities or "Doe" in result.key_entities
        assert "ACME" in result.key_entities or "Corporation" in result.key_entities


class TestOrchestratorAgent:
    """Test orchestrator agent functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from backend.src.querying.agents.registry import PluginRegistry
        
        self.registry = PluginRegistry()
        
        # Mock metadata plugin
        self.mock_metadata_plugin = Mock()
        self.mock_metadata_plugin.validate_params.return_value = True
        self.mock_metadata_plugin.execute.return_value = {
            "response": "Found 1 document:\nQuarterly_Report_Q3.docx\nPath: /test/Quarterly_Report_Q3.docx"
        }
        self.registry.register_plugin("metadata", self.mock_metadata_plugin)
        
        # Mock semantic search plugin
        self.mock_semantic_plugin = Mock()
        self.mock_semantic_plugin.validate_params.return_value = True
        self.mock_semantic_plugin.execute.return_value = {
            "response": "The quarterly report shows revenue growth of 15% compared to last quarter."
        }
        self.registry.register_plugin("semantic_search", self.mock_semantic_plugin)
        
        self.orchestrator = OrchestratorAgent(self.registry)
    
    def test_orchestrator_creation(self):
        """Test orchestrator initialization."""
        assert self.orchestrator is not None
        assert self.orchestrator.name == "orchestrator"
        assert hasattr(self.orchestrator, 'intent_analyzer')
        assert hasattr(self.orchestrator, 'agents')
    
    def test_document_discovery_query(self):
        """Test document discovery query processing."""
        query = "find the quarterly report"
        
        response = self.orchestrator.process_query(query)
        
        # Should not be an error response
        assert not response.startswith("❌")
        
        # Should have called metadata plugin for discovery
        assert self.mock_metadata_plugin.execute.called
    
    def test_content_analysis_query(self):
        """Test content analysis query processing."""
        query = "what does the quarterly report say about revenue"
        
        response = self.orchestrator.process_query(query)
        
        # Should not be an error response
        assert not response.startswith("❌")
        
        # Should have called both plugins (discovery then analysis)
        assert self.mock_metadata_plugin.execute.called
        assert self.mock_semantic_plugin.execute.called
    
    def test_plan_creation_for_discovery(self):
        """Test execution plan creation for document discovery."""
        query = "get the budget document"
        intent_result = self.orchestrator.intent_analyzer.analyze_intent(query)
        
        plan = self.orchestrator._create_execution_plan(query, intent_result)
        
        assert plan.query == query
        assert plan.intent == "document_discovery"
        assert len(plan.steps) >= 1
        
        # Should have discovery step
        discovery_steps = [s for s in plan.steps if s.step_type == StepType.DISCOVER_DOCUMENT]
        assert len(discovery_steps) >= 1
    
    def test_plan_creation_for_analysis(self):
        """Test execution plan creation for content analysis."""
        query = "analyze the meeting decisions"
        intent_result = self.orchestrator.intent_analyzer.analyze_intent(query)
        
        plan = self.orchestrator._create_execution_plan(query, intent_result)
        
        assert plan.intent == "content_analysis"
        assert len(plan.steps) >= 2
        
        # Should have both discovery and extraction steps
        step_types = [s.step_type for s in plan.steps]
        assert StepType.DISCOVER_DOCUMENT in step_types
        assert StepType.EXTRACT_CONTENT in step_types
    
    def test_step_execution_with_agents(self):
        """Test step execution routing to appropriate agents."""
        # Create a simple plan
        plan = ExecutionPlan(query="test", intent="test")
        step_id = plan.add_step(StepType.DISCOVER_DOCUMENT, "discovery", {"query": "test"})
        step = plan.get_step(step_id)
        
        context = AgentContext(session_id="test", query="test")
        
        # Execute the step
        result = self.orchestrator._execute_single_step(step, context)
        
        # Should have succeeded (mock plugin returns success)
        assert result.status == StepStatus.COMPLETED
        assert result.agent_used == "discovery"
    
    def test_error_handling(self):
        """Test error handling in orchestrator."""
        # Create orchestrator without plugins
        empty_registry = PluginRegistry()
        orchestrator = OrchestratorAgent(empty_registry)
        
        query = "find the quarterly report"
        response = orchestrator.process_query(query)
        
        # Should handle gracefully (may return no results found)
        assert isinstance(response, str)
        # Should not crash
    
    def test_context_preservation(self):
        """Test that context is preserved across steps."""
        query = "analyze the quarterly report"
        
        # Mock the execution to verify context usage
        original_execute = self.orchestrator._execute_single_step
        context_snapshots = []
        
        def mock_execute(step, context):
            context_snapshots.append(context.get_summary())
            return original_execute(step, context)
        
        self.orchestrator._execute_single_step = mock_execute
        
        response = self.orchestrator.process_query(query)
        
        # Should have captured context at different points
        assert len(context_snapshots) > 0
        
        # Context should have the original query
        for snapshot in context_snapshots:
            assert snapshot["query"] == query


class TestAgentIntegration:
    """Test integration between legacy Agent and agentic architecture."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from backend.src.querying.agents.agent import Agent
        from backend.src.querying.agents.registry import PluginRegistry
        
        self.registry = PluginRegistry()
        
        # Mock plugins
        self.mock_metadata_plugin = Mock()
        self.mock_metadata_plugin.validate_params.return_value = True
        self.mock_metadata_plugin.execute.return_value = {
            "response": "Found 1 document: test.pdf"
        }
        self.registry.register_plugin("metadata", self.mock_metadata_plugin)
        
        self.agent = Agent(self.registry)
    
    def test_agentic_mode_enabled(self):
        """Test agent with agentic mode enabled."""
        self.agent.set_agentic_mode(True)
        
        query = "find the quarterly report"
        response = self.agent.process_query(query)
        
        # Should use agentic orchestrator
        assert isinstance(response, str)
        assert not response.startswith("❌")
    
    def test_agentic_mode_disabled(self):
        """Test agent with agentic mode disabled (legacy behavior)."""
        self.agent.set_agentic_mode(False)
        
        query = "find the quarterly report"
        response = self.agent.process_query(query)
        
        # Should use legacy processing
        assert isinstance(response, str)
        # Should have called plugins directly
        assert self.mock_metadata_plugin.execute.called
    
    def test_mode_switching(self):
        """Test switching between agentic and legacy modes."""
        query = "list all files"
        
        # Test agentic mode
        self.agent.set_agentic_mode(True)
        agentic_response = self.agent.process_query(query)
        
        # Reset mock
        self.mock_metadata_plugin.reset_mock()
        
        # Test legacy mode
        self.agent.set_agentic_mode(False)
        legacy_response = self.agent.process_query(query)
        
        # Both should return valid responses
        assert isinstance(agentic_response, str)
        assert isinstance(legacy_response, str)
        
        # Legacy mode should have called plugin directly
        assert self.mock_metadata_plugin.execute.called
    
    def test_capabilities_enhancement(self):
        """Test that agentic mode adds new capabilities."""
        # Legacy capabilities
        self.agent.set_agentic_mode(False)
        legacy_caps = self.agent.get_capabilities()
        
        # Agentic capabilities
        self.agent.set_agentic_mode(True)
        agentic_caps = self.agent.get_capabilities()
        
        # Agentic mode should have additional capabilities
        agentic_specific = set(agentic_caps) - set(legacy_caps)
        assert len(agentic_specific) > 0
        
        # Should include agentic features
        assert any("reasoning" in cap for cap in agentic_specific)
    
    def test_reasoning_explanation(self):
        """Test reasoning explanation in different modes."""
        query = "test query"
        
        # Test agentic explanation
        self.agent.set_agentic_mode(True)
        self.agent.process_query(query)
        agentic_explanation = self.agent.explain_reasoning()
        
        # Test legacy explanation
        self.agent.set_agentic_mode(False)
        self.agent.process_query(query)
        legacy_explanation = self.agent.explain_reasoning()
        
        # Both should provide explanations
        assert agentic_explanation is not None
        assert legacy_explanation is not None
        
        # Should indicate different modes
        assert "Agentic" in agentic_explanation
        assert "Legacy" in legacy_explanation or "Plugins used" in legacy_explanation


def test_end_to_end_agentic_processing():
    """Test complete end-to-end agentic processing."""
    from backend.src.querying.agents.agent import Agent
    from backend.src.querying.agents.registry import PluginRegistry
    
    # Set up complete system
    registry = PluginRegistry()
    
    # Mock all required plugins
    mock_metadata = Mock()
    mock_metadata.validate_params.return_value = True
    mock_metadata.execute.return_value = {
        "response": "Found 1 document:\nQuarterly_Report_Q3.docx\nPath: /test/Quarterly_Report_Q3.docx"
    }
    registry.register_plugin("metadata", mock_metadata)
    
    mock_semantic = Mock()
    mock_semantic.validate_params.return_value = True
    mock_semantic.execute.return_value = {
        "response": "The quarterly report shows strong performance with 15% revenue growth."
    }
    registry.register_plugin("semantic_search", mock_semantic)
    
    # Create agent and test various query types
    agent = Agent(registry)
    agent.set_agentic_mode(True)
    
    test_queries = [
        "find the quarterly report",
        "what does the budget document say about expenses",
        "list all PDF files",
        "analyze meeting decisions across documents"
    ]
    
    for query in test_queries:
        response = agent.process_query(query)
        
        # Should process without errors
        assert isinstance(response, str)
        assert not response.startswith("❌")
        
        # Should have reasoning
        reasoning = agent.explain_reasoning()
        assert reasoning is not None
        assert "Agentic" in reasoning