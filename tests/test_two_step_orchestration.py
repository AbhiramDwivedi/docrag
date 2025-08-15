"""Integration tests for two-step orchestration process."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import time

from backend.src.shared.constraints import ConstraintExtractor
from backend.src.querying.agents.agentic.orchestrator_agent import OrchestratorAgent
from backend.src.querying.agents.agentic.execution_plan import StepType, StepResult, StepStatus
from backend.src.querying.agents.agentic.llm_intent_analyzer import QueryIntent
from backend.src.querying.agents.agentic.context import AgentContext
from backend.src.querying.agents.registry import PluginRegistry


class TestTwoStepOrchestration:
    """Integration tests for the two-step orchestration process."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock plugin registry with realistic responses."""
        registry = Mock(spec=PluginRegistry)
        
        # Mock metadata plugin with realistic file data
        metadata_plugin = Mock()
        metadata_plugin.execute.return_value = {
            "response": "Found 9 PDF files (showing widened results for content filtering):\n• budget_q1.pdf\n• budget_q2.pdf\n• sales_report.pdf",
            "data": {
                "files": [
                    {
                        "name": "budget_q1.pdf",
                        "path": "/docs/budget_q1.pdf",
                        "size": 1200000,
                        "modified": time.time() - 86400,  # 1 day ago
                        "type": "PDF"
                    },
                    {
                        "name": "budget_q2.pdf", 
                        "path": "/docs/budget_q2.pdf",
                        "size": 950000,
                        "modified": time.time() - 172800,  # 2 days ago
                        "type": "PDF"
                    },
                    {
                        "name": "sales_report.pdf",
                        "path": "/docs/sales_report.pdf",
                        "size": 800000,
                        "modified": time.time() - 259200,  # 3 days ago
                        "type": "PDF"
                    },
                    {
                        "name": "budget_annual.pdf",
                        "path": "/docs/budget_annual.pdf", 
                        "size": 1500000,
                        "modified": time.time() - 345600,  # 4 days ago
                        "type": "PDF"
                    },
                    {
                        "name": "finance_overview.pdf",
                        "path": "/docs/finance_overview.pdf",
                        "size": 700000,
                        "modified": time.time() - 432000,  # 5 days ago
                        "type": "PDF"
                    }
                ]
            }
        }
        
        # Mock semantic search plugin with content-filtered results
        semantic_plugin = Mock()
        semantic_plugin.execute.return_value = {
            "response": "Based on budget-related content analysis...",
            "sources": [
                {
                    "file_path": "/docs/budget_q1.pdf",
                    "similarity": 0.95,
                    "content": "Q1 budget allocation shows 15% increase..."
                },
                {
                    "file_path": "/docs/budget_q2.pdf", 
                    "similarity": 0.92,
                    "content": "Q2 budget review indicates..."
                },
                {
                    "file_path": "/docs/budget_annual.pdf",
                    "similarity": 0.88,
                    "content": "Annual budget planning framework..."
                }
            ]
        }
        
        # Mock plugin validation
        metadata_plugin.validate_params.return_value = True
        semantic_plugin.validate_params.return_value = True
        
        registry.get_plugin.side_effect = lambda name: {
            "metadata": metadata_plugin,
            "semantic_search": semantic_plugin
        }.get(name)
        
        return registry
    
    @pytest.fixture
    def orchestrator(self, mock_registry):
        """Create orchestrator with mocked plugins."""
        return OrchestratorAgent(mock_registry)
    
    def test_two_step_orchestration_end_to_end(self, orchestrator, mock_registry):
        """Test complete two-step orchestration process end-to-end."""
        query = "list 3 latest pdfs that mention budget"
        
        # Mock intent analysis to return metadata query
        with patch.object(orchestrator.intent_analyzer, 'analyze_intent') as mock_intent:
            mock_intent.return_value = Mock(
                primary_intent=QueryIntent.METADATA_QUERY,
                confidence=0.9,
                complexity=Mock()
            )
            
            # Execute query - this should create and execute a two-step plan
            result = orchestrator.process_query(query)
            
            # Verify the result structure
            assert result is not None
            assert "final_response" in result or "response" in result
            
            # Verify that both plugins were called
            metadata_plugin = mock_registry.get_plugin("metadata")
            semantic_plugin = mock_registry.get_plugin("semantic_search")
            
            # Metadata plugin should be called for Step 1 (with widened count)
            assert metadata_plugin.execute.called
            metadata_call_args = metadata_plugin.execute.call_args[0][0]
            
            # Should request widened count (3 * 3 = 9) for metadata filtering
            assert metadata_call_args.get("count") == 9
            assert "PDF" in metadata_call_args.get("file_types", [])
            
            # Semantic plugin should be called for Step 2 (content filtering)
            assert semantic_plugin.execute.called
            semantic_call_args = semantic_plugin.execute.call_args[0][0]
            
            # Should limit to original requested count (3)
            assert semantic_call_args.get("max_documents") == 3
            assert "budget" in semantic_call_args.get("question", "")
            assert len(semantic_call_args.get("target_docs", [])) > 0
    
    def test_single_step_vs_two_step_plan_creation(self, orchestrator):
        """Test that plans are created correctly based on content terms."""
        
        # Test single-step plan (no content terms)
        query_metadata_only = "list 5 latest pdfs"
        
        with patch.object(orchestrator.intent_analyzer, 'analyze_intent') as mock_intent:
            mock_intent.return_value = Mock(
                primary_intent=QueryIntent.METADATA_QUERY,
                confidence=0.9
            )
            
            with patch.object(orchestrator, '_execute_plan') as mock_execute:
                with patch.object(orchestrator, '_synthesize_response') as mock_synthesize:
                    mock_execute.return_value = Mock()
                    mock_synthesize.return_value = "Mocked response"
                    
                    orchestrator.process_query(query_metadata_only)
                    
                    # Should create single-step plan
                    plan = mock_execute.call_args[0][0]
                    assert len(plan.steps) == 1
                    assert plan.steps[0].step_type == StepType.RETURN_METADATA
        
        # Test two-step plan (with content terms)
        query_with_content = "list 5 latest pdfs about budget planning"
        
        with patch.object(orchestrator.intent_analyzer, 'analyze_intent') as mock_intent:
            mock_intent.return_value = Mock(
                primary_intent=QueryIntent.METADATA_QUERY,
                confidence=0.9
            )
            
            with patch.object(orchestrator, '_execute_plan') as mock_execute:
                with patch.object(orchestrator, '_synthesize_response') as mock_synthesize:
                    mock_execute.return_value = Mock()
                    mock_synthesize.return_value = "Mocked response"
                    
                    orchestrator.process_query(query_with_content)
                    
                    # Should create two-step plan
                    plan = mock_execute.call_args[0][0]
                    assert len(plan.steps) == 2
                    assert plan.steps[0].step_type == StepType.RETURN_METADATA
                    assert plan.steps[1].step_type == StepType.ANALYZE_CONTENT
                    
                    # Verify step dependencies
                    assert plan.steps[0].id in plan.steps[1].dependencies
    
    def test_content_filtering_parameters(self, orchestrator):
        """Test that content filtering step receives correct parameters."""
        query = "list 10 latest spreadsheets that mention quarterly report"
        
        with patch.object(orchestrator.intent_analyzer, 'analyze_intent') as mock_intent:
            mock_intent.return_value = Mock(
                primary_intent=QueryIntent.METADATA_QUERY,
                confidence=0.9
            )
            
            with patch.object(orchestrator, '_execute_plan') as mock_execute:
                with patch.object(orchestrator, '_synthesize_response') as mock_synthesize:
                    mock_execute.return_value = Mock()
                    mock_synthesize.return_value = "Mocked response"
                    
                    orchestrator.process_query(query)
                    
                    plan = mock_execute.call_args[0][0]
                    
                    # Verify metadata step parameters (widened count)
                    metadata_step = plan.steps[0]
                    assert metadata_step.parameters["count"] == 30  # 10 * 3
                    assert set(metadata_step.parameters["file_types"]) == {"XLS", "XLSX"}
                    
                    # Verify content step parameters (original count)
                    content_step = plan.steps[1]
                    assert content_step.parameters["max_documents"] == 10
                    assert "quarterly report" in content_step.parameters["query"]
                    assert content_step.parameters["target_docs_from_step"] == metadata_step
    
    def test_constraint_extraction_comprehensive(self):
        """Test constraint extraction accuracy for various patterns."""
        test_cases = [
            # (query, exp_count, exp_file_types, has_content_terms)
            ("list 5 latest pdfs", 5, ["PDF"], False),
            ("show 10 recent docs about budget", 10, ["DOC", "DOCX"], True),
            ("find newest presentations on sales strategy", None, ["PPT", "PPTX"], True),
            ("get twenty spreadsheets mentioning quarterly earnings", 20, ["XLS", "XLSX"], True),
            ("list latest emails and docs", None, ["EMAIL", "DOC", "DOCX"], False),
        ]
        
        for query, exp_count, exp_types, has_content in test_cases:
            constraints = ConstraintExtractor.extract(query)
            
            assert constraints.count == exp_count, f"Count mismatch for: {query}"
            assert set(constraints.file_types) == set(exp_types), f"File types mismatch for: {query}"
            assert (len(constraints.content_terms) > 0) == has_content, f"Content terms mismatch for: {query}"
    
    def test_widening_factor_bounds(self, orchestrator):
        """Test that widening factor respects semantic_max_k bounds."""
        # Query that would exceed semantic_max_k after widening
        query = "list 30 latest pdfs about machine learning"  # 30 * 3 = 90, but semantic_max_k = 50
        
        with patch.object(orchestrator.intent_analyzer, 'analyze_intent') as mock_intent:
            mock_intent.return_value = Mock(
                primary_intent=QueryIntent.METADATA_QUERY,
                confidence=0.9
            )
            
            with patch.object(orchestrator, '_execute_plan') as mock_execute:
                with patch.object(orchestrator, '_synthesize_response') as mock_synthesize:
                    mock_execute.return_value = Mock()
                    mock_synthesize.return_value = "Mocked response"
                    
                    orchestrator.process_query(query)
                    
                    plan = mock_execute.call_args[0][0]
                    metadata_step = plan.steps[0]
                    
                    # Should be bounded by semantic_max_k (50)
                    assert metadata_step.parameters["count"] == 50
    
    def test_step_execution_ordering(self, orchestrator, mock_registry):
        """Test that steps execute in correct order with proper data flow."""
        query = "list 3 latest pdfs about budget"
        
        # Track execution order
        execution_order = []
        
        def track_metadata_execution(params):
            execution_order.append(("metadata", params))
            return {
                "response": "Found 9 files",
                "data": {
                    "files": [
                        {"name": "test.pdf", "path": "/test.pdf", "type": "PDF"}
                    ]
                }
            }
        
        def track_semantic_execution(params):
            execution_order.append(("semantic", params))
            # Verify that target_docs were passed from metadata step
            assert "target_docs" in params
            assert len(params["target_docs"]) > 0
            return {
                "response": "Filtered content",
                "sources": []
            }
        
        metadata_plugin = mock_registry.get_plugin("metadata")
        semantic_plugin = mock_registry.get_plugin("semantic_search")
        
        metadata_plugin.execute.side_effect = track_metadata_execution
        semantic_plugin.execute.side_effect = track_semantic_execution
        
        with patch.object(orchestrator.intent_analyzer, 'analyze_intent') as mock_intent:
            mock_intent.return_value = Mock(
                primary_intent=QueryIntent.METADATA_QUERY,
                confidence=0.9,
                complexity=Mock()
            )
            
            orchestrator.process_query(query)
            
            # Verify execution order
            assert len(execution_order) == 2
            assert execution_order[0][0] == "metadata"
            assert execution_order[1][0] == "semantic"
            
            # Verify data flow from metadata to semantic
            metadata_params = execution_order[0][1]
            semantic_params = execution_order[1][1]
            
            assert metadata_params.get("count") == 9  # Widened
            assert semantic_params.get("max_documents") == 3  # Original request
    
    def test_error_handling_in_two_step_process(self, orchestrator, mock_registry):
        """Test error handling when steps fail in two-step process."""
        query = "list 5 latest pdfs about budget"
        
        # Make metadata step fail
        metadata_plugin = mock_registry.get_plugin("metadata")
        metadata_plugin.execute.side_effect = Exception("Database error")
        
        with patch.object(orchestrator.intent_analyzer, 'analyze_intent') as mock_intent:
            mock_intent.return_value = Mock(
                primary_intent=QueryIntent.METADATA_QUERY,
                confidence=0.9,
                complexity=Mock()
            )
            
            result = orchestrator.process_query(query)
            
            # Should handle error gracefully
            assert result is not None
            # The exact error handling depends on implementation,
            # but it should not crash and should provide some feedback
    
    def test_context_data_flow(self, orchestrator, mock_registry):
        """Test that context properly stores and retrieves data between steps."""
        query = "list 2 latest pdfs about budget"
        
        def verify_context_usage(params):
            # This would be called by semantic plugin in step 2
            # It should receive target_docs from metadata step results
            return {
                "response": "Content filtered",
                "sources": [
                    {"file_path": "/test.pdf", "similarity": 0.9, "content": "Budget info"}
                ]
            }
        
        semantic_plugin = mock_registry.get_plugin("semantic_search")
        semantic_plugin.execute.side_effect = verify_context_usage
        
        with patch.object(orchestrator.intent_analyzer, 'analyze_intent') as mock_intent:
            mock_intent.return_value = Mock(
                primary_intent=QueryIntent.METADATA_QUERY,
                confidence=0.9,
                complexity=Mock()
            )
            
            result = orchestrator.process_query(query)
            
            # Verify that semantic plugin was called (meaning context worked)
            assert semantic_plugin.execute.called