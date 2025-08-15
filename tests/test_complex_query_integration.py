"""Integration tests for complex query analysis feature."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import time

from backend.src.shared.constraints import ConstraintExtractor, QueryConstraints
from backend.src.querying.agents.agentic.orchestrator_agent import OrchestratorAgent
from backend.src.querying.agents.agentic.execution_plan import StepType
from backend.src.querying.agents.agentic.llm_intent_analyzer import QueryIntent
from backend.src.querying.agents.registry import PluginRegistry
from backend.src.shared.config import get_settings


class TestComplexQueryAnalysis:
    """Integration tests for the complete complex query analysis feature."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock plugin registry."""
        registry = Mock(spec=PluginRegistry)
        
        # Mock metadata plugin
        metadata_plugin = Mock()
        metadata_plugin.execute.return_value = {
            "response": "Found 5 files:\n• doc1.pdf (1.2MB, 2024-01-15 14:30)\n• doc2.pdf (0.8MB, 2024-01-14 10:15)",
            "files": [
                {
                    "name": "doc1.pdf",
                    "path": "/path/to/doc1.pdf", 
                    "size": 1200000,
                    "modified": time.time() - 86400,  # 1 day ago
                    "type": "PDF"
                },
                {
                    "name": "doc2.pdf",
                    "path": "/path/to/doc2.pdf",
                    "size": 800000, 
                    "modified": time.time() - 172800,  # 2 days ago
                    "type": "PDF"
                }
            ]
        }
        
        # Mock semantic search plugin 
        semantic_plugin = Mock()
        semantic_plugin.execute.return_value = {
            "response": "Based on the documents, here are budget-related findings...",
            "sources": [
                {
                    "file_path": "/path/to/doc1.pdf",
                    "similarity": 0.95,
                    "content": "Budget allocation for Q1..."
                }
            ]
        }
        
        registry.get_plugin.side_effect = lambda name: {
            "metadata": metadata_plugin,
            "semantic_search": semantic_plugin
        }.get(name)
        
        return registry
    
    @pytest.fixture
    def orchestrator(self, mock_registry):
        """Create orchestrator with mocked plugins."""
        return OrchestratorAgent(mock_registry)
    
    def test_single_step_metadata_only_plan(self, orchestrator):
        """Test single-step plan for metadata-only queries."""
        query = "list 3 latest pdfs"
        
        # Mock intent analysis
        with patch.object(orchestrator.intent_analyzer, 'analyze_intent') as mock_intent:
            mock_intent.return_value = Mock(
                primary_intent=QueryIntent.METADATA_QUERY,
                confidence=0.9
            )
            
            # Mock plan execution to focus on plan creation
            with patch.object(orchestrator, '_execute_plan') as mock_execute:
                with patch.object(orchestrator, '_synthesize_response') as mock_synthesize:
                    mock_execute.return_value = Mock()
                    mock_synthesize.return_value = "Mocked response"
                    
                    # Execute query
                    result = orchestrator.process_query(query)
                    
                    # Verify plan was created with correct parameters
                    mock_execute.assert_called_once()
                    plan = mock_execute.call_args[0][0]  # First argument is the plan
                    
                    # Should have exactly 1 step (metadata only)
                    assert len(plan.steps) == 1
                    step = plan.steps[0]
                    assert step.step_type == StepType.RETURN_METADATA
                    assert step.parameters["count"] == 3
                    assert "PDF" in step.parameters["file_types"]
                    assert step.parameters["recency"] is True
    
    def test_two_step_plan_with_content_terms(self, orchestrator):
        """Test two-step plan for queries with content terms."""
        query = "list 5 latest pdfs that mention budget planning"
        
        # Mock intent analysis
        with patch.object(orchestrator.intent_analyzer, 'analyze_intent') as mock_intent:
            mock_intent.return_value = Mock(
                primary_intent=QueryIntent.METADATA_QUERY,
                confidence=0.9
            )
            
            # Mock plan execution
            with patch.object(orchestrator, '_execute_plan') as mock_execute:
                with patch.object(orchestrator, '_synthesize_response') as mock_synthesize:
                    mock_execute.return_value = Mock()
                    mock_synthesize.return_value = "Mocked response"
                    
                    # Execute query
                    result = orchestrator.process_query(query)
                    
                    # Verify plan was created with two steps
                    plan = mock_execute.call_args[0][0]
                    assert len(plan.steps) == 2
                    
                    # Step 1: Metadata with widened count
                    metadata_step = plan.steps[0]
                    assert metadata_step.step_type == StepType.RETURN_METADATA
                    assert metadata_step.parameters["count"] == 15  # 5 * 3 (widen factor)
                    assert "PDF" in metadata_step.parameters["file_types"]
                    
                    # Step 2: Content analysis
                    content_step = plan.steps[1]
                    assert content_step.step_type == StepType.ANALYZE_CONTENT
                    assert content_step.parameters["max_documents"] == 5
                    assert "budget planning" in content_step.parameters["query"]
                    assert metadata_step.id in content_step.dependencies
    
    def test_constraint_extraction_accuracy(self):
        """Test constraint extraction handles various query patterns."""
        test_cases = [
            # (query, expected_count, expected_file_types, expected_content_terms)
            ("list 10 latest spreadsheets", 10, ["XLS", "XLSX"], []),
            ("show me recent decks about sales", None, ["PPT", "PPTX"], ["sales"]),
            ("find five newest word documents", 5, ["DOC", "DOCX"], []),
            ("get latest emails that mention project updates", None, ["EMAIL"], ["project", "updates"]),
            ("list pdfs and docs from this week", None, ["DOC", "DOCX", "PDF"], ["week"]),
        ]
        
        for query, exp_count, exp_types, exp_content in test_cases:
            result = ConstraintExtractor.extract(query)
            
            assert result.count == exp_count, f"Count mismatch for: {query}"
            assert set(result.file_types) == set(exp_types), f"File types mismatch for: {query}"
            
            # Content terms should contain expected terms
            if exp_content:
                content_set = set(result.content_terms)
                expected_set = set(exp_content)
                assert expected_set.issubset(content_set), f"Content terms mismatch for: {query}"
            else:
                assert len(result.content_terms) == 0, f"Unexpected content terms for: {query}"
    
    def test_deterministic_ordering_specification(self):
        """Test that deterministic ordering requirements are met."""
        # This test verifies the specification rather than execution
        # since we don't have a real database in the test environment
        
        from backend.src.querying.agents.plugins.metadata_commands import MetadataCommandsPlugin
        
        # Verify the SQL query includes deterministic ordering
        plugin = MetadataCommandsPlugin()
        
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        with patch.object(plugin, '_get_db_connection', return_value=mock_conn):
            with patch.object(plugin, '_has_enhanced_schema', return_value=True):
                # Execute get_latest_files
                plugin._get_latest_files({"count": 5, "file_type": "PDF"}, mock_conn)
                
                # Verify SQL query was called
                assert mock_cursor.execute.called
                sql_query = mock_cursor.execute.call_args[0][0]
                
                # Check that deterministic ordering is present
                assert "ORDER BY f.modified_time DESC, f.file_name ASC" in sql_query
    
    def test_multiple_file_types_support(self):
        """Test that multiple file types are properly handled."""
        query = "list latest presentations and spreadsheets"
        constraints = ConstraintExtractor.extract(query)
        
        # Should extract both presentation and spreadsheet types
        expected_types = {"PPT", "PPTX", "XLS", "XLSX"}
        actual_types = set(constraints.file_types)
        
        assert expected_types == actual_types
    
    def test_safe_count_bounds(self):
        """Test that count extraction respects safe bounds."""
        test_cases = [
            ("list 0 files", 1),    # Minimum bound
            ("list 150 files", 100), # Maximum bound  
            ("list five files", 5),  # Number word - normal case
        ]
        
        for query, expected_count in test_cases:
            result = ConstraintExtractor.extract(query)
            assert result.count == expected_count
    
    def test_config_integration(self):
        """Test that configuration settings are properly integrated."""
        settings = get_settings()
        
        # Verify new configuration fields exist with expected defaults
        assert hasattr(settings, 'default_latest_count')
        assert hasattr(settings, 'content_filter_widen_factor')
        assert hasattr(settings, 'semantic_max_k')
        
        assert settings.default_latest_count == 10
        assert settings.content_filter_widen_factor == 3
        assert settings.semantic_max_k == 50
    
    def test_backward_compatibility(self):
        """Test that simple queries still work as before."""
        query = "latest files"
        constraints = ConstraintExtractor.extract(query)
        
        # Should detect recency but no specific constraints
        assert constraints.recency is True
        assert constraints.count is None  # Uses default
        assert len(constraints.file_types) == 0  # No specific types
        assert len(constraints.content_terms) == 0  # No content filtering
    
    def test_content_term_filtering(self):
        """Test content term extraction filters out stopwords and constraint terms."""
        query = "list the latest 5 pdfs that mention the quarterly budget report"
        constraints = ConstraintExtractor.extract(query)
        
        # Should extract meaningful content terms only
        content_terms = set(constraints.content_terms)
        
        # Should include meaningful terms
        assert "quarterly" in content_terms
        assert "budget" in content_terms
        assert "report" in content_terms
        
        # Should exclude stopwords and constraint terms
        assert "the" not in content_terms
        assert "list" not in content_terms
        assert "latest" not in content_terms
        assert "that" not in content_terms
        assert "mention" not in content_terms
    
    def test_edge_case_queries(self):
        """Test edge cases and unusual query patterns."""
        edge_cases = [
            "",  # Empty query
            "   ",  # Whitespace only
            "the and of",  # Only stopwords
            "list!!!",  # Special characters
            "find documents containing AI/ML research",  # Forward slash
        ]
        
        for query in edge_cases:
            # Should not crash
            result = ConstraintExtractor.extract(query)
            assert isinstance(result, QueryConstraints)
    
    @pytest.mark.parametrize("query,expected_recency", [
        ("latest files", True),
        ("newest documents", True),
        ("recent presentations", True),
        ("most recent pdfs", True), 
        ("current spreadsheets", True),
        ("all files", False),
        ("find documents", False),
        ("show spreadsheets", False),
    ])
    def test_recency_detection_comprehensive(self, query, expected_recency):
        """Comprehensive test for recency term detection."""
        result = ConstraintExtractor.extract(query)
        assert result.recency == expected_recency