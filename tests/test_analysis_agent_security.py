"""Security validation tests for analysis agent."""

import pytest
from unittest.mock import Mock

from backend.src.querying.agents.agentic.analysis_agent import AnalysisAgent
from backend.src.querying.agents.agentic.execution_plan import ExecutionStep, StepType
from backend.src.querying.agents.agentic.context import AgentContext
from backend.src.querying.agents.registry import PluginRegistry


class TestAnalysisAgentSecurity:
    """Test security validation in analysis agent."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock plugin registry."""
        registry = Mock(spec=PluginRegistry)
        
        # Mock semantic search plugin
        semantic_plugin = Mock()
        semantic_plugin.validate_params.return_value = True
        semantic_plugin.execute.return_value = {
            "response": "Test response",
            "sources": []
        }
        
        registry.get_plugin.return_value = semantic_plugin
        return registry
    
    @pytest.fixture
    def analysis_agent(self, mock_registry):
        """Create analysis agent with mocked registry."""
        return AnalysisAgent(mock_registry)
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock agent context."""
        context = Mock(spec=AgentContext)
        context.query = "test query"
        context.discovered_documents = []
        return context
    
    def test_query_validation_rejects_sql_injection(self, analysis_agent):
        """Test that SQL injection attempts are rejected."""
        dangerous_queries = [
            "'; DROP TABLE files; --",
            "test UNION SELECT * FROM files",
            "query'; DELETE FROM chunks;",
            "search <script>alert('xss')</script>",
            "find javascript:void(0)",
        ]
        
        for dangerous_query in dangerous_queries:
            with pytest.raises(ValueError, match="dangerous pattern"):
                analysis_agent._validate_query_parameter(dangerous_query)
    
    def test_query_validation_accepts_safe_queries(self, analysis_agent):
        """Test that safe queries are accepted."""
        safe_queries = [
            "find documents about budget",
            "search for quarterly reports",
            "locate files with project information",
            "budget planning documents",
            "meeting notes from last week",
        ]
        
        for safe_query in safe_queries:
            # Should not raise exception
            validated = analysis_agent._validate_query_parameter(safe_query)
            assert validated == safe_query
    
    def test_query_validation_length_limit(self, analysis_agent):
        """Test query length validation."""
        # Test maximum allowed length
        max_query = "a" * 1000
        validated = analysis_agent._validate_query_parameter(max_query)
        assert len(validated) == 1000
        
        # Test over limit
        over_limit_query = "a" * 1001
        with pytest.raises(ValueError, match="too long"):
            analysis_agent._validate_query_parameter(over_limit_query)
    
    def test_query_validation_empty_or_invalid(self, analysis_agent):
        """Test validation of empty or invalid queries."""
        invalid_queries = [
            "",  # Empty string
            "   ",  # Whitespace only
            None,  # None
            123,  # Not a string
            [],  # Wrong type
        ]
        
        for invalid_query in invalid_queries:
            with pytest.raises(ValueError, match="non-empty string"):
                analysis_agent._validate_query_parameter(invalid_query)
    
    def test_count_validation_accepts_valid_counts(self, analysis_agent):
        """Test that valid count parameters are accepted."""
        valid_counts = [1, 5, 10, 50, 100, 500, 1000, "10", "25", 25.0]
        
        for count in valid_counts:
            validated = analysis_agent._validate_count_parameter(count)
            assert isinstance(validated, int)
            assert validated > 0
    
    def test_count_validation_rejects_invalid_counts(self, analysis_agent):
        """Test that invalid count parameters are rejected."""
        invalid_counts = [
            0,  # Zero
            -1,  # Negative
            -10,  # Negative
            1001,  # Too large
            10000,  # Way too large
            "invalid",  # Non-numeric string
            None,  # None
            [],  # Wrong type
        ]
        
        for count in invalid_counts:
            with pytest.raises(ValueError):
                analysis_agent._validate_count_parameter(count)
    
    def test_target_documents_validation_accepts_valid_docs(self, analysis_agent):
        """Test that valid target document lists are accepted."""
        valid_doc_lists = [
            [{"path": "/path/to/doc1.pdf"}],
            [{"path": "/doc1.pdf"}, {"file_path": "/doc2.pdf"}],
            [{"path": "/documents/report.pdf", "name": "report.pdf"}],
        ]
        
        for doc_list in valid_doc_lists:
            validated = analysis_agent._validate_target_documents(doc_list)
            assert isinstance(validated, list)
            assert len(validated) == len(doc_list)
            assert all(isinstance(path, str) for path in validated)
    
    def test_target_documents_validation_rejects_invalid_docs(self, analysis_agent):
        """Test that invalid target document lists are rejected."""
        invalid_doc_lists = [
            [],  # Empty list
            None,  # None
            "not a list",  # Wrong type
            [{}],  # Empty dict
            [{"name": "doc.pdf"}],  # Missing path
            [{"path": ""}],  # Empty path
            [{"path": None}],  # None path
            [{"path": 123}],  # Non-string path
        ]
        
        for doc_list in invalid_doc_lists:
            with pytest.raises(ValueError):
                analysis_agent._validate_target_documents(doc_list)
    
    def test_target_documents_validation_size_limit(self, analysis_agent):
        """Test target documents list size limits."""
        # Test maximum allowed size
        max_docs = [{"path": f"/doc{i}.pdf"} for i in range(1000)]
        validated = analysis_agent._validate_target_documents(max_docs)
        assert len(validated) == 1000
        
        # Test over limit
        over_limit_docs = [{"path": f"/doc{i}.pdf"} for i in range(1001)]
        with pytest.raises(ValueError, match="Too many target documents"):
            analysis_agent._validate_target_documents(over_limit_docs)
    
    def test_target_documents_path_length_limit(self, analysis_agent):
        """Test individual document path length limits."""
        # Test maximum allowed path length
        long_path = "/path/to/" + "a" * 490 + ".pdf"  # Total ~500 chars
        docs = [{"path": long_path}]
        validated = analysis_agent._validate_target_documents(docs)
        assert len(validated) == 1
        
        # Test over limit path
        too_long_path = "/path/to/" + "a" * 500 + ".pdf"  # Over 500 chars
        docs = [{"path": too_long_path}]
        with pytest.raises(ValueError, match="path too long"):
            analysis_agent._validate_target_documents(docs)
    
    def test_security_validation_in_analyze_content(self, analysis_agent, mock_registry, mock_context):
        """Test that security validation is applied in _analyze_content method."""
        # Create step with dangerous query
        step = ExecutionStep(
            step_type=StepType.ANALYZE_CONTENT,
            agent_id="analysis",
            parameters={
                "query": "'; DROP TABLE files; --",
                "max_documents": 5,
                "target_docs_from_step": Mock(id="step1")
            }
        )
        
        # Mock get_target_documents to return valid documents
        analysis_agent._get_target_documents = Mock(return_value=[
            {"path": "/doc1.pdf"}
        ])
        
        # Execute step - should fail due to dangerous query
        result = analysis_agent._analyze_content(step, mock_context)
        
        assert not result.is_successful()
        assert "dangerous pattern" in result.error_message
    
    def test_security_validation_in_extract_content(self, analysis_agent, mock_registry, mock_context):
        """Test that security validation is applied in _extract_content method."""
        # Create step with invalid count parameter
        step = ExecutionStep(
            step_type=StepType.EXTRACT_CONTENT,
            agent_id="analysis",
            parameters={
                "query": "valid query",
                "max_documents": -5,  # Invalid count
            }
        )
        
        # Execute step - should fail due to invalid count
        result = analysis_agent._extract_content(step, mock_context)
        
        assert not result.is_successful()
        assert "Invalid extraction parameters" in result.error_message
    
    def test_parameter_validation_preserves_functionality(self, analysis_agent, mock_registry, mock_context):
        """Test that security validation doesn't break normal functionality."""
        # Create step with completely valid parameters
        step = ExecutionStep(
            step_type=StepType.ANALYZE_CONTENT,
            agent_id="analysis",
            parameters={
                "query": "find budget documents",
                "max_documents": 10,
                "k": 20,
                "target_docs_from_step": Mock(id="step1")
            }
        )
        
        # Mock get_target_documents to return valid documents
        analysis_agent._get_target_documents = Mock(return_value=[
            {"path": "/budget_q1.pdf"},
            {"path": "/budget_q2.pdf"}
        ])
        
        # Mock context.get_step_result to return step results
        mock_context.get_step_result = Mock(return_value=Mock(
            is_successful=lambda: True,
            get_result_data=lambda: {
                "discovered_documents": [
                    {"path": "/budget_q1.pdf"},
                    {"path": "/budget_q2.pdf"}
                ]
            }
        ))
        mock_context.add_extracted_content = Mock()
        
        # Execute step - should succeed with valid parameters
        result = analysis_agent._analyze_content(step, mock_context)
        
        # Should succeed (though mock execution might not be fully realistic)
        # The important thing is that validation didn't prevent execution
        assert result is not None
        
        # Verify that the plugin was called with validated parameters
        semantic_plugin = mock_registry.get_plugin("semantic_search")
        assert semantic_plugin.execute.called
        
        # Verify parameters passed to plugin are sanitized
        call_args = semantic_plugin.execute.call_args[0][0]
        assert call_args["question"] == "find budget documents"
        assert call_args["max_documents"] == 10
        assert call_args["k"] == 20
        assert isinstance(call_args["target_docs"], list)
        assert len(call_args["target_docs"]) == 2
    
    def test_path_traversal_detection(self, analysis_agent):
        """Test detection of potential path traversal attacks."""
        # These should be allowed but logged as warnings
        suspicious_paths = [
            {"path": "/absolute/path/to/doc.pdf"},
            {"path": "../relative/path/doc.pdf"},
        ]
        
        for path_dict in suspicious_paths:
            # Should not raise exception but might log warnings
            validated = analysis_agent._validate_target_documents([path_dict])
            assert len(validated) == 1
    
    def test_fallback_path_extraction(self, analysis_agent):
        """Test fallback mechanisms for extracting document paths."""
        various_path_formats = [
            {"path": "/doc1.pdf"},
            {"file_path": "/doc2.pdf"},
            {"path": "/doc3.pdf", "file_path": "/ignored.pdf"},  # path takes precedence
        ]
        
        validated = analysis_agent._validate_target_documents(various_path_formats)
        
        expected_paths = ["/doc1.pdf", "/doc2.pdf", "/doc3.pdf"]
        assert validated == expected_paths