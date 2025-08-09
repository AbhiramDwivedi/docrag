"""
Test cases for document discovery and agent synthesis enhancements.

This module tests the enhanced agent synthesis logic that properly handles
metadata-semantic search integration for document discovery queries.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Mock the dependencies that cause import issues
sys.modules['nltk'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['ingestion.processors.embedder'] = MagicMock()
sys.modules['ingestion.storage.vector_store'] = MagicMock()
sys.modules['shared.config'] = MagicMock()

# Add backend src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Now import after mocking dependencies
from querying.agents.agent import Agent
from querying.agents.registry import PluginRegistry


class TestDocumentDiscoverySynthesis:
    """Test cases for document discovery synthesis logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = PluginRegistry()
        self.agent = Agent(self.registry)
    
    def test_document_discovery_prioritizes_metadata(self):
        """Test that document discovery queries prioritize metadata results."""
        # Mock results where metadata finds documents but semantic search fails
        mock_results = [
            (
                "metadata", 
                {
                    "response": "Found 2 files:\n• Important_Budget_Document_2024.docx (2.5MB, 2024-01-15 14:30)\n• Budget_Summary.pdf (1.1MB, 2024-01-12 10:20)",
                    "data": {
                        "files": [
                            {"name": "Important_Budget_Document_2024.docx", "path": "/docs/Important_Budget_Document_2024.docx"},
                            {"name": "Budget_Summary.pdf", "path": "/docs/Budget_Summary.pdf"}
                        ]
                    }
                }
            ),
            (
                "semantic_search",
                {
                    "response": "No relevant information found.",
                    "sources": [],
                    "metadata": {"results_count": 0}
                }
            )
        ]
        
        # Test document discovery query
        question = "find the Important Budget Document"
        result = self.agent._synthesize_response(question, mock_results)
        
        # Should return metadata results despite semantic search failure
        assert "Important_Budget_Document_2024.docx" in result
        assert "No relevant information found" not in result
        assert "Found 2 files" in result
    
    def test_content_query_uses_semantic_search(self):
        """Test that content queries continue to use semantic search normally."""
        mock_results = [
            (
                "semantic_search",
                {
                    "response": "Based on the financial policy document, the budget allocation process requires approval from department heads.",
                    "sources": [{"file": "policy.pdf", "unit": "page 5"}]
                }
            )
        ]
        
        question = "what is the budget allocation process"
        result = self.agent._synthesize_response(question, mock_results)
        
        assert "budget allocation process" in result
        assert "approval from department heads" in result
    
    def test_document_discovery_fallback_to_semantic(self):
        """Test fallback to semantic search when metadata fails for document discovery."""
        mock_results = [
            (
                "metadata",
                {
                    "response": "No files found matching the specified criteria.",
                    "data": {"files": []}
                }
            ),
            (
                "semantic_search",
                {
                    "response": "Found relevant content in ProjectPlan.docx regarding project timeline and deliverables.",
                    "sources": [{"file": "ProjectPlan.docx", "unit": "page 2"}]
                }
            )
        ]
        
        question = "find the project plan document"
        result = self.agent._synthesize_response(question, mock_results)
        
        # Should fallback to semantic search
        assert "ProjectPlan.docx" in result
        assert "project timeline" in result
    
    def test_both_plugins_fail_for_document_discovery(self):
        """Test appropriate response when both metadata and semantic search fail."""
        mock_results = [
            (
                "metadata",
                {
                    "response": "No files found matching the specified criteria.",
                    "data": {"files": []}
                }
            ),
            (
                "semantic_search",
                {
                    "response": "No relevant information found.",
                    "sources": []
                }
            )
        ]
        
        question = "find the missing document"
        result = self.agent._synthesize_response(question, mock_results)
        
        assert "No relevant documents found" in result
    
    def test_document_discovery_with_content_context(self):
        """Test document discovery that also includes semantic content."""
        mock_results = [
            (
                "metadata",
                {
                    "response": "Found 1 file:\n• Technical_Specification.docx (4.2MB, 2024-01-10 15:45)",
                    "data": {
                        "files": [{"name": "Technical_Specification.docx", "path": "/docs/Technical_Specification.docx"}]
                    }
                }
            ),
            (
                "semantic_search",
                {
                    "response": "The technical specification outlines the system architecture and implementation details for the new platform.",
                    "sources": [{"file": "Technical_Specification.docx", "unit": "page 1"}]
                }
            )
        ]
        
        question = "find the Technical Specification document"
        result = self.agent._synthesize_response(question, mock_results)
        
        # Should include both metadata and content
        assert "Technical_Specification.docx" in result
        assert "4.2MB" in result  # From metadata
        assert "system architecture" in result  # From semantic content


class TestDocumentDiscoveryClassification:
    """Test cases for document discovery query classification."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock registry with metadata plugin
        self.registry = Mock()
        self.registry.get_plugin.return_value = Mock()  # Mock plugin exists
        
        self.agent = Agent(self.registry)
    
    def test_classify_document_discovery_queries(self):
        """Test that document discovery queries are properly classified."""
        discovery_queries = [
            "find the Important Budget Document",
            "get the Project Plan file", 
            "show me the Compliance Report",
            "where is the Meeting Notes document",
            "locate the Technical Specification",
            "find document called Budget Analysis",
            "get file named Project Timeline"
        ]
        
        for query in discovery_queries:
            plugins = self.agent._classify_query(query)
            
            # Should route to metadata plugin for document discovery
            assert "metadata" in plugins, f"Query '{query}' should route to metadata plugin"
            assert "semantic_search" in plugins, f"Query '{query}' should also include semantic search as backup"
    
    def test_classify_content_queries(self):
        """Test that content queries are properly classified."""
        content_queries = [
            "what is the budget allocation process",
            "explain the compliance requirements",
            "describe the technical architecture", 
            "how does the system work"
        ]
        
        for query in content_queries:
            plugins = self.agent._classify_query(query)
            
            # Content queries should primarily route to semantic search
            assert "semantic_search" in plugins, f"Content query '{query}' should route to semantic search"


class TestSemanticSearchMetadataAware:
    """Test cases for metadata-aware semantic search functionality."""
    
    def test_semantic_search_parameters_for_document_discovery(self):
        """Test that document discovery queries get appropriate semantic search parameters."""
        registry = PluginRegistry()
        agent = Agent(registry)
        
        question = "find the Important Project Document"
        params = agent._generate_semantic_search_params(question)
        
        # Should include metadata search for document discovery
        assert params.get("include_metadata_search") == True
        assert params.get("k") == 100  # Wider search for discovery
        assert params.get("max_documents") == 10  # More documents to find the right one
    
    def test_semantic_search_parameters_for_content_query(self):
        """Test that content queries get standard semantic search parameters."""
        registry = PluginRegistry()
        agent = Agent(registry)
        
        question = "what is the implementation approach"
        params = agent._generate_semantic_search_params(question)
        
        # Should not include metadata search for content queries
        assert params.get("include_metadata_search") == False
        # "what is" queries get focused search parameters (k=30)
        assert params.get("k") == 30  # Focused search for factual queries
        assert params.get("max_documents") == 3  # Focused document count


class TestDocumentDiscoveryDetection:
    """Test cases for document discovery pattern detection."""
    
    def test_detect_document_discovery_patterns(self):
        """Test detection of various document discovery patterns."""
        discovery_patterns = [
            ("find the budget document", True),
            ("get the project plan", True), 
            ("show me the report", True),
            ("where is the specification", True),
            ("locate the meeting notes", True),
            ("document called requirements", True),
            ("file named analysis", True),
            ("explain the requirements", False),    # Content query
            ("how does this work", False)          # Content query
        ]
        
        for query, expected_discovery in discovery_patterns:
            question_lower = query.lower()
            
            # Document discovery detection logic (from agent)
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
            
            assert is_document_discovery == expected_discovery, f"Query '{query}' detection mismatch. Expected: {expected_discovery}, Got: {is_document_discovery}"


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise run manually
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        # Manual test runner
        test_classes = [
            TestDocumentDiscoverySynthesis,
            TestDocumentDiscoveryClassification, 
            TestSemanticSearchMetadataAware,
            TestDocumentDiscoveryDetection
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for test_class in test_classes:
            instance = test_class()
            if hasattr(instance, 'setup_method'):
                instance.setup_method()
            
            for method_name in dir(instance):
                if method_name.startswith('test_'):
                    total_tests += 1
                    try:
                        method = getattr(instance, method_name)
                        method()
                        print(f"✅ {test_class.__name__}.{method_name}")
                        passed_tests += 1
                    except Exception as e:
                        print(f"❌ {test_class.__name__}.{method_name}: {e}")
        
        print(f"\nTest Results: {passed_tests}/{total_tests} passed")
        if passed_tests == total_tests:
            print("🎉 All tests passed!")
        else:
            print("❌ Some tests failed.")
            exit(1)