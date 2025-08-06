"""Performance and end-to-end tests for the agent framework."""

import pytest
import time
import sys
from pathlib import Path
from unittest.mock import patch, Mock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from querying.agents.factory import create_default_agent
from interface.cli.ask import answer


class TestPerformance:
    """Test performance characteristics of the agent framework."""
    
    def test_agent_initialization_performance(self):
        """Test that agent initialization is reasonably fast."""
        start_time = time.time()
        agent = create_default_agent()
        initialization_time = time.time() - start_time
        
        # Should initialize in under 1 second
        assert initialization_time < 1.0
        assert agent.registry.get_plugin_count() == 2
    
    def test_metadata_query_performance(self):
        """Test metadata query performance."""
        agent = create_default_agent()
        
        start_time = time.time()
        result = agent.process_query("how many files do we have?")
        query_time = time.time() - start_time
        
        # Should complete in under 2 seconds (as per requirements)
        assert query_time < 2.0
        assert "files in the collection" in result
    
    def test_multiple_queries_performance(self):
        """Test performance with multiple sequential queries."""
        agent = create_default_agent()
        
        queries = [
            "how many files do we have?",
            "what file types are available?",
            "show me recent files",
            "count of PDF files",
            "list all files"
        ]
        
        start_time = time.time()
        for query in queries:
            result = agent.process_query(query)
            assert len(result) > 0  # Should get some response
        
        total_time = time.time() - start_time
        average_time = total_time / len(queries)
        
        # Average query time should be reasonable
        assert average_time < 1.0
        
        # Total time for 5 queries should be under 5 seconds
        assert total_time < 5.0
    
    def test_concurrent_query_handling(self):
        """Test that the agent can handle concurrent queries."""
        import threading
        import queue
        
        agent = create_default_agent()
        results_queue = queue.Queue()
        
        def query_worker(query, thread_id):
            start_time = time.time()
            result = agent.process_query(f"{query} (thread {thread_id})")
            query_time = time.time() - start_time
            results_queue.put((thread_id, result, query_time))
        
        # Start multiple threads with different queries
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=query_worker, 
                args=["how many files do we have?", i]
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 3
        
        # All queries should complete successfully
        for thread_id, result, query_time in results:
            assert "files in the collection" in result
            assert query_time < 2.0  # Each query should be fast


class TestEndToEnd:
    """End-to-end tests covering complete workflows."""
    
    def test_cli_backward_compatibility(self):
        """Test that CLI maintains exact backward compatibility."""
        # Test the exact same interface that existed before
        
        # Metadata query - should work without API key
        result = answer("how many files do we have?")
        assert isinstance(result, str)
        assert "files in the collection" in result
        assert "OpenAI API key" not in result
        
        # Content query - should show API key error
        result = answer("what is the compliance policy?")
        assert isinstance(result, str)
        assert "OpenAI API key not configured" in result
        
        # Empty query
        result = answer("")
        assert isinstance(result, str)
        assert "Please provide a question" in result
    
    def test_agent_capabilities_introspection(self):
        """Test agent introspection capabilities."""
        agent = create_default_agent()
        
        # Test capabilities listing
        capabilities = agent.get_capabilities()
        expected_capabilities = [
            "semantic_search", "document_query", "content_analysis", "vector_search",
            "metadata_query", "file_statistics", "collection_analysis", "file_counts", "file_types"
        ]
        
        for capability in expected_capabilities:
            assert capability in capabilities
    
    def test_agent_reasoning_explanation(self):
        """Test agent reasoning explanation."""
        agent = create_default_agent()
        
        # Initially no reasoning available
        explanation = agent.explain_reasoning()
        assert explanation is None
        
        # Process a query
        result = agent.process_query("how many files do we have?")
        assert len(result) > 0
        
        # Now should have reasoning
        explanation = agent.explain_reasoning()
        assert explanation is not None
        assert "Query: how many files do we have?" in explanation
        assert "Execution time:" in explanation
        assert "Plugins used:" in explanation
        assert "metadata" in explanation.lower()
    
    def test_query_classification_accuracy(self):
        """Test that queries are correctly classified."""
        agent = create_default_agent()
        
        # Test various metadata queries
        metadata_queries = [
            ("how many files", "metadata"),
            ("count of PDFs", "metadata"),
            ("what file types", "metadata"),
            ("show me files", "metadata"),
            ("list documents", "metadata"),
            ("recent files", "metadata"),
            ("total number of", "metadata")
        ]
        
        for query, expected_type in metadata_queries:
            agent.process_query(query)
            explanation = agent.explain_reasoning()
            
            if expected_type == "metadata":
                assert "metadata" in explanation.lower()
                # Should not trigger semantic search for metadata queries
                assert "semantic" not in explanation.lower() or "Using semantic search" not in explanation
        
        # Test content queries
        content_queries = [
            "what is the policy?",
            "explain compliance",
            "describe the procedure",
            "tell me about security"
        ]
        
        for query in content_queries:
            agent.process_query(query)
            explanation = agent.explain_reasoning()
            assert "semantic search" in explanation.lower()
    
    def test_error_handling_robustness(self):
        """Test error handling in various scenarios."""
        agent = create_default_agent()
        
        # Test with various problematic inputs
        problematic_queries = [
            "",  # Empty
            "   ",  # Whitespace only
            "a" * 1000,  # Very long query
            "SELECT * FROM chunks;",  # SQL injection attempt
            "🚀💻🔥",  # Unicode/emoji
            "\n\t\r",  # Control characters
        ]
        
        for query in problematic_queries:
            result = agent.process_query(query)
            # Should always return a string response, never crash
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_plugin_registry_management(self):
        """Test plugin registry management capabilities."""
        from querying.agents.registry import PluginRegistry
        from querying.agents.plugins.metadata_commands import MetadataCommandsPlugin
        
        registry = PluginRegistry()
        
        # Test registration
        plugin = MetadataCommandsPlugin()
        registry.register(plugin)
        
        assert registry.get_plugin_count() == 1
        assert registry.get_plugin("metadata") is not None
        
        # Test capability discovery
        capabilities = registry.list_capabilities()
        assert "metadata" in capabilities
        assert len(capabilities["metadata"]) > 0
        
        # Test finding plugins by capability
        metadata_plugins = registry.find_plugins_for_capability("metadata_query")
        assert len(metadata_plugins) == 1
        assert "metadata" in metadata_plugins
        
        # Test unregistration
        success = registry.unregister("metadata")
        assert success is True
        assert registry.get_plugin_count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])