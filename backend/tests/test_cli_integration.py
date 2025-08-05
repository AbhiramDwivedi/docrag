"""Test CLI integration with the agent framework."""

import pytest
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from interface.cli.ask import answer


class TestCLIIntegration:
    """Test CLI functionality with the agent framework."""
    
    def test_answer_function_metadata_query(self):
        """Test the answer function with metadata queries."""
        result = answer("how many files do we have?")
        assert "files in the collection" in result
        assert "OpenAI API key" not in result
    
    def test_answer_function_content_query(self):
        """Test the answer function with content queries."""
        result = answer("what is the compliance policy?")
        assert "OpenAI API key not configured" in result
    
    def test_answer_function_empty_query(self):
        """Test the answer function with empty query."""
        result = answer("")
        assert "Please provide a question" in result
        
        result = answer("   ")  # whitespace only
        assert "Please provide a question" in result
    
    def test_answer_function_file_types_query(self):
        """Test the answer function with file types query."""
        result = answer("what file types are available?")
        assert ("No files found" in result or "File types in the collection" in result)
        assert "OpenAI API key" not in result
    
    def test_cli_module_execution(self):
        """Test CLI module execution via subprocess."""
        # Test metadata query
        result = subprocess.run(
            [sys.executable, "-m", "interface.cli.ask", "how many files do we have?"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "files in the collection" in result.stdout
        assert "OpenAI API key" not in result.stdout
    
    def test_cli_module_content_query(self):
        """Test CLI module with content query via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "interface.cli.ask", "what is PCI compliance?"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "OpenAI API key not configured" in result.stdout
    
    def test_query_classification_examples(self):
        """Test various query classifications."""
        # Metadata queries - should work without API key
        metadata_queries = [
            "how many files do we have?",
            "count of PDF files",
            "list all files",
            "show me recent files",
            "what file types are there?",
            "total number of documents"
        ]
        
        for query in metadata_queries:
            result = answer(query)
            # Metadata queries should not require API key
            assert "OpenAI API key not configured" not in result, f"Query '{query}' incorrectly routed to semantic search"
        
        # Content queries - should require API key
        content_queries = [
            "what is the policy?",
            "explain compliance requirements",
            "describe the procedure",
            "what does the document say about security?",
            "find information about data protection"
        ]
        
        for query in content_queries:
            result = answer(query)
            # Content queries should require API key
            assert "OpenAI API key not configured" in result, f"Query '{query}' not routed to semantic search"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])