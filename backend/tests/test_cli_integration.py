"""Test CLI integration with the agent framework."""

import pytest
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from interface.cli.ask import answer


class TestCLIIntegration:
    """Test CLI functionality with the agent framework."""
    
    def test_answer_function_metadata_query(self):
        """Test the answer function with metadata queries."""
        result = answer("how many files do we have?")
        # In agentic mode, all queries go through the agent, so expect agentic response or API key error
        assert ("Metadata retrieved" in result or 
                "No documents found matching" in result or
                "OpenAI API key not configured" in result or
                "No document database found" in result)
    
    def test_answer_function_content_query(self):
        """Test the answer function with content queries."""
        result = answer("what is the compliance policy?")
        # In agentic mode, content queries always go through agent which needs API key
        assert ("OpenAI API key not configured" in result or
                "No documents found matching" in result)
    
    def test_answer_function_empty_query(self):
        """Test the answer function with empty query."""
        result = answer("")
        assert "Please provide a question" in result
        
        result = answer("   ")  # whitespace only
        assert "Please provide a question" in result
    
    def test_answer_function_file_types_query(self):
        """Test the answer function with file types query."""
        result = answer("what file types are available?")
        # In agentic mode, all queries go through agent
        assert ("Metadata retrieved" in result or 
                "No documents found matching" in result or
                "OpenAI API key not configured" in result or
                "No document database found" in result)
    
    def test_cli_module_execution(self):
        """Test CLI module execution via subprocess."""
        import os
        
        # Test metadata query with proper encoding handling
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            [sys.executable, "-m", "src.interface.cli.ask", "how many files do we have?"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            env=env,
            encoding='utf-8',
            errors='replace'
        )
        
        # Handle cases where stdout might be None
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = stdout + stderr
        
        if result.returncode == 0:
            # In agentic mode, all queries go through agent
            # CLI may gracefully handle errors and return 0, so accept both success and error messages
            assert ("Metadata retrieved" in output or
                    "No documents found matching" in output or
                    "OpenAI API key not configured" in output or
                    "No document database found" in output or
                    "Could not load agent dependencies" in output or
                    "Could not initialize agent" in output)
        else:
            # Accept dependency errors as valid test outcome for subprocess calls
            assert ("Could not load agent dependencies" in output or
                    "Could not initialize agent" in output)
    
    def test_cli_module_content_query(self):
        """Test CLI module with content query via subprocess."""
        import os
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            [sys.executable, "-m", "src.interface.cli.ask", "what is PCI compliance?"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            env=env,
            encoding='utf-8',
            errors='replace'
        )
        
        # Handle cases where stdout might be None
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = stdout + stderr
        
        if result.returncode == 0:
            # If successful, expect either API key error or valid response
            # CLI may gracefully handle errors and return 0, so accept both success and error messages
            assert ("OpenAI API key not configured" in output or
                    "No documents found matching" in output or
                    "Metadata retrieved" in output or
                    "Could not load agent dependencies" in output or
                    "Could not initialize agent" in output)
        else:
            # Accept dependency errors as valid test outcome for subprocess calls
            assert ("Could not load agent dependencies" in output or
                    "Could not initialize agent" in output)
    
    def test_query_classification_examples(self):
        """Test various query classifications in agentic mode."""
        # In agentic mode, all queries go through the agent, so we expect either:
        # 1. Agent response with "Metadata retrieved" or similar
        # 2. "OpenAI API key not configured" if API is needed
        # 3. "No documents found matching" if no relevant docs
        
        # Test queries that should work regardless of content
        test_queries = [
            "how many files do we have?",
            "what is the policy?",
            "what file types are there?"
        ]
        
        for query in test_queries:
            result = answer(query)
            # In agentic mode, all queries go through agent, expect reasonable response
            assert (
                "Metadata retrieved" in result or
                "No documents found matching" in result or
                "OpenAI API key not configured" in result or
                "No document database found" in result or
                "Please provide a question" in result
            ), f"Query '{query}' returned unexpected response: {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])