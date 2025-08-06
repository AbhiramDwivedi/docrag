"""Test API integration with the agent framework."""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docquest.querying.api import app


class TestAPIIntegration:
    """Test API functionality with the agent framework."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_api_metadata_query(self):
        """Test API with metadata query."""
        response = self.client.post('/query', json={'question': 'how many files do we have?'})
        
        assert response.status_code == 200
        data = response.json()
        assert 'answer' in data
        assert (
            'files in the collection' in data['answer']
            or 'No document database found' in data['answer']
        )
        assert 'OpenAI API key' not in data['answer']
    
    def test_api_content_query(self):
        """Test API with content query."""
        response = self.client.post('/query', json={'question': 'what is the compliance policy?'})
        
        assert response.status_code == 200
        data = response.json()
        assert 'answer' in data
        assert 'OpenAI API key not configured' in data['answer']
    
    def test_api_file_types_query(self):
        """Test API with file types query."""
        response = self.client.post('/query', json={'question': 'what file types are available?'})
        
        assert response.status_code == 200
        data = response.json()
        assert 'answer' in data
        assert (
            'No files found' in data['answer']
            or 'File types in the collection' in data['answer']
            or 'No document database found' in data['answer']
        )
        assert 'OpenAI API key' not in data['answer']
    
    def test_api_empty_query(self):
        """Test API with empty query."""
        response = self.client.post('/query', json={'question': ''})
        
        assert response.status_code == 200
        data = response.json()
        assert 'answer' in data
        assert 'Please provide a question' in data['answer']
    
    def test_api_malformed_request(self):
        """Test API with malformed request."""
        # Missing question field
        response = self.client.post('/query', json={})
        
        # FastAPI should return validation error
        assert response.status_code == 422
    
    def test_api_query_classification(self):
        """Test query classification through API."""
        # Test multiple metadata queries
        metadata_queries = [
            "count of files",
            "list files",
            "show me recent documents",
            "how many PDFs?"
        ]
        
        for query in metadata_queries:
            response = self.client.post('/query', json={'question': query})
            assert response.status_code == 200
            data = response.json()
            # Should not require OpenAI API key
            assert 'OpenAI API key not configured' not in data['answer']
        
        # Test content queries
        content_queries = [
            "explain the policy",
            "what does it say about security?",
            "describe compliance requirements"
        ]
        
        for query in content_queries:
            response = self.client.post('/query', json={'question': query})
            assert response.status_code == 200
            data = response.json()
            # Should require OpenAI API key
            assert 'OpenAI API key not configured' in data['answer']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
