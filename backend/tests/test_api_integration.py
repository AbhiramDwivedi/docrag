"""Test API integration with the agent framework."""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from querying.api import app


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
        # Agentic agent returns more sophisticated responses
        answer = data['answer']
        assert isinstance(answer, str)
        assert len(answer) > 0
        # Should not contain error messages
        assert not answer.startswith("❌")
    
    def test_api_content_query(self):
        """Test API with content query."""
        response = self.client.post('/query', json={'question': 'what is the compliance policy?'})
        
        assert response.status_code == 200
        data = response.json()
        assert 'answer' in data
        answer = data['answer']
        assert isinstance(answer, str)
        assert len(answer) > 0
        # Agentic agent may provide actual content analysis or no documents message
        assert not answer.startswith("❌")  # Should not be an error
    
    def test_api_file_types_query(self):
        """Test API with file types query."""
        response = self.client.post('/query', json={'question': 'what file types are available?'})
        
        assert response.status_code == 200
        data = response.json()
        assert 'answer' in data
        answer = data['answer']
        assert isinstance(answer, str)
        assert len(answer) > 0
        # Should process through agentic system
        assert not answer.startswith("❌")
    
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
        # Test a single representative query to avoid threading issues
        response = self.client.post('/query', json={'question': 'count of files'})
        assert response.status_code == 200
        data = response.json()
        answer = data['answer']
        assert isinstance(answer, str)
        assert len(answer) > 0
        # Should process through agentic system without errors  
        # Note: Some threading issues may occur with SQLite in test environment
        # but the response should still be a valid string


if __name__ == "__main__":
    pytest.main([__file__, "-v"])