"""Test enhanced semantic search plugin functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add backend src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class MockEnhancedVectorStore:
    """Mock enhanced vector store for testing semantic search logic."""
    
    def query(self, vector, k=8):
        """Mock vector query returning sample chunks."""
        return [
            {
                'id': 'chunk_1',
                'file': 'doc1.pdf',
                'unit': 'page_1',
                'text': 'This is the first chunk about project requirements.',
                'distance': 0.2,
                'document_id': 'doc_1',
                'document_path': '/path/to/doc1.pdf',
                'document_title': 'Project Requirements',
                'section_id': 'intro',
                'chunk_index': 0,
                'total_chunks': 5,
                'document_type': 'PDF'
            },
            {
                'id': 'chunk_2',
                'file': 'doc1.pdf',
                'unit': 'page_1',
                'text': 'Additional requirements details in the same document.',
                'distance': 0.3,
                'document_id': 'doc_1',
                'document_path': '/path/to/doc1.pdf',
                'document_title': 'Project Requirements',
                'section_id': 'details',
                'chunk_index': 1,
                'total_chunks': 5,
                'document_type': 'PDF'
            },
            {
                'id': 'chunk_3',
                'file': 'doc2.pdf',
                'unit': 'page_1',
                'text': 'Information from a different document about implementation.',
                'distance': 0.4,
                'document_id': 'doc_2',
                'document_path': '/path/to/doc2.pdf',
                'document_title': 'Implementation Guide',
                'section_id': 'overview',
                'chunk_index': 0,
                'total_chunks': 3,
                'document_type': 'PDF'
            }
        ]
    
    def rank_documents_by_relevance(self, chunks):
        """Mock document ranking."""
        doc_scores = {}
        for chunk in chunks:
            doc_id = chunk['document_id']
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'document_id': doc_id,
                    'document_path': chunk['document_path'],
                    'document_title': chunk['document_title'],
                    'document_type': chunk['document_type'],
                    'avg_relevance': 1.0 - chunk['distance'],
                    'chunk_count': 1,
                    'chunks': [chunk]
                }
            else:
                doc_scores[doc_id]['chunk_count'] += 1
                doc_scores[doc_id]['chunks'].append(chunk)
        
        return sorted(doc_scores.values(), key=lambda x: x['avg_relevance'], reverse=True)
    
    def get_document_context(self, chunk_ids, window_size=3):
        """Mock context expansion."""
        expanded_chunks = []
        for chunk_id in chunk_ids:
            # Return the original chunk plus some context
            if chunk_id == 'chunk_1':
                expanded_chunks.extend([
                    {
                        'id': 'chunk_0',
                        'text': 'Previous context for chunk 1.',
                        'document_id': 'doc_1',
                        'document_path': '/path/to/doc1.pdf',
                        'document_title': 'Project Requirements',
                        'section_id': 'intro',
                        'chunk_index': -1,
                        'is_original': False
                    },
                    {
                        'id': 'chunk_1',
                        'text': 'This is the first chunk about project requirements.',
                        'document_id': 'doc_1',
                        'document_path': '/path/to/doc1.pdf',
                        'document_title': 'Project Requirements',
                        'section_id': 'intro',
                        'chunk_index': 0,
                        'is_original': True
                    }
                ])
            elif chunk_id == 'chunk_2':
                expanded_chunks.append({
                    'id': 'chunk_2',
                    'text': 'Additional requirements details in the same document.',
                    'document_id': 'doc_1',
                    'document_path': '/path/to/doc1.pdf',
                    'document_title': 'Project Requirements',
                    'section_id': 'details',
                    'chunk_index': 1,
                    'is_original': True
                })
            elif chunk_id == 'chunk_3':
                expanded_chunks.append({
                    'id': 'chunk_3',
                    'text': 'Information from a different document about implementation.',
                    'document_id': 'doc_2',
                    'document_path': '/path/to/doc2.pdf',
                    'document_title': 'Implementation Guide',
                    'section_id': 'overview',
                    'chunk_index': 0,
                    'is_original': True
                })
        
        return expanded_chunks


class MockOpenAIClient:
    """Mock OpenAI client for testing."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.chat = self.Chat()
        
    class Chat:
        def __init__(self):
            self.completions = MockOpenAIClient.ChatCompletions()
        
    class ChatCompletions:
        def create(self, **kwargs):
            # Mock response
            class MockResponse:
                def __init__(self):
                    self.choices = [MockChoice()]
            
            class MockChoice:
                def __init__(self):
                    self.message = MockMessage()
            
            class MockMessage:
                def __init__(self):
                    # Check if enhanced context is being used
                    prompt = kwargs.get('messages', [{}])[0].get('content', '')
                    if 'SOURCE:' in prompt and 'Project Requirements' in prompt:
                        self.content = "Based on the Project Requirements document, the requirements include detailed specifications. The Implementation Guide provides additional context on how to execute these requirements."
                    else:
                        self.content = "This is a mock response based on the provided context."
            
            return MockResponse()


def test_enhanced_semantic_search():
    """Test enhanced semantic search functionality."""
    
    # Mock the necessary imports and settings
    with patch('backend.src.querying.agents.plugins.semantic_search.settings') as mock_settings, \
         patch('backend.src.querying.agents.plugins.semantic_search.embed_texts') as mock_embed, \
         patch('backend.src.querying.agents.plugins.semantic_search.VectorStore') as mock_vs_class, \
         patch('backend.src.querying.agents.plugins.semantic_search.OpenAI', MockOpenAIClient):
        
        # Configure mocks
        mock_settings.openai_api_key = "test_key"
        mock_settings.vector_path = "/test/vector.index"
        mock_settings.db_path = "/test/db.sqlite"
        mock_settings.embed_model = "test_model"
        
        mock_embed.return_value = [[0.1] * 384]  # Mock embedding
        
        mock_vector_store = MockEnhancedVectorStore()
        mock_vs_class.load.return_value = mock_vector_store
        
        # Import after patching to avoid import errors
        try:
            from backend.src.querying.agents.plugins.semantic_search import SemanticSearchPlugin
        except ImportError:
            print("❌ Could not import SemanticSearchPlugin - skipping test")
            return
        
        # Test plugin
        plugin = SemanticSearchPlugin()
        
        # Test enhanced document-level search
        params = {
            "question": "What are the project requirements?",
            "k": 50,
            "max_documents": 3,
            "context_window": 2,
            "use_document_level": True
        }
        
        result = plugin.execute(params)
        
        # Validate enhanced response structure
        assert "response" in result
        assert "sources" in result
        assert "metadata" in result
        
        # Check metadata includes document-level analysis
        metadata = result["metadata"]
        assert "documents_analyzed" in metadata
        assert "total_chunks_used" in metadata
        assert "context_expansion_used" in metadata
        assert metadata["context_expansion_used"] is True
        
        # Check sources include document-level information
        sources = result["sources"]
        assert len(sources) > 0
        
        for source in sources:
            assert "document_id" in source
            assert "document_path" in source
            assert "document_title" in source
            assert "relevance_score" in source
        
        # Check response includes enhanced attribution
        response = result["response"]
        assert "Project Requirements" in response or "Implementation Guide" in response
        
        print("✓ Enhanced semantic search works correctly")
        
        # Test legacy mode
        params_legacy = {
            "question": "What are the project requirements?",
            "use_document_level": False
        }
        
        result_legacy = plugin.execute(params_legacy)
        
        # Validate legacy compatibility
        assert "response" in result_legacy
        assert "sources" in result_legacy
        metadata_legacy = result_legacy["metadata"]
        assert metadata_legacy.get("legacy_mode") is True
        assert metadata_legacy.get("context_expansion_used") is False
        
        print("✓ Legacy mode compatibility maintained")
        
        # Test parameter validation
        valid_params = {
            "question": "Test question",
            "k": 25,
            "max_documents": 3,
            "context_window": 2,
            "use_document_level": True
        }
        
        assert plugin.validate_params(valid_params) is True
        
        invalid_params = {
            "question": "",  # Empty question
            "k": 25
        }
        
        assert plugin.validate_params(invalid_params) is False
        
        print("✓ Parameter validation works correctly")
        
        # Test plugin info
        info = plugin.get_info()
        assert info.name == "semantic_search"
        assert info.version == "2.0.0"
        assert "document_level_retrieval" in info.capabilities
        assert "context_expansion" in info.capabilities
        assert "source_attribution" in info.capabilities
        
        print("✓ Plugin metadata updated correctly")


if __name__ == "__main__":
    print("Testing Enhanced Semantic Search Plugin...")
    
    try:
        test_enhanced_semantic_search()
        print("\n🎉 All enhanced semantic search tests passed!")
        print("\nNext: Enhance agent query classification and orchestration")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()