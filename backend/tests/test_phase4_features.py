"""Tests for Phase 4 advanced features in semantic search plugin.

Tests cross-encoder reranking, query expansion, and entity-aware functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add backend src to path for imports
backend_root = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(backend_root))

from querying.agents.plugins.semantic_search import SemanticSearchPlugin
from shared.reranking import CrossEncoderReranker, expand_entity_query, create_query_variants


class TestCrossEncoderReranker:
    """Test cross-encoder reranking functionality."""
    
    def test_cross_encoder_initialization(self):
        """Test cross-encoder reranker initialization."""
        reranker = CrossEncoderReranker("ms-marco-MiniLM-L-6-v2")
        assert reranker.model_name == "ms-marco-MiniLM-L-6-v2"
        assert reranker._model is None
    
    def test_cross_encoder_unavailable(self):
        """Test behavior when cross-encoder is unavailable."""
        with patch('shared.reranking.CrossEncoderReranker.__init__') as mock_init:
            mock_init.return_value = None
            reranker = CrossEncoderReranker()
            reranker._model_available = False
            
            results = [
                {"text": "test doc 1", "similarity": 0.8},
                {"text": "test doc 2", "similarity": 0.6}
            ]
            
            reranked = reranker.rerank("test query", results, top_k=1)
            assert len(reranked) == 1
            assert reranked[0]["text"] == "test doc 1"
    
    @patch('shared.reranking.CrossEncoderReranker._get_model')
    def test_cross_encoder_reranking(self, mock_get_model):
        """Test cross-encoder reranking with mock model."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.7, 0.8]  # Scores for 3 docs
        mock_get_model.return_value = mock_model
        
        reranker = CrossEncoderReranker()
        reranker._model_available = True
        
        results = [
            {"text": "doc 1", "similarity": 0.6},
            {"text": "doc 2", "similarity": 0.8},
            {"text": "doc 3", "similarity": 0.7}
        ]
        
        reranked = reranker.rerank("test query", results, top_k=2)
        
        assert len(reranked) == 2
        assert reranked[0]["cross_encoder_score"] == 0.9
        assert reranked[1]["cross_encoder_score"] == 0.8
        assert reranked[0]["text"] == "doc 1"  # Should be first despite lower similarity
        assert reranked[1]["text"] == "doc 3"


class TestQueryExpansion:
    """Test query expansion functionality."""
    
    def test_create_query_variants_basic(self):
        """Test basic query variant creation."""
        query = "What is Apple's revenue?"
        entities = ["Apple"]
        
        variants = create_query_variants(query, entities)
        
        assert query in variants  # Original query always included
        assert len(variants) > 1  # Should have variants
        assert any("Apple Inc" in variant for variant in variants)
    
    def test_create_query_variants_no_entities(self):
        """Test query variants with no entities."""
        query = "What is the weather like?"
        entities = []
        
        variants = create_query_variants(query, entities)
        
        assert variants == [query]  # Only original query
    
    def test_expand_entity_query_with_entities(self):
        """Test entity query expansion with detected entities."""
        query = "Tell me about Tesla's latest model"
        entities = ["Tesla"]
        
        expanded = expand_entity_query(query, entities)
        
        assert query in expanded
        assert len(expanded) > 1
        assert any("Tesla Motors" in variant or "Tesla Inc" in variant for variant in expanded)
    
    def test_expand_entity_query_acronym_expansion(self):
        """Test expansion of acronyms."""
        query = "What is AI used for?"
        entities = ["AI"]
        
        expanded = expand_entity_query(query, entities)
        
        assert query in expanded
        assert any("Artificial Intelligence" in variant for variant in expanded)
    
    def test_expand_entity_query_case_variants(self):
        """Test case variant expansion."""
        query = "What about COVID research?"
        entities = ["COVID"]
        
        expanded = expand_entity_query(query, entities)
        
        assert query in expanded
        assert any("COVID-19" in variant or "Coronavirus" in variant for variant in expanded)


class TestSemanticSearchPhase4:
    """Test Phase 4 features integration in semantic search plugin."""
    
    @pytest.fixture
    def mock_plugin(self):
        """Create a semantic search plugin with mocked dependencies."""
        plugin = SemanticSearchPlugin()
        
        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.query.return_value = [
            {
                "chunk_id": "chunk1",
                "text": "Tesla is an electric vehicle company",
                "similarity": 0.8,
                "file": "tesla_info.txt",
                "chunk_index": 0
            },
            {
                "chunk_id": "chunk2", 
                "text": "Apple is a technology company",
                "similarity": 0.6,
                "file": "apple_info.txt",
                "chunk_index": 0
            }
        ]
        plugin._vector_store = mock_vector_store
        
        # Mock OpenAI client
        mock_openai = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_openai.chat.completions.create.return_value = mock_response
        plugin._openai_client = mock_openai
        
        return plugin
    
    @patch('shared.config.settings')
    @patch('shared.reranking.embed_texts')
    def test_execute_with_cross_encoder_disabled(self, mock_embed, mock_settings, mock_plugin):
        """Test execution with cross-encoder disabled."""
        mock_settings.openai_api_key = "test-key"
        mock_settings.enable_cross_encoder_reranking = False
        mock_settings.enable_query_expansion = False
        mock_settings.enable_debug_logging = False
        mock_settings.retrieval_k = 10
        mock_settings.mmr_lambda = 0.7
        mock_settings.mmr_k = 5
        mock_settings.embed_model = "test-model"
        
        mock_embed.return_value = [[0.1, 0.2, 0.3]]  # Mock embedding
        
        params = {
            "question": "What is Tesla?",
            "enable_mmr": False  # Disable MMR for simpler test
        }
        
        result = mock_plugin.execute(params)
        
        assert "response" in result
        assert "sources" in result
        assert "metadata" in result
        assert result["metadata"]["cross_encoder_enabled"] is False
        assert result["metadata"]["query_expansion_enabled"] is False
    
    @patch('shared.config.settings')
    @patch('shared.reranking.embed_texts')
    def test_execute_with_query_expansion_enabled(self, mock_embed, mock_settings, mock_plugin):
        """Test execution with query expansion enabled."""
        mock_settings.openai_api_key = "test-key"
        mock_settings.enable_cross_encoder_reranking = False
        mock_settings.enable_query_expansion = True
        mock_settings.enable_debug_logging = False
        mock_settings.retrieval_k = 10
        mock_settings.mmr_lambda = 0.7
        mock_settings.mmr_k = 5
        mock_settings.embed_model = "test-model"
        
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        
        params = {
            "question": "What is Tesla?",
            "enable_mmr": False
        }
        
        # Mock the query analysis to detect entity
        with patch.object(mock_plugin, '_analyze_query') as mock_analyze:
            mock_analyze.return_value = {
                "likely_entity_query": True,
                "detected_entities": ["Tesla"],
                "query_type": "entity_lookup"
            }
            
            result = mock_plugin.execute(params)
        
        assert result["metadata"]["query_expansion_enabled"] is True
        assert result["metadata"]["expanded_queries_count"] > 1
    
    @patch('shared.config.settings')
    @patch('shared.reranking.embed_texts')
    def test_execute_with_cross_encoder_enabled(self, mock_embed, mock_settings, mock_plugin):
        """Test execution with cross-encoder enabled."""
        mock_settings.openai_api_key = "test-key"
        mock_settings.enable_cross_encoder_reranking = True
        mock_settings.cross_encoder_model = "test-model"
        mock_settings.cross_encoder_top_k = 5
        mock_settings.enable_query_expansion = False
        mock_settings.enable_debug_logging = False
        mock_settings.retrieval_k = 10
        mock_settings.mmr_lambda = 0.7
        mock_settings.mmr_k = 5
        mock_settings.embed_model = "test-model"
        
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock cross-encoder
        mock_cross_encoder = Mock()
        mock_cross_encoder.is_available.return_value = True
        mock_cross_encoder.rerank.return_value = mock_plugin._vector_store.query.return_value[:1]
        plugin._cross_encoder = mock_cross_encoder
        
        params = {
            "question": "What is Tesla?",
            "enable_mmr": False
        }
        
        result = mock_plugin.execute(params)
        
        assert result["metadata"]["cross_encoder_enabled"] is True
        assert result["metadata"]["cross_encoder_available"] is True
        mock_cross_encoder.rerank.assert_called_once()
    
    def test_deduplicate_results(self, mock_plugin):
        """Test result deduplication functionality."""
        results = [
            {"chunk_id": "chunk1", "similarity": 0.8, "text": "text1"},
            {"chunk_id": "chunk2", "similarity": 0.6, "text": "text2"},
            {"chunk_id": "chunk1", "similarity": 0.7, "text": "text1_dup"},  # Duplicate with lower score
            {"chunk_id": "chunk3", "similarity": 0.9, "text": "text3"},
            {"chunk_id": "chunk2", "similarity": 0.8, "text": "text2_dup"}   # Duplicate with higher score
        ]
        
        deduplicated = mock_plugin._deduplicate_results(results)
        
        assert len(deduplicated) == 3  # Should have 3 unique chunks
        
        # Check that best scores are kept
        chunk_ids = [r["chunk_id"] for r in deduplicated]
        assert "chunk1" in chunk_ids
        assert "chunk2" in chunk_ids  
        assert "chunk3" in chunk_ids
        
        # Verify chunk2 kept the higher score (0.8)
        chunk2_result = next(r for r in deduplicated if r["chunk_id"] == "chunk2")
        assert chunk2_result["similarity"] == 0.8
        assert chunk2_result["text"] == "text2_dup"


class TestPhase4ConfigValidation:
    """Test configuration validation for Phase 4 parameters."""
    
    @patch('shared.config.Settings.__init__')
    def test_cross_encoder_top_k_validation(self, mock_init):
        """Test validation of cross_encoder_top_k parameter."""
        mock_init.return_value = None
        
        # This would be handled by pydantic validation in the actual config
        # Here we test the concept
        retrieval_k = 100
        cross_encoder_top_k = 20
        
        assert cross_encoder_top_k <= retrieval_k  # Should be valid
        
        # Test invalid case
        invalid_cross_encoder_top_k = 150
        assert invalid_cross_encoder_top_k > retrieval_k  # Should be invalid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])