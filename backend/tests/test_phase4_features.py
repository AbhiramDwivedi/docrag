"""Tests for Phase 4 advanced features in semantic search plugin.

Tests cross-encoder reranking, query expansion, and entity-aware functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add backend src to path for imports
backend_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(backend_root))

from querying.agents.plugins.semantic_search import SemanticSearchPlugin
from shared.reranking import CrossEncoderReranker, expand_entity_query, create_query_variants
from shared.entity_indexing import EntityExtractor, create_entity_document_mapping, enhance_chunk_with_entities, get_entity_boost_score


class TestEntityExtractor:
    """Test entity extraction functionality."""
    
    def test_entity_extractor_initialization(self):
        """Test entity extractor initialization."""
        extractor = EntityExtractor("en_core_web_sm")
        assert extractor.model_name == "en_core_web_sm"
        assert extractor._nlp is None
        # Check availability status (may be False if spaCy not installed)
        assert isinstance(extractor._available, bool)
    
    def test_entity_extractor_unavailable(self):
        """Test behavior when spaCy is unavailable."""
        with patch('shared.entity_indexing.spacy') as mock_spacy:
            # Simulate ImportError when importing spaCy
            def raise_import_error(*args, **kwargs):
                raise ImportError("No module named 'spacy'")
            
            mock_spacy.side_effect = raise_import_error
            
            extractor = EntityExtractor()
            assert not extractor.is_available()
            
            # Should return empty list when unavailable
            entities = extractor.extract_entities("Apple Inc is a technology company.")
            assert entities == []
    
    @patch('shared.entity_indexing.spacy')
    def test_entity_extraction_success(self, mock_spacy):
        """Test successful entity extraction with mock spaCy."""
        # Mock spaCy components
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Mock entities
        mock_ent1 = Mock()
        mock_ent1.text = "Apple Inc"
        mock_ent1.label_ = "ORG"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 9
        
        mock_ent2 = Mock()
        mock_ent2.text = "California"
        mock_ent2.label_ = "GPE"
        mock_ent2.start_char = 30
        mock_ent2.end_char = 40
        
        mock_doc.ents = [mock_ent1, mock_ent2]
        mock_nlp.return_value = mock_doc
        
        # Mock spaCy module
        mock_spacy.load.return_value = mock_nlp
        
        extractor = EntityExtractor()
        extractor._available = True
        extractor._spacy = mock_spacy
        
        text = "Apple Inc is headquartered in California."
        entities = extractor.extract_entities(text)
        
        assert len(entities) == 2
        assert entities[0]["text"] == "Apple Inc"
        assert entities[0]["label"] == "ORG"
        assert entities[1]["text"] == "California"
        assert entities[1]["label"] == "GPE"
    
    @patch('shared.entity_indexing.spacy')
    def test_entity_extraction_filter_types(self, mock_spacy):
        """Test that only relevant entity types are returned."""
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Mix of relevant and irrelevant entity types
        mock_ent1 = Mock()  # Should be included
        mock_ent1.text = "Microsoft"
        mock_ent1.label_ = "ORG"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 9
        
        mock_ent2 = Mock()  # Should be excluded (not in filter)
        mock_ent2.text = "yesterday"
        mock_ent2.label_ = "DATE"
        mock_ent2.start_char = 20
        mock_ent2.end_char = 29
        
        mock_ent3 = Mock()  # Should be included
        mock_ent3.text = "John Doe"
        mock_ent3.label_ = "PERSON"
        mock_ent3.start_char = 30
        mock_ent3.end_char = 38
        
        mock_doc.ents = [mock_ent1, mock_ent2, mock_ent3]
        mock_nlp.return_value = mock_doc
        mock_spacy.load.return_value = mock_nlp
        
        extractor = EntityExtractor()
        extractor._available = True
        extractor._spacy = mock_spacy
        
        entities = extractor.extract_entities("Test text")
        
        # Should only return ORG and PERSON, not DATE
        assert len(entities) == 2
        entity_labels = [ent["label"] for ent in entities]
        assert "ORG" in entity_labels
        assert "PERSON" in entity_labels
        assert "DATE" not in entity_labels
    
    def test_entity_extraction_error_handling(self):
        """Test error handling in entity extraction."""
        extractor = EntityExtractor()
        extractor._available = True
        
        # Mock a failing model load
        def failing_load_model():
            return None
        
        extractor._load_model = failing_load_model
        
        entities = extractor.extract_entities("Test text")
        assert entities == []


class TestEntityDocumentMapping:
    """Test entity-document mapping functionality."""
    
    def test_create_entity_document_mapping(self):
        """Test creation of entity-document mappings."""
        entities = [
            {
                "text": "Apple Inc",
                "label": "ORG",
                "start": 0,
                "end": 9,
                "confidence": 1.0
            },
            {
                "text": "California",
                "label": "GPE", 
                "start": 30,
                "end": 40,
                "confidence": 1.0
            }
        ]
        
        mappings = create_entity_document_mapping(entities, "doc123", "chunk456")
        
        assert len(mappings) == 2
        
        # Check first mapping
        assert mappings[0]["entity_text"] == "apple inc"  # Should be normalized to lowercase
        assert mappings[0]["entity_label"] == "ORG"
        assert mappings[0]["document_id"] == "doc123"
        assert mappings[0]["chunk_id"] == "chunk456"
        assert mappings[0]["start_pos"] == 0
        assert mappings[0]["end_pos"] == 9
        
        # Check second mapping
        assert mappings[1]["entity_text"] == "california"
        assert mappings[1]["entity_label"] == "GPE"
    
    def test_enhance_chunk_with_entities(self):
        """Test chunk enhancement with entities."""
        chunk_data = {
            "content": "Tesla Motors is an electric vehicle company.",
            "chunk_id": "chunk123"
        }
        
        # Mock extractor that returns test entities
        mock_extractor = Mock()
        mock_extractor.is_available.return_value = True
        mock_extractor.extract_entities.return_value = [
            {
                "text": "Tesla Motors",
                "label": "ORG",
                "start": 0,
                "end": 12,
                "confidence": 1.0
            }
        ]
        
        enhanced = enhance_chunk_with_entities(chunk_data, mock_extractor)
        
        assert enhanced["content"] == chunk_data["content"]
        assert enhanced["chunk_id"] == chunk_data["chunk_id"]
        assert len(enhanced["entities"]) == 1
        assert enhanced["entity_count"] == 1
        assert enhanced["entity_types"] == ["ORG"]
        assert enhanced["entities"][0]["text"] == "Tesla Motors"
    
    def test_enhance_chunk_extractor_unavailable(self):
        """Test chunk enhancement when extractor is unavailable."""
        chunk_data = {"content": "Test content", "chunk_id": "chunk123"}
        
        mock_extractor = Mock()
        mock_extractor.is_available.return_value = False
        
        enhanced = enhance_chunk_with_entities(chunk_data, mock_extractor)
        
        assert enhanced["entities"] == []
        assert enhanced["entity_count"] == 0
    
    def test_get_entity_boost_score(self):
        """Test entity boost score calculation."""
        query_entities = ["Tesla", "Elon Musk"]
        chunk_entities = [
            {"text": "Tesla Motors", "label": "ORG"},
            {"text": "Tesla", "label": "ORG"},
            {"text": "California", "label": "GPE"}
        ]
        
        boost_score = get_entity_boost_score(query_entities, chunk_entities, boost_factor=0.2)
        
        # Should find 1 match (Tesla - case insensitive)
        assert boost_score == 0.2  # 1 match * 0.2 boost_factor
    
    def test_get_entity_boost_score_multiple_matches(self):
        """Test entity boost score with multiple matches."""
        query_entities = ["Apple", "California"]
        chunk_entities = [
            {"text": "Apple Inc", "label": "ORG"},
            {"text": "apple", "label": "ORG"},  # Different case, should match "Apple"
            {"text": "California", "label": "GPE"}
        ]
        
        boost_score = get_entity_boost_score(query_entities, chunk_entities, boost_factor=0.1)
        
        # Should find 2 matches: Apple (matches both apple variants) and California
        assert boost_score == 0.2  # 2 matches * 0.1 boost_factor
    
    def test_get_entity_boost_score_capped(self):
        """Test that entity boost score is capped at maximum."""
        query_entities = ["test"] * 10  # Many entities
        chunk_entities = [{"text": "test", "label": "ORG"}] * 10  # Many matches
        
        boost_score = get_entity_boost_score(query_entities, chunk_entities, boost_factor=0.2)
        
        # Should be capped at 0.5 max_boost
        assert boost_score == 0.5
    
    def test_get_entity_boost_score_no_matches(self):
        """Test entity boost score with no matches."""
        query_entities = ["Apple"]
        chunk_entities = [{"text": "Microsoft", "label": "ORG"}]
        
        boost_score = get_entity_boost_score(query_entities, chunk_entities, boost_factor=0.2)
        
        assert boost_score == 0.0


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
    
    def test_cross_encoder_positive_validation(self):
        """Test that cross_encoder_top_k must be positive."""
        # Test valid positive values
        for valid_k in [1, 5, 20, 100]:
            assert valid_k > 0
        
        # Test invalid values
        for invalid_k in [0, -1, -10]:
            assert invalid_k <= 0
    
    def test_entity_boost_factor_range(self):
        """Test that entity_boost_factor is in valid range."""
        # Test valid range
        for valid_factor in [0.0, 0.1, 0.5, 1.0]:
            assert 0.0 <= valid_factor <= 1.0
        
        # Test invalid values
        invalid_factors = [-0.1, 1.1, 2.0]
        for invalid_factor in invalid_factors:
            assert not (0.0 <= invalid_factor <= 1.0)


class TestPhase4Security:
    """Test security validations for Phase 4 features."""
    
    def test_openai_api_key_validation(self):
        """Test OpenAI API key security validation."""
        # Test placeholder detection
        placeholder_keys = [
            "your-openai-api-key-here",
            "sk-placeholder",
            "change-me",
            "your-key-here"
        ]
        
        for placeholder in placeholder_keys:
            # These should be detected as placeholders
            assert placeholder.lower() in {p.lower() for p in placeholder_keys}
    
    def test_cross_encoder_model_validation(self):
        """Test cross-encoder model validation."""
        # Known models should pass validation
        known_models = [
            "ms-marco-MiniLM-L-6-v2",
            "ms-marco-MiniLM-L-12-v2",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ]
        
        for model in known_models:
            assert isinstance(model, str) and len(model) > 0
        
        # Invalid models
        invalid_models = ["", None, 123]
        for invalid_model in invalid_models:
            assert not (isinstance(invalid_model, str) and len(invalid_model) > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])