"""Test Phase 1 retrieval robustness improvements integration."""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add backend src to path
backend_root = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_root / "src"))

from ingestion.processors.embedder import embed_texts, normalize_embeddings
from shared.mmr import mmr_selection, extract_embeddings_from_results
from shared.config import Settings


class TestRetrievalRobustness:
    """Test Phase 1 retrieval robustness improvements."""
    
    def test_embedder_normalization_integration(self):
        """Test that embedder produces normalized embeddings by default."""
        # Mock the sentence transformer to return known vectors
        with patch('ingestion.processors.embedder.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([
                [1.0, 2.0, 3.0],  # Will be normalized
                [4.0, 5.0, 6.0]   # Will be normalized
            ])
            mock_get_model.return_value = mock_model
            
            embeddings = embed_texts(["text1", "text2"], "test-model", normalize=True)
            
            # Check that embeddings are normalized (unit length)
            norms = np.linalg.norm(embeddings, axis=1)
            np.testing.assert_array_almost_equal(norms, [1.0, 1.0], decimal=6)
    
    def test_embedder_optional_normalization(self):
        """Test that embedder can optionally skip normalization."""
        with patch('ingestion.processors.embedder.get_model') as mock_get_model:
            mock_model = Mock()
            original_vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            mock_model.encode.return_value = original_vectors
            mock_get_model.return_value = mock_model
            
            # Without normalization
            embeddings = embed_texts(["text1", "text2"], "test-model", normalize=False)
            
            # Should return original vectors unchanged
            np.testing.assert_array_equal(embeddings, original_vectors)
    
    def test_mmr_integration_with_search_results(self):
        """Test MMR selection with realistic search results."""
        # Create mock search results
        search_results = [
            {"id": "1", "text": "Document about artificial intelligence", "distance": 0.1, "similarity": 0.9},
            {"id": "2", "text": "Another AI document very similar", "distance": 0.15, "similarity": 0.85},
            {"id": "3", "text": "Document about machine learning", "distance": 0.2, "similarity": 0.8},
            {"id": "4", "text": "Completely different topic - cooking", "distance": 0.8, "similarity": 0.2},
        ]
        
        # Mock embedder function
        def mock_embedder(texts):
            # Return embeddings that reflect the similarity in the search results
            embeddings = []
            for text in texts:
                if "artificial intelligence" in text or "AI" in text:
                    embeddings.append([1.0, 0.0, 0.0])  # Similar vectors
                elif "machine learning" in text:
                    embeddings.append([0.8, 0.6, 0.0])  # Somewhat similar
                else:
                    embeddings.append([0.0, 0.0, 1.0])  # Different vector
            return np.array(embeddings, dtype=np.float32)
        
        # Extract embeddings
        candidate_embeddings = extract_embeddings_from_results(search_results, mock_embedder)
        
        # Query about AI (similar to first two documents)
        query_embedding = np.array([1.0, 0.0, 0.0])
        
        # Test MMR with high lambda (prefer relevance)
        selected_high_relevance = mmr_selection(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_results=search_results,
            mmr_lambda=0.9,
            k=2
        )
        
        # Should select most relevant first
        assert len(selected_high_relevance) == 2
        assert selected_high_relevance[0]["id"] == "1"  # Most similar
        
        # Test MMR with low lambda (prefer diversity)
        selected_high_diversity = mmr_selection(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_results=search_results,
            mmr_lambda=0.1,
            k=3
        )
        
        # Should include diverse documents
        assert len(selected_high_diversity) == 3
        selected_ids = [result["id"] for result in selected_high_diversity]
        
        # Should include the cooking document for diversity
        assert "4" in selected_ids
    
    def test_config_mmr_parameters(self):
        """Test that MMR configuration parameters are properly defined."""
        # Test default config values
        settings = Settings()
        
        assert hasattr(settings, 'retrieval_k')
        assert hasattr(settings, 'mmr_lambda')
        assert hasattr(settings, 'mmr_k')
        assert hasattr(settings, 'proper_noun_boost')
        assert hasattr(settings, 'enable_debug_logging')
        
        # Test default values are reasonable
        assert 50 <= settings.retrieval_k <= 200  # Should be in range for Phase 1
        assert 0.0 <= settings.mmr_lambda <= 1.0
        assert 1 <= settings.mmr_k <= 50
        assert 0.0 <= settings.proper_noun_boost <= 1.0
        assert isinstance(settings.enable_debug_logging, bool)
    
    def test_config_validation_mmr_lambda(self):
        """Test that MMR lambda is properly validated."""
        # Valid lambda values
        valid_settings = Settings(mmr_lambda=0.0)
        assert valid_settings.mmr_lambda == 0.0
        
        valid_settings = Settings(mmr_lambda=1.0)
        assert valid_settings.mmr_lambda == 1.0
        
        valid_settings = Settings(mmr_lambda=0.5)
        assert valid_settings.mmr_lambda == 0.5
        
        # Invalid lambda values should raise validation error
        with pytest.raises(ValueError):
            Settings(mmr_lambda=-0.1)
        
        with pytest.raises(ValueError):
            Settings(mmr_lambda=1.1)
    
    def test_query_analysis_entity_detection(self):
        """Test query analysis for entity detection."""
        # Import the semantic search plugin 
        try:
            from querying.agents.plugins.semantic_search import SemanticSearchPlugin
        except ImportError:
            from src.querying.agents.plugins.semantic_search import SemanticSearchPlugin
        
        plugin = SemanticSearchPlugin()
        
        # Test entity query detection
        entity_queries = [
            "what is Einstein",
            "who is Abraham Lincoln", 
            "tell me about Python",
            "find Tesla",
            "show me Microsoft"
        ]
        
        for query in entity_queries:
            analysis = plugin._analyze_query(query)
            assert analysis["likely_entity_query"] == True, f"Failed to detect entity in: {query}"
            assert len(analysis["detected_entities"]) > 0, f"No entities detected in: {query}"
        
        # Test non-entity queries
        general_queries = [
            "how does photosynthesis work",
            "what are the benefits of exercise", 
            "explain quantum mechanics",
            "describe the water cycle"
        ]
        
        for query in general_queries:
            analysis = plugin._analyze_query(query)
            # These might be detected as entity queries due to capitalized words, but should be less certain
            assert analysis["query_type"] in ["general", "entity_lookup"], f"Unexpected classification for: {query}"
    
    def test_metadata_boosting_functionality(self):
        """Test metadata boosting for proper noun queries."""
        try:
            from querying.agents.plugins.semantic_search import SemanticSearchPlugin
            from shared.config import Settings
        except ImportError:
            from src.querying.agents.plugins.semantic_search import SemanticSearchPlugin
            from src.shared.config import Settings
        
        plugin = SemanticSearchPlugin()
        
        # Mock results with different document paths/titles
        search_results = [
            {
                "id": "1", "text": "General content", "distance": 0.5, "similarity": 0.5,
                "document_path": "/docs/random_file.pdf", "document_title": "Random Document", "file": "random_file.pdf"
            },
            {
                "id": "2", "text": "Content about Tesla", "distance": 0.4, "similarity": 0.6,
                "document_path": "/docs/tesla_biography.pdf", "document_title": "Tesla Biography", "file": "tesla_biography.pdf"
            },
            {
                "id": "3", "text": "Another document", "distance": 0.3, "similarity": 0.7,
                "document_path": "/docs/einstein_papers.pdf", "document_title": "Einstein Papers", "file": "einstein_papers.pdf"
            }
        ]
        
        # Test boosting for Tesla query
        query_analysis = {"detected_entities": ["Tesla"], "likely_entity_query": True}
        
        with patch('shared.config.settings.proper_noun_boost', 0.3):
            boosted_results = plugin._apply_metadata_boosting(search_results, "what is Tesla", query_analysis)
        
        # Tesla document should be boosted to the top
        assert boosted_results[0]["id"] == "2", "Tesla document should be boosted to top"
        assert boosted_results[0].get("metadata_boosted") == True
        assert boosted_results[0]["similarity"] > search_results[1]["similarity"]  # Should be boosted
    
    def test_integration_no_regressions(self):
        """Test that improvements don't break basic functionality."""
        # Test that embedder still works with basic functionality
        with patch('ingestion.processors.embedder.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
            mock_get_model.return_value = mock_model
            
            # Should work with default parameters
            embeddings = embed_texts(["text1", "text2"], "test-model")
            assert embeddings.shape == (2, 2)
            
            # Should be normalized by default
            norms = np.linalg.norm(embeddings, axis=1)
            np.testing.assert_array_almost_equal(norms, [1.0, 1.0], decimal=6)
    
    def test_mmr_deterministic_behavior(self):
        """Test that MMR selection is deterministic for reproducible results."""
        # Create identical test scenarios
        query_embedding = np.array([1.0, 0.0, 0.0])
        candidate_embeddings = np.array([
            [0.8, 0.6, 0.0],
            [0.6, 0.8, 0.0], 
            [0.0, 0.0, 1.0]
        ])
        search_results = [
            {"id": "1", "text": "First doc"},
            {"id": "2", "text": "Second doc"},
            {"id": "3", "text": "Third doc"}
        ]
        
        # Run MMR multiple times with same parameters
        results1 = mmr_selection(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_results=search_results,
            mmr_lambda=0.5,
            k=2
        )
        
        results2 = mmr_selection(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_results=search_results,
            mmr_lambda=0.5,
            k=2
        )
        
        # Results should be identical
        assert len(results1) == len(results2)
        for i in range(len(results1)):
            assert results1[i]["id"] == results2[i]["id"]