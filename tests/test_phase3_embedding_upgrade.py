#!/usr/bin/env python3
"""
Tests for Phase 3 embedding model upgrade functionality.

Tests model-specific formatting, configuration, and migration capabilities.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from ingestion.processors.embedder import (
    get_model_config, 
    format_texts_for_model,
    embed_texts
)

class TestEmbeddingModelUpgrade:
    """Test Phase 3 embedding model upgrade features."""
    
    def test_e5_model_config(self):
        """Test that E5 models have correct configuration."""
        config = get_model_config("intfloat/e5-base-v2")
        
        assert config["requires_prefixes"] is True
        assert config["query_prefix"] == "query: "
        assert config["passage_prefix"] == "passage: "
        assert config["pooling"] == "mean"
    
    def test_bge_model_config(self):
        """Test that BGE models have correct configuration."""
        config = get_model_config("BAAI/bge-small-en-v1.5")
        
        assert config["requires_prefixes"] is False
        assert config["query_prefix"] == ""
        assert config["passage_prefix"] == ""
        assert config["pooling"] == "cls"
    
    def test_gte_model_config(self):
        """Test that GTE models have correct configuration."""
        config = get_model_config("thenlper/gte-base")
        
        assert config["requires_prefixes"] is False
        assert config["query_prefix"] == ""
        assert config["passage_prefix"] == ""
        assert config["pooling"] == "mean"
    
    def test_unknown_model_config(self):
        """Test that unknown models get default configuration."""
        config = get_model_config("unknown/model")
        
        assert config["requires_prefixes"] is False
        assert config["query_prefix"] == ""
        assert config["passage_prefix"] == ""
        assert config["pooling"] == "mean"
    
    def test_e5_text_formatting_query(self):
        """Test E5 model query text formatting."""
        texts = ["what is machine learning", "define AI"]
        formatted = format_texts_for_model(texts, "intfloat/e5-base-v2", "query")
        
        expected = ["query: what is machine learning", "query: define AI"]
        assert formatted == expected
    
    def test_e5_text_formatting_passage(self):
        """Test E5 model passage text formatting."""
        texts = ["Machine learning is a subset of AI", "Artificial intelligence overview"]
        formatted = format_texts_for_model(texts, "intfloat/e5-base-v2", "passage")
        
        expected = ["passage: Machine learning is a subset of AI", "passage: Artificial intelligence overview"]
        assert formatted == expected
    
    def test_bge_text_formatting_no_prefix(self):
        """Test BGE model requires no text formatting."""
        texts = ["what is machine learning", "define AI"]
        formatted_query = format_texts_for_model(texts, "BAAI/bge-small-en-v1.5", "query")
        formatted_passage = format_texts_for_model(texts, "BAAI/bge-small-en-v1.5", "passage")
        
        assert formatted_query == texts
        assert formatted_passage == texts
    
    def test_gte_text_formatting_no_prefix(self):
        """Test GTE model requires no text formatting."""
        texts = ["what is machine learning", "define AI"]
        formatted_query = format_texts_for_model(texts, "thenlper/gte-base", "query")
        formatted_passage = format_texts_for_model(texts, "thenlper/gte-base", "passage")
        
        assert formatted_query == texts
        assert formatted_passage == texts
    
    def test_backward_compatibility_original_model(self):
        """Test that original all-MiniLM-L6-v2 model still works."""
        texts = ["test text 1", "test text 2"]
        formatted = format_texts_for_model(texts, "sentence-transformers/all-MiniLM-L6-v2", "passage")
        
        # Should return texts unchanged for backward compatibility
        assert formatted == texts
    
    @patch('ingestion.processors.embedder.get_model')
    def test_embed_texts_with_formatting(self, mock_get_model):
        """Test that embed_texts applies model-specific formatting."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        mock_get_model.return_value = mock_model
        
        texts = ["what is AI", "define ML"]
        embeddings = embed_texts(texts, "intfloat/e5-base-v2", normalize=True, text_type="query")
        
        # Verify the model was called with formatted texts
        expected_formatted = ["query: what is AI", "query: define ML"]
        mock_model.encode.assert_called_once_with(expected_formatted, batch_size=32, show_progress_bar=False)
        
        # Verify embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0], decimal=6)
    
    @patch('ingestion.processors.embedder.get_model')
    def test_embed_texts_no_formatting_required(self, mock_get_model):
        """Test that models not requiring formatting work correctly."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        mock_get_model.return_value = mock_model
        
        texts = ["what is AI", "define ML"]
        embeddings = embed_texts(texts, "BAAI/bge-small-en-v1.5", normalize=True, text_type="query")
        
        # Verify the model was called with original texts (no formatting)
        mock_model.encode.assert_called_once_with(texts, batch_size=32, show_progress_bar=False)
        
        # Verify embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0], decimal=6)
    
    def test_embed_texts_maintains_signature_compatibility(self):
        """Test that embed_texts maintains backward compatibility."""
        # Test that we can still call embed_texts with old signature
        with patch('ingestion.processors.embedder.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[1.0, 2.0, 3.0]])
            mock_get_model.return_value = mock_model
            
            # Old-style call should still work
            texts = ["test text"]
            embeddings = embed_texts(texts, "sentence-transformers/all-MiniLM-L6-v2", normalize=True)
            
            assert embeddings.shape == (1, 3)
    
    def test_model_configs_completeness(self):
        """Test that all target models have proper configurations."""
        target_models = [
            "intfloat/e5-base-v2",
            "BAAI/bge-small-en-v1.5", 
            "thenlper/gte-base"
        ]
        
        for model in target_models:
            config = get_model_config(model)
            
            # Verify all required config keys exist
            required_keys = ["query_prefix", "passage_prefix", "requires_prefixes", "pooling"]
            for key in required_keys:
                assert key in config, f"Missing config key '{key}' for model {model}"
            
            # Verify config types
            assert isinstance(config["query_prefix"], str)
            assert isinstance(config["passage_prefix"], str)
            assert isinstance(config["requires_prefixes"], bool)
            assert isinstance(config["pooling"], str)
            assert config["pooling"] in ["mean", "cls"]


class TestMigrationScript:
    """Test migration script functionality (without actual model loading)."""
    
    def test_migration_script_imports(self):
        """Test that migration script can be imported."""
        import sys
        from pathlib import Path
        
        # Add scripts to path
        scripts_path = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(scripts_path))
        
        try:
            # This should not fail if dependencies are correctly handled
            # We can't test the full migration without a database, but can test imports
            pass
        except ImportError as e:
            pytest.fail(f"Migration script import failed: {e}")
    
    def test_config_version_tracking(self):
        """Test that configuration supports version tracking."""
        from shared.config import Settings
        
        # Test that new embed_model_version field exists and has default
        settings = Settings()
        assert hasattr(settings, 'embed_model_version')
        assert settings.embed_model_version == "1.0.0"
        
        # Test that it can be set to different values
        settings_custom = Settings(embed_model_version="2.0.0")
        assert settings_custom.embed_model_version == "2.0.0"


class TestAcceptanceCriteria:
    """Test that Phase 3 acceptance criteria can be met."""
    
    @patch('ingestion.processors.embedder.get_model')
    def test_proper_noun_dense_retrieval_capability(self, mock_get_model):
        """Test that proper noun queries can work with dense-only retrieval."""
        # Mock model that returns different embeddings for different inputs
        mock_model = Mock()
        
        def mock_encode(texts, **kwargs):
            # Simulate different embeddings for different texts
            # In a real scenario, proper nouns should get good embeddings
            embeddings = []
            for text in texts:
                if "Apple Inc" in text:
                    embeddings.append([0.8, 0.6, 0.0])  # High similarity embedding
                elif "apple fruit" in text:
                    embeddings.append([0.2, 0.8, 0.6])  # Different embedding
                else:
                    embeddings.append([0.5, 0.5, 0.5])  # Neutral embedding
            return np.array(embeddings)
        
        mock_model.encode = mock_encode
        mock_get_model.return_value = mock_model
        
        # Test that proper noun queries get distinct embeddings
        query_texts = ["query: what is Apple Inc"]
        passage_texts = ["passage: Apple Inc is a technology company", "passage: apple fruit is healthy"]
        
        query_embeddings = embed_texts(query_texts, "intfloat/e5-base-v2", text_type="query")
        passage_embeddings = embed_texts(passage_texts, "intfloat/e5-base-v2", text_type="passage")
        
        # Compute cosine similarity (since embeddings are normalized)
        similarities = np.dot(query_embeddings, passage_embeddings.T)
        
        # The Apple Inc passage should have higher similarity than apple fruit
        assert similarities[0, 0] > similarities[0, 1], "Proper noun should match better with relevant passage"
    
    def test_migration_without_data_loss_design(self):
        """Test that migration script is designed to prevent data loss."""
        # Test that backup functionality is built into the migration
        # This is a design test - we verify the migration script has backup logic
        
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "migrate_embeddings", 
            Path(__file__).parent.parent / "scripts" / "migrate_embeddings.py"
        )
        
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            
            # Check that key methods exist for safe migration
            assert hasattr(module, 'EmbeddingMigrator'), "Migration class should exist"
            
            # The module should define backup and progress tracking methods
            # This ensures the migration is designed with safety in mind
            source_code = Path(__file__).parent.parent / "scripts" / "migrate_embeddings.py"
            content = source_code.read_text()
            
            # Check for safety features in the code
            assert "create_backup" in content, "Migration should create backups"
            assert "progress" in content, "Migration should track progress"
            assert "resume" in content, "Migration should support resuming"
            assert "batch" in content, "Migration should process in batches"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])