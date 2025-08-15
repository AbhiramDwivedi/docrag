"""Tests for enhanced model version consistency validation."""
import pytest
from backend.src.shared.config import Settings
from pydantic import ValidationError


class TestModelVersionValidation:
    """Test comprehensive model version validation."""
    
    def test_valid_semantic_versioning(self):
        """Test that valid semantic versions are accepted."""
        # Should not raise
        settings = Settings(
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
            embed_model_version="1.0.0"
        )
        assert settings.embed_model_version == "1.0.0"
    
    def test_invalid_semantic_versioning_format(self):
        """Test that invalid version formats are rejected."""
        with pytest.raises(ValidationError, match="must follow semantic versioning format"):
            Settings(
                embed_model="sentence-transformers/all-MiniLM-L6-v2",
                embed_model_version="1.0"  # Missing patch version
            )
        
        with pytest.raises(ValidationError, match="must follow semantic versioning format"):
            Settings(
                embed_model="sentence-transformers/all-MiniLM-L6-v2",
                embed_model_version="v1.0.0"  # Has 'v' prefix
            )
        
        with pytest.raises(ValidationError, match="must follow semantic versioning format"):
            Settings(
                embed_model="sentence-transformers/all-MiniLM-L6-v2",
                embed_model_version="1.0.0-beta"  # Has pre-release tag
            )
    
    def test_empty_model_name_validation(self):
        """Test that empty model names are rejected."""
        with pytest.raises(ValidationError, match="embed_model must be a non-empty string"):
            Settings(
                embed_model="",
                embed_model_version="1.0.0"
            )
    
    def test_known_model_version_consistency(self):
        """Test version consistency for known models."""
        # Correct version for MiniLM model
        settings = Settings(
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
            embed_model_version="1.0.0"
        )
        assert settings.embed_model_version == "1.0.0"
        
        # Correct version for E5 model
        settings = Settings(
            embed_model="intfloat/e5-base-v2",
            embed_model_version="2.0.0"
        )
        assert settings.embed_model_version == "2.0.0"
    
    def test_version_mismatch_warning_logged(self, caplog):
        """Test that version mismatches generate warnings but don't fail."""
        import logging
        caplog.set_level(logging.WARNING)
        
        # Wrong version for MiniLM model - should warn but not fail
        settings = Settings(
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
            embed_model_version="2.0.0"  # Should be 1.0.0
        )
        
        assert settings.embed_model_version == "2.0.0"  # Value accepted
        assert "may not be compatible with model" in caplog.text
        assert "Expected versions: ['1.0.0']" in caplog.text
        assert "compatibility issues with embeddings and migration" in caplog.text
    
    def test_unknown_model_info_logged(self, caplog):
        """Test that unknown models generate info messages."""
        import logging
        caplog.set_level(logging.INFO)
        
        settings = Settings(
            embed_model="custom/unknown-model",
            embed_model_version="3.0.0"
        )
        
        assert settings.embed_model_version == "3.0.0"
        assert "Unknown embedding model 'custom/unknown-model'" in caplog.text
        assert "Ensure compatibility manually" in caplog.text
    
    def test_version_only_validation_allowed(self):
        """Test that version can be validated independently when model not set."""
        # This should work - version format validation only
        settings = Settings(embed_model_version="1.0.0")
        assert settings.embed_model_version == "1.0.0"
        
        # This should fail - invalid format
        with pytest.raises(ValidationError, match="must follow semantic versioning format"):
            Settings(embed_model_version="invalid")
    
    def test_all_known_models_covered(self):
        """Test that all Phase 3 models have proper version mappings."""
        phase_3_models = [
            "intfloat/e5-base-v2",
            "intfloat/e5-small-v2", 
            "intfloat/e5-large-v2",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-large-en-v1.5",
            "thenlper/gte-base",
            "thenlper/gte-small",
            "thenlper/gte-large"
        ]
        
        # All should work with version 2.0.0 without warnings
        for model in phase_3_models:
            settings = Settings(
                embed_model=model,
                embed_model_version="2.0.0"
            )
            assert settings.embed_model == model
            assert settings.embed_model_version == "2.0.0"
    
    def test_legacy_model_compatibility(self):
        """Test that legacy MiniLM model still works."""
        settings = Settings(
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
            embed_model_version="1.0.0"
        )
        assert settings.embed_model_version == "1.0.0"
        
        # Wrong version should warn but not fail - we test this with caplog above
