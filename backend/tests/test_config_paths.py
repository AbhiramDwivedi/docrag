#!/usr/bin/env python3
"""Test configuration path management functionality."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shared.config import Settings, load_settings


class TestConfigPathManagement:
    """Test cases for configuration path management."""
    
    def test_default_knowledge_graph_path(self):
        """Test that default knowledge_graph_path is set correctly."""
        settings = Settings()
        assert settings.knowledge_graph_path == Path("data/knowledge_graph.db")
    
    def test_custom_knowledge_graph_path_from_dict(self):
        """Test setting custom knowledge_graph_path from config dict."""
        config_data = {
            "knowledge_graph_path": "custom/path/kg.db"
        }
        settings = Settings(**config_data)
        assert settings.knowledge_graph_path == Path("custom/path/kg.db")
    
    def test_resolve_storage_path_relative(self, tmp_path):
        """Test path resolution for relative paths."""
        settings = Settings()
        
        # Use tmp_path as the working directory context
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            relative_path = Path("data/test.db")
            resolved = settings.resolve_storage_path(relative_path)
            
            # Should be resolved relative to cwd
            assert resolved == tmp_path / "data/test.db"
            # Directory should be created
            assert resolved.parent.exists()
    
    def test_resolve_storage_path_absolute(self, tmp_path):
        """Test path resolution for absolute paths."""
        settings = Settings()
        
        absolute_path = tmp_path / "absolute" / "test.db"
        resolved = settings.resolve_storage_path(absolute_path)
        
        # Should return the same absolute path
        assert resolved == absolute_path
        # Directory should be created
        assert resolved.parent.exists()
    
    def test_load_settings_with_knowledge_graph_path(self, tmp_path):
        """Test loading settings from YAML with knowledge_graph_path."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "sync_root": "~/test_docs",
            "db_path": "test_data/docmeta.db",
            "vector_path": "test_data/vector.index",
            "knowledge_graph_path": "test_data/kg.db",
            "embed_model": "test-model"
        }
        
        config_file.write_text(yaml.dump(config_data))
        
        # Mock the CONFIG_PATH to point to our test file
        with patch('shared.config.CONFIG_PATH', config_file):
            settings = load_settings()
            
            assert settings.knowledge_graph_path == Path("test_data/kg.db")
            assert settings.db_path == Path("test_data/docmeta.db")
            assert settings.vector_path == Path("test_data/vector.index")
    
    def test_config_template_includes_knowledge_graph_path(self):
        """Test that config.yaml.template includes knowledge_graph_path field."""
        template_path = Path(__file__).parent.parent / "src/shared/config.yaml.template"
        
        # Read and parse the template
        template_content = template_path.read_text()
        config_data = yaml.safe_load(template_content)
        
        # Check that knowledge_graph_path is present
        assert "knowledge_graph_path" in config_data
        assert config_data["knowledge_graph_path"] == "data/knowledge_graph.db"
    
    def test_backwards_compatibility(self):
        """Test that existing configs without knowledge_graph_path still work."""
        # Old config without knowledge_graph_path
        old_config = {
            "sync_root": "~/Documents",
            "db_path": "data/docmeta.db",
            "vector_path": "data/vector.index"
        }
        
        settings = Settings(**old_config)
        
        # Should use default value
        assert settings.knowledge_graph_path == Path("data/knowledge_graph.db")
        assert settings.db_path == Path("data/docmeta.db")
        assert settings.vector_path == Path("data/vector.index")
    
    def test_utils_get_default_paths_uses_config(self):
        """Test that get_default_paths uses configuration instead of hardcoded values."""
        from shared.utils import get_default_paths
        
        # Create custom settings
        custom_settings = Settings(
            knowledge_graph_path=Path("custom/kg.db"),
            db_path=Path("custom/db.db"),
            vector_path=Path("custom/vector.index")
        )
        
        with patch('shared.config.get_settings', return_value=custom_settings):
            paths = get_default_paths()
            
            # Normalize paths for cross-platform comparison
            assert Path(paths["knowledge_graph_path"]) == Path("custom/kg.db")
            assert Path(paths["db_path"]) == Path("custom/db.db") 
            assert Path(paths["vector_path"]) == Path("custom/vector.index")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])