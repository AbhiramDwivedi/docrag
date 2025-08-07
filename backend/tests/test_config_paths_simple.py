#!/usr/bin/env python3
"""Test configuration path management functionality - simplified version."""

import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shared.config import Settings, load_settings


def test_default_knowledge_graph_path():
    """Test that default knowledge_graph_path is set correctly."""
    settings = Settings()
    assert settings.knowledge_graph_path == Path("data/knowledge_graph.db")
    print("✅ Default knowledge_graph_path test passed")


def test_custom_knowledge_graph_path_from_dict():
    """Test setting custom knowledge_graph_path from config dict."""
    config_data = {
        "knowledge_graph_path": "custom/path/kg.db"
    }
    settings = Settings(**config_data)
    assert settings.knowledge_graph_path == Path("custom/path/kg.db")
    print("✅ Custom knowledge_graph_path test passed")


def test_resolve_storage_path_relative():
    """Test path resolution for relative paths."""
    settings = Settings()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Use tmp_path as the working directory context
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            relative_path = Path("data/test.db")
            resolved = settings.resolve_storage_path(relative_path)
            
            # Should be resolved relative to cwd
            assert resolved == tmp_path / "data/test.db"
            # Directory should be created
            assert resolved.parent.exists()
    
    print("✅ Relative path resolution test passed")


def test_resolve_storage_path_absolute():
    """Test path resolution for absolute paths."""
    settings = Settings()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        absolute_path = tmp_path / "absolute" / "test.db"
        resolved = settings.resolve_storage_path(absolute_path)
        
        # Should return the same absolute path
        assert resolved == absolute_path
        # Directory should be created
        assert resolved.parent.exists()
    
    print("✅ Absolute path resolution test passed")


def test_config_template_includes_knowledge_graph_path():
    """Test that config.yaml.template includes knowledge_graph_path field."""
    template_path = Path(__file__).parent.parent / "src/shared/config.yaml.template"
    
    # Read and parse the template
    template_content = template_path.read_text()
    config_data = yaml.safe_load(template_content)
    
    # Check that knowledge_graph_path is present
    assert "knowledge_graph_path" in config_data
    assert config_data["knowledge_graph_path"] == "data/knowledge_graph.db"
    
    print("✅ Config template includes knowledge_graph_path test passed")


def test_backwards_compatibility():
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
    
    print("✅ Backwards compatibility test passed")


def test_utils_get_default_paths_uses_config():
    """Test that get_default_paths uses configuration instead of hardcoded values."""
    from shared.utils import get_default_paths
    
    # Create custom settings
    custom_settings = Settings(
        knowledge_graph_path=Path("custom/kg.db"),
        db_path=Path("custom/db.db"),
        vector_path=Path("custom/vector.index")
    )
    
    with patch('shared.utils.get_settings', return_value=custom_settings):
        paths = get_default_paths()
        
        assert paths["knowledge_graph_path"] == "custom/kg.db"
        assert paths["db_path"] == "custom/db.db"
        assert paths["vector_path"] == "custom/vector.index"
    
    print("✅ Utils get_default_paths uses config test passed")


if __name__ == "__main__":
    print("=== Running Configuration Path Management Tests ===")
    
    try:
        test_default_knowledge_graph_path()
        test_custom_knowledge_graph_path_from_dict()
        test_resolve_storage_path_relative()
        test_resolve_storage_path_absolute()
        test_config_template_includes_knowledge_graph_path()
        test_backwards_compatibility()
        test_utils_get_default_paths_uses_config()
        
        print("\n🎉 All tests passed! Configuration path management is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)