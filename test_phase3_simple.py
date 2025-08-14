#!/usr/bin/env python3
"""
Simple test for Phase 3 embedding functionality without external dependencies.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

def test_model_configs():
    """Test model configurations directly."""
    # Import only the specific functions we need
    from ingestion.processors.embedder import get_model_config, format_texts_for_model
    
    print("Testing model configurations...")
    
    # Test E5 model
    e5_config = get_model_config("intfloat/e5-base-v2")
    assert e5_config["requires_prefixes"] is True
    assert e5_config["query_prefix"] == "query: "
    assert e5_config["passage_prefix"] == "passage: "
    print("✅ E5 model config correct")
    
    # Test BGE model  
    bge_config = get_model_config("BAAI/bge-small-en-v1.5")
    assert bge_config["requires_prefixes"] is False
    assert bge_config["query_prefix"] == ""
    assert bge_config["passage_prefix"] == ""
    print("✅ BGE model config correct")
    
    # Test GTE model
    gte_config = get_model_config("thenlper/gte-base")
    assert gte_config["requires_prefixes"] is False
    print("✅ GTE model config correct")
    
    # Test text formatting
    texts = ["what is AI", "define machine learning"]
    
    # E5 should add prefixes
    e5_formatted = format_texts_for_model(texts, "intfloat/e5-base-v2", "query")
    expected_e5 = ["query: what is AI", "query: define machine learning"]
    assert e5_formatted == expected_e5
    print("✅ E5 text formatting correct")
    
    # BGE should not modify
    bge_formatted = format_texts_for_model(texts, "BAAI/bge-small-en-v1.5", "query") 
    assert bge_formatted == texts
    print("✅ BGE text formatting correct")
    
    print("All model configuration tests passed!")

def test_config_version_tracking():
    """Test configuration version tracking."""
    from shared.config import Settings
    
    # Test default version
    settings = Settings()
    assert hasattr(settings, 'embed_model_version')
    assert settings.embed_model_version == "1.0.0"
    print("✅ Default version tracking works")
    
    # Test custom version
    custom_settings = Settings(embed_model_version="2.0.0")
    assert custom_settings.embed_model_version == "2.0.0" 
    print("✅ Custom version tracking works")

def test_migration_script_structure():
    """Test that migration script has proper structure."""
    migration_script = Path(__file__).parent.parent / "scripts" / "migrate_embeddings.py"
    assert migration_script.exists()
    print("✅ Migration script exists")
    
    content = migration_script.read_text()
    
    # Check for key functionality
    required_features = [
        "create_backup",
        "progress", 
        "resume",
        "EmbeddingMigrator",
        "batch_size"
    ]
    
    for feature in required_features:
        assert feature in content, f"Missing feature: {feature}"
        print(f"✅ Migration script has {feature}")

if __name__ == "__main__":
    try:
        print("🧪 Testing Phase 3 embedding upgrade functionality...\n")
        
        test_model_configs()
        print()
        
        test_config_version_tracking()
        print()
        
        test_migration_script_structure()
        print()
        
        print("🎉 All Phase 3 tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)