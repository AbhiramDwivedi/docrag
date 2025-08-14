#!/usr/bin/env python3
"""
Standalone test of Phase 3 functionality
"""

from typing import List, Dict, Any

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model-specific configuration for text formatting and processing."""
    # Model-specific configurations
    model_configs = {
        # E5 models require specific prefixes for queries and passages
        "intfloat/e5-base-v2": {
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "requires_prefixes": True,
            "pooling": "mean"
        },
        "intfloat/e5-small-v2": {
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "requires_prefixes": True,
            "pooling": "mean"
        },
        # BGE models work well with no prefixes
        "BAAI/bge-small-en-v1.5": {
            "query_prefix": "",
            "passage_prefix": "",
            "requires_prefixes": False,
            "pooling": "cls"
        },
        "BAAI/bge-base-en-v1.5": {
            "query_prefix": "",
            "passage_prefix": "",
            "requires_prefixes": False,
            "pooling": "cls"
        },
        # GTE models work well without prefixes
        "thenlper/gte-base": {
            "query_prefix": "",
            "passage_prefix": "",
            "requires_prefixes": False,
            "pooling": "mean"
        },
        "thenlper/gte-small": {
            "query_prefix": "",
            "passage_prefix": "",
            "requires_prefixes": False,
            "pooling": "mean"
        }
    }
    
    # Default configuration for unknown models (backward compatibility)
    default_config = {
        "query_prefix": "",
        "passage_prefix": "",
        "requires_prefixes": False,
        "pooling": "mean"
    }
    
    return model_configs.get(model_name, default_config)

def format_texts_for_model(texts: List[str], model_name: str, text_type: str = "passage") -> List[str]:
    """Format texts according to model-specific requirements."""
    config = get_model_config(model_name)
    
    if not config["requires_prefixes"]:
        return texts
    
    if text_type == "query":
        prefix = config["query_prefix"]
    elif text_type == "passage":
        prefix = config["passage_prefix"]
    else:
        prefix = config["passage_prefix"]  # Default to passage prefix
    
    return [prefix + text for text in texts]

def test_model_configs():
    """Test model configurations."""
    print("Testing model configurations...")
    
    # Test E5 model
    e5_config = get_model_config("intfloat/e5-base-v2")
    assert e5_config["requires_prefixes"] is True
    assert e5_config["query_prefix"] == "query: "
    assert e5_config["passage_prefix"] == "passage: "
    assert e5_config["pooling"] == "mean"
    print("✅ E5 model config correct")
    
    # Test BGE model  
    bge_config = get_model_config("BAAI/bge-small-en-v1.5")
    assert bge_config["requires_prefixes"] is False
    assert bge_config["query_prefix"] == ""
    assert bge_config["passage_prefix"] == ""
    assert bge_config["pooling"] == "cls"
    print("✅ BGE model config correct")
    
    # Test GTE model
    gte_config = get_model_config("thenlper/gte-base")
    assert gte_config["requires_prefixes"] is False
    assert gte_config["query_prefix"] == ""
    assert gte_config["passage_prefix"] == ""
    assert gte_config["pooling"] == "mean"
    print("✅ GTE model config correct")
    
    # Test unknown model (backward compatibility)
    unknown_config = get_model_config("unknown/model")
    assert unknown_config["requires_prefixes"] is False
    print("✅ Unknown model defaults correct")

def test_text_formatting():
    """Test text formatting functionality."""
    print("\nTesting text formatting...")
    
    texts = ["what is AI", "define machine learning"]
    
    # E5 should add prefixes
    e5_formatted_query = format_texts_for_model(texts, "intfloat/e5-base-v2", "query")
    expected_e5_query = ["query: what is AI", "query: define machine learning"]
    assert e5_formatted_query == expected_e5_query
    print("✅ E5 query formatting correct")
    
    e5_formatted_passage = format_texts_for_model(texts, "intfloat/e5-base-v2", "passage")
    expected_e5_passage = ["passage: what is AI", "passage: define machine learning"]
    assert e5_formatted_passage == expected_e5_passage
    print("✅ E5 passage formatting correct")
    
    # BGE should not modify
    bge_formatted = format_texts_for_model(texts, "BAAI/bge-small-en-v1.5", "query") 
    assert bge_formatted == texts
    print("✅ BGE text formatting correct")
    
    # GTE should not modify
    gte_formatted = format_texts_for_model(texts, "thenlper/gte-base", "query")
    assert gte_formatted == texts
    print("✅ GTE text formatting correct")

def test_target_models():
    """Test all target models for Phase 3."""
    print("\nTesting target models...")
    
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
        
        print(f"✅ Model {model} config complete")

def test_backward_compatibility():
    """Test backward compatibility with existing models."""
    print("\nTesting backward compatibility...")
    
    # Test original model
    original_model = "sentence-transformers/all-MiniLM-L6-v2"
    config = get_model_config(original_model)
    
    # Should use default config (no prefixes required)
    assert config["requires_prefixes"] is False
    assert config["query_prefix"] == ""
    assert config["passage_prefix"] == ""
    print("✅ Backward compatibility maintained")
    
    # Test that text formatting doesn't change existing behavior
    texts = ["test text 1", "test text 2"]
    formatted = format_texts_for_model(texts, original_model, "passage")
    assert formatted == texts
    print("✅ Existing model behavior unchanged")

if __name__ == "__main__":
    try:
        print("🧪 Testing Phase 3 embedding upgrade functionality...\n")
        
        test_model_configs()
        test_text_formatting()
        test_target_models()
        test_backward_compatibility()
        
        print("\n🎉 All Phase 3 core functionality tests passed!")
        print("\n📝 Summary of Phase 3 features:")
        print("   • E5 models: Require 'query:' and 'passage:' prefixes")
        print("   • BGE models: No prefixes required, use CLS pooling")
        print("   • GTE models: No prefixes required, use mean pooling")
        print("   • Backward compatibility: All existing models still work")
        print("   • Model-specific formatting: Automatic based on model name")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)