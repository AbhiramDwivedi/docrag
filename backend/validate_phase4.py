#!/usr/bin/env python3
"""Simple validation script for Phase 4 features.

This script validates that the Phase 4 implementation can be imported
and basic functionality works without requiring external dependencies.
"""

import sys
from pathlib import Path

# Add backend src to path
backend_root = Path(__file__).parent / "src"
sys.path.insert(0, str(backend_root))

def test_imports():
    """Test that all Phase 4 modules can be imported."""
    print("Testing imports...")
    
    try:
        from shared.reranking import CrossEncoderReranker, expand_entity_query
        print("✓ Reranking module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import reranking module: {e}")
        return False
    
    try:
        from shared.config import settings
        print("✓ Config module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import config module: {e}")
        return False
    
    return True

def test_cross_encoder_basic():
    """Test basic cross-encoder functionality."""
    print("\nTesting cross-encoder...")
    
    from shared.reranking import CrossEncoderReranker
    
    # Test initialization
    reranker = CrossEncoderReranker("ms-marco-MiniLM-L-6-v2")
    print(f"✓ CrossEncoderReranker initialized with model: {reranker.model_name}")
    
    # Test availability check (should work even without sentence-transformers)
    available = reranker.is_available()
    print(f"✓ Cross-encoder availability: {available}")
    
    # Test reranking with mock data (should handle unavailable gracefully)
    test_results = [
        {"text": "Apple is a fruit", "similarity": 0.7},
        {"text": "Apple Inc is a technology company", "similarity": 0.6}
    ]
    
    reranked = reranker.rerank("What is Apple?", test_results, top_k=1)
    print(f"✓ Reranking completed, returned {len(reranked)} results")
    
    return True

def test_query_expansion():
    """Test query expansion functionality."""
    print("\nTesting query expansion...")
    
    from shared.reranking import expand_entity_query, create_query_variants
    
    # Test entity detection and expansion
    test_cases = [
        ("What is Tesla?", ["Tesla"]),
        ("Tell me about AI", ["AI"]),
        ("COVID research updates", ["COVID"]),
        ("No entities here", [])
    ]
    
    for query, entities in test_cases:
        variants = expand_entity_query(query, entities)
        print(f"✓ Query: '{query}' -> {len(variants)} variants")
        if len(variants) > 1:
            print(f"  Expanded variants: {variants[1:]}")
    
    # Test synonym mapping
    variants = create_query_variants("What is Apple's stock price?", ["Apple"])
    assert len(variants) > 1, "Should create variants for Apple"
    print(f"✓ Apple query expansion: {len(variants)} variants")
    
    return True

def test_config_attributes():
    """Test that Phase 4 config attributes are available."""
    print("\nTesting configuration...")
    
    try:
        from shared.config import get_settings
        settings = get_settings()
        
        # Test Phase 4 attributes
        phase4_attrs = [
            'enable_cross_encoder_reranking',
            'cross_encoder_model', 
            'cross_encoder_top_k',
            'enable_query_expansion',
            'query_expansion_method',
            'enable_entity_indexing',
            'entity_boost_factor'
        ]
        
        for attr in phase4_attrs:
            if hasattr(settings, attr):
                value = getattr(settings, attr)
                print(f"✓ {attr}: {value}")
            else:
                print(f"✗ Missing config attribute: {attr}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_semantic_search_plugin():
    """Test that semantic search plugin can be imported with new features."""
    print("\nTesting semantic search plugin...")
    
    try:
        from querying.agents.plugins.semantic_search import SemanticSearchPlugin
        
        plugin = SemanticSearchPlugin()
        print("✓ SemanticSearchPlugin imported and initialized")
        
        # Test plugin info includes Phase 4 capabilities
        info = plugin.get_info()
        phase4_capabilities = [
            "cross_encoder_reranking",
            "query_expansion", 
            "entity_indexing",
            "precision_optimization"
        ]
        
        for capability in phase4_capabilities:
            if capability in info.capabilities:
                print(f"✓ Plugin supports: {capability}")
            else:
                print(f"✗ Missing capability: {capability}")
                return False
        
        # Check version bump
        if "2.2.0" in info.version:
            print(f"✓ Plugin version updated: {info.version}")
        else:
            print(f"? Plugin version: {info.version}")
        
        return True
        
    except Exception as e:
        print(f"✗ Semantic search plugin test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("Phase 4 Feature Validation")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_cross_encoder_basic,
        test_query_expansion,
        test_config_attributes,
        test_semantic_search_plugin
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All Phase 4 features validated successfully!")
        return 0
    else:
        print("❌ Some tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())