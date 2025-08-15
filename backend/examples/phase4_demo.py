"""Example usage of Phase 4 advanced features in DocQuest.

This example demonstrates how to use the new Phase 4 features:
- Cross-encoder reranking for improved precision
- Query expansion for entity queries
- Configuration of advanced features
"""

import sys
from pathlib import Path

# Add backend src to path for imports
backend_root = Path(__file__).parent / "src"
sys.path.insert(0, str(backend_root))

from shared.config import load_settings
from querying.agents.plugins.semantic_search import SemanticSearchPlugin


def demo_cross_encoder_config():
    """Demonstrate cross-encoder configuration."""
    print("Cross-encoder Reranking Configuration")
    print("-" * 40)
    
    # Example configuration overrides for cross-encoder
    config_overrides = {
        "enable_cross_encoder_reranking": True,
        "cross_encoder_model": "ms-marco-MiniLM-L-6-v2",
        "cross_encoder_top_k": 15,  # Rerank top 100 -> top 15
    }
    
    settings = load_settings(config_overrides)
    
    print(f"Cross-encoder enabled: {settings.enable_cross_encoder_reranking}")
    print(f"Cross-encoder model: {settings.cross_encoder_model}")
    print(f"Top-k after reranking: {settings.cross_encoder_top_k}")
    print()
    
    return settings


def demo_query_expansion_config():
    """Demonstrate query expansion configuration."""
    print("Query Expansion Configuration")
    print("-" * 40)
    
    config_overrides = {
        "enable_query_expansion": True,
        "query_expansion_method": "synonyms",
    }
    
    settings = load_settings(config_overrides)
    
    print(f"Query expansion enabled: {settings.enable_query_expansion}")
    print(f"Expansion method: {settings.query_expansion_method}")
    print()
    
    return settings


def demo_entity_indexing_config():
    """Demonstrate entity indexing configuration."""
    print("Entity Indexing Configuration")
    print("-" * 40)
    
    config_overrides = {
        "enable_entity_indexing": True,
        "entity_boost_factor": 0.25,
    }
    
    settings = load_settings(config_overrides)
    
    print(f"Entity indexing enabled: {settings.enable_entity_indexing}")
    print(f"Entity boost factor: {settings.entity_boost_factor}")
    print()
    
    return settings


def demo_semantic_search_with_phase4():
    """Demonstrate semantic search with Phase 4 features enabled."""
    print("Semantic Search with Phase 4 Features")
    print("-" * 40)
    
    # Initialize plugin
    plugin = SemanticSearchPlugin()
    
    # Show plugin capabilities
    info = plugin.get_info()
    print(f"Plugin: {info.name} v{info.version}")
    print(f"Description: {info.description}")
    print()
    
    print("Phase 4 Capabilities:")
    phase4_caps = [cap for cap in info.capabilities if cap in [
        "cross_encoder_reranking", "query_expansion", 
        "entity_indexing", "precision_optimization"
    ]]
    for cap in phase4_caps:
        print(f"  ✓ {cap}")
    print()
    
    # Example parameters for Phase 4 features
    example_params = {
        "question": "What is Tesla's latest electric vehicle model?",
        "k": 50,  # Initial retrieval
        "enable_cross_encoder": True,  # Override config
        "enable_query_expansion": True,  # Override config
        "enable_mmr": True,
        "mmr_k": 10  # Final results after MMR
    }
    
    print("Example search parameters:")
    for key, value in example_params.items():
        print(f"  {key}: {value}")
    print()
    
    # Note: We can't actually execute the search without a vector store and API key
    print("Note: Actual search execution requires:")
    print("  - Configured vector store with indexed documents")
    print("  - Valid OpenAI API key")
    print("  - Optional: sentence-transformers for cross-encoder")
    print()


def demo_config_yaml_example():
    """Show example configuration in YAML format."""
    print("Example config.yaml for Phase 4")
    print("-" * 40)
    
    yaml_config = """
# Phase 4: Advanced features for improved precision and entity-awareness
enable_cross_encoder_reranking: true   # Enable cross-encoder reranking
cross_encoder_model: "ms-marco-MiniLM-L-6-v2"  # Cross-encoder model
cross_encoder_top_k: 20                # Results after reranking (from top 100)

enable_query_expansion: true           # Enable query expansion for entities
query_expansion_method: "synonyms"     # Expansion method

enable_entity_indexing: false          # Enable entity-aware indexing (requires spaCy)
entity_boost_factor: 0.2               # Boost factor for entity matches

# Combined with existing settings
retrieval_k: 100                       # Initial retrieval count
mmr_k: 20                             # Final results after MMR
mmr_lambda: 0.7                       # MMR diversity balance

enable_hybrid_search: true            # Use with existing hybrid search
enable_debug_logging: true            # See Phase 4 debug info
"""
    
    print(yaml_config)


def main():
    """Run Phase 4 feature demonstrations."""
    print("DocQuest Phase 4: Advanced Features Demo")
    print("=" * 50)
    print()
    
    # Demonstrate configuration
    demo_cross_encoder_config()
    demo_query_expansion_config() 
    demo_entity_indexing_config()
    
    # Demonstrate plugin usage
    demo_semantic_search_with_phase4()
    
    # Show configuration example
    demo_config_yaml_example()
    
    print("Phase 4 Implementation Summary:")
    print("-" * 40)
    print("✓ Cross-encoder reranking: Improves precision by reranking top results")
    print("✓ Query expansion: Expands entity queries with variants and synonyms")
    print("✓ Entity-aware configuration: Ready for entity indexing during ingestion")
    print("✓ Backward compatibility: All features are opt-in via configuration")
    print("✓ Performance considerations: Features have reasonable defaults and limits")
    print()
    print("Benefits:")
    print("  - Higher precision in final answers through cross-encoder reranking")
    print("  - Better coverage for entity variants through query expansion")
    print("  - Configurable features that don't cause regressions when disabled")
    print("  - Foundation for future entity-aware indexing improvements")


if __name__ == "__main__":
    main()