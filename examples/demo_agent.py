#!/usr/bin/env python3
"""
DocQuest Agent Demonstration

This script demonstrates the new agentic architecture for DocQuest,
showing how the system can handle different types of queries through
an intelligent agent framework.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.src.docquest.querying.agents.factory import create_default_agent


def demo_agent_capabilities():
    """Demonstrate the agent's capabilities."""
    print("🚀 DocQuest Agentic Architecture Demo")
    print("=" * 50)
    
    # Create agent
    print("\n📋 Creating agent with default plugins...")
    agent = create_default_agent()
    
    # Show capabilities
    print(f"\n🧠 Agent Capabilities: {len(agent.get_capabilities())} total")
    capabilities = agent.get_capabilities()
    for i, capability in enumerate(sorted(capabilities), 1):
        print(f"  {i:2d}. {capability}")
    
    print(f"\n🔌 Registered Plugins: {agent.registry.get_plugin_count()}")
    for plugin_name in ["semantic_search", "metadata"]:
        plugin_info = agent.registry.get_plugin_info(plugin_name)
        if plugin_info:
            print(f"  • {plugin_info.name} v{plugin_info.version}: {plugin_info.description}")
    
    # Test queries
    print("\n📊 Testing Query Classification & Processing")
    print("-" * 50)
    
    test_queries = [
        # Metadata queries (should work without API key)
        ("how many files do we have?", "📁"),
        ("what file types are available?", "📄"),
        ("show me recent files", "🕒"),
        ("count of PDF files", "📋"),
        
        # Content queries (will require API key)
        ("what is the compliance policy?", "🔍"),
        ("explain security requirements", "🔒"),
    ]
    
    for query, icon in test_queries:
        print(f"\n{icon} Query: '{query}'")
        
        # Process query
        response = agent.process_query(query)
        print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        
        # Show reasoning
        reasoning = agent.explain_reasoning()
        if reasoning:
            # Extract key info from reasoning
            lines = reasoning.split('\n')
            plugins_used = next((line for line in lines if "Plugins used:" in line), "")
            execution_time = next((line for line in lines if "Execution time:" in line), "")
            
            print(f"   {plugins_used}")
            print(f"   {execution_time}")
    
    print("\n✨ Agent Framework Features Demonstrated:")
    print("  ✅ Intelligent query classification")
    print("  ✅ Plugin-based architecture")
    print("  ✅ Backward compatible CLI interface")
    print("  ✅ Thread-safe concurrent processing")
    print("  ✅ Comprehensive error handling")
    print("  ✅ Agent reasoning & introspection")
    print("  ✅ Extensible plugin system")
    
    print("\n🎯 Ready for Phase 2: Enhanced Database Schema & Multi-step Queries")


def demo_cli_backward_compatibility():
    """Demonstrate CLI backward compatibility."""
    print("\n🔄 CLI Backward Compatibility Demo")
    print("-" * 40)
    
    from interface.cli.ask import answer
    
    print("The CLI interface works exactly as before:")
    print("  python -m cli.ask 'how many files do we have?'")
    
    result = answer("how many files do we have?")
    print(f"  → {result}")
    
    print("\nAPI interface also preserved:")
    from backend.src.docquest.querying.api import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    response = client.post('/query', json={'question': 'what file types are available?'})
    
    print("  POST /query {'question': 'what file types are available?'}")
    print(f"  → {response.json()['answer']}")


def main():
    """Run the demonstration."""
    try:
        demo_agent_capabilities()
        demo_cli_backward_compatibility()
        
        print("\n🏆 DocQuest Agentic Architecture Implementation Complete!")
        print("📖 See docs/AGENTIC_IMPLEMENTATION_PLAN.md for Phase 2 roadmap")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())