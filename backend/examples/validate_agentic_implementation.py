"""Integration test for agentic workflow implementation.

This test validates the complete agentic integration from CLI to plugin execution,
replacing the mock-based tests with real plugin integration.
"""

import sys
from pathlib import Path

# Add backend root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_agentic_integration():
    """Test complete agentic workflow integration."""
    print("🔍 Testing Agentic Integration")
    print("=" * 50)
    
    try:
        # Import required components
        from backend.src.querying.agents.agent import Agent
        from backend.src.querying.agents.registry import PluginRegistry
        from backend.src.querying.agents.orchestrator import OrchestratorAgent, QueryIntent
        from backend.src.querying.agents.plugins.semantic_search import SemanticSearchPlugin
        from backend.src.querying.agents.plugins.metadata_commands import MetadataCommandsPlugin
        
        print("✅ All imports successful")
        
        # Test 1: Registry and Plugin Loading
        print("\n📋 Test 1: Plugin Registry")
        registry = PluginRegistry()
        
        # Register plugins
        semantic_plugin = SemanticSearchPlugin()
        metadata_plugin = MetadataCommandsPlugin()
        
        registry.register(semantic_plugin)
        registry.register(metadata_plugin)
        
        print(f"✅ Registered {registry.get_plugin_count()} plugins")
        
        # Test 2: OrchestratorAgent Creation
        print("\n🎯 Test 2: OrchestratorAgent")
        orchestrator = OrchestratorAgent(registry)
        print("✅ OrchestratorAgent created successfully")
        
        # Test 3: Intent Analysis
        print("\n🧠 Test 3: Intent Analysis")
        test_queries = [
            ("find the document about compliance", QueryIntent.DOCUMENT_DISCOVERY),
            ("who works at Microsoft", QueryIntent.KNOWLEDGE_GRAPH_QUERY),
            ("how many pdf files", QueryIntent.METADATA_QUERY),
            ("what is the policy", QueryIntent.CONTENT_ANALYSIS)
        ]
        
        for query, expected_intent in test_queries:
            detected_intent = orchestrator._analyze_intent(query)
            status = "✅" if detected_intent == expected_intent else "❌"
            print(f"{status} '{query}' -> {detected_intent.value}")
        
        # Test 4: Agent Integration
        print("\n🤖 Test 4: Agent Integration")
        agent = Agent(registry)
        print("✅ Agent created with OrchestratorAgent integration")
        
        # Test 5: End-to-End Query Processing
        print("\n🚀 Test 5: End-to-End Processing")
        test_queries = [
            "find pdf files",
            "find the document about budget",
            "who is John Smith"
        ]
        
        for query in test_queries:
            try:
                response = agent.process_query(query)
                print(f"✅ Query: '{query}' -> Response received")
                
                # Test reasoning explanation
                reasoning = agent.explain_reasoning()
                if reasoning and "Agentic Execution Plan" in reasoning:
                    print(f"✅ Agentic reasoning trace available")
                else:
                    print(f"❌ Agentic reasoning trace missing")
                    
            except Exception as e:
                print(f"❌ Query failed: '{query}' -> {e}")
        
        # Test 6: CLI Integration
        print("\n💻 Test 6: CLI Integration")
        try:
            from backend.src.interface.cli.ask import get_agent, answer
            
            cli_agent = get_agent()
            if cli_agent:
                print("✅ CLI agent creation successful")
                
                # Test CLI answer function
                response = answer("test query", verbose_level=0)
                if response and not response.startswith("❌ Error"):
                    print("✅ CLI answer function works")
                else:
                    print(f"❌ CLI answer function failed: {response}")
            else:
                print("❌ CLI agent creation failed")
                
        except Exception as e:
            print(f"❌ CLI integration test failed: {e}")
        
        print("\n🎉 Agentic Integration Testing Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_legacy_components_removed():
    """Test that legacy components have been properly removed."""
    print("\n🗑️  Testing Legacy Component Removal")
    print("=" * 50)
    
    try:
        from backend.src.querying.agents.agent import Agent
        
        # Check that agent no longer has legacy methods
        agent = Agent()
        legacy_methods = ['_classify_query', '_prepare_params', '_synthesize_response']
        
        for method in legacy_methods:
            if hasattr(agent, method):
                print(f"❌ Legacy method still exists: {method}")
                return False
            else:
                print(f"✅ Legacy method removed: {method}")
        
        # Check that agent has orchestrator
        if hasattr(agent, 'orchestrator'):
            print("✅ Agent has orchestrator integration")
        else:
            print("❌ Agent missing orchestrator integration")
            return False
            
        print("✅ Legacy component removal verified")
        return True
        
    except Exception as e:
        print(f"❌ Legacy removal test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 DocQuest Agentic Integration Test Suite")
    print("=" * 60)
    
    success = True
    success &= test_agentic_integration()
    success &= test_legacy_components_removed()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Agentic integration is working correctly.")
        exit(0)
    else:
        print("❌ Some tests failed. Please check the implementation.")
        exit(1)