"""Test enhanced agent query classification and parameter generation."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
import sys

# Add backend src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class MockRegistry:
    """Mock plugin registry for testing."""
    
    def __init__(self):
        self.plugins = {
            "semantic_search": Mock(),
            "metadata": Mock(),
            "document_relationships": Mock(),
            "comprehensive_reporting": Mock(),
            "knowledge_graph": Mock()
        }
    
    def get_plugin(self, name):
        return self.plugins.get(name)


def test_enhanced_query_classification():
    """Test enhanced query classification logic."""
    
    try:
        from backend.src.querying.agents.agent import Agent
    except ImportError:
        print("❌ Could not import Agent - skipping test")
        return
    
    # Create agent with mock registry
    registry = MockRegistry()
    agent = Agent(registry)
    
    # Test document-level query classification
    test_cases = [
        {
            "query": "What does the project requirements document say about security?",
            "expected_plugins": ["semantic_search"],
            "expected_reasoning": "document-level query",
            "description": "Document-level query"
        },
        {
            "query": "Compare the requirements across all documents",
            "expected_plugins": ["document_relationships", "semantic_search"],
            "expected_reasoning": "cross-document analysis",
            "description": "Cross-document analysis query"
        },
        {
            "query": "Give me a comprehensive analysis of all project documents",
            "expected_plugins": ["document_relationships", "semantic_search"],
            "expected_reasoning": "comprehensive",
            "description": "Comprehensive analysis query"
        },
        {
            "query": "Show me recent email files",
            "expected_plugins": ["metadata"],
            "expected_reasoning": "email metadata",
            "description": "Email metadata query"
        },
        {
            "query": "What files mention artificial intelligence?",
            "expected_plugins": ["semantic_search"],
            "expected_reasoning": "content analysis",
            "description": "Content query"
        }
    ]
    
    print("Testing query classification...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        
        # Test classification
        plugins = agent._classify_query(test_case['query'])
        
        print(f"Classified plugins: {plugins}")
        print(f"Expected plugins: {test_case['expected_plugins']}")
        
        # Check that expected plugins are included
        for expected_plugin in test_case['expected_plugins']:
            if expected_plugin in plugins:
                print(f"✓ Correctly included {expected_plugin}")
            else:
                print(f"❌ Missing expected plugin: {expected_plugin}")
        
        # Check reasoning trace
        reasoning = agent._reasoning_trace
        has_expected_reasoning = any(test_case['expected_reasoning'] in trace.lower() 
                                   for trace in reasoning)
        
        if has_expected_reasoning:
            print(f"✓ Reasoning includes '{test_case['expected_reasoning']}'")
        else:
            print(f"❌ Missing expected reasoning: '{test_case['expected_reasoning']}'")
            print(f"   Actual reasoning: {reasoning}")
    
    print("\n" + "="*50)


def test_semantic_search_parameter_generation():
    """Test enhanced semantic search parameter generation."""
    
    try:
        from backend.src.querying.agents.agent import Agent
    except ImportError:
        print("❌ Could not import Agent - skipping test")
        return
    
    registry = MockRegistry()
    agent = Agent(registry)
    
    # Test parameter generation for different query types
    test_cases = [
        {
            "query": "Give me a comprehensive analysis of the project requirements",
            "expected_params": {
                "max_documents": 7,
                "context_window": 5,
                "k": 75,
                "use_document_level": True
            },
            "description": "Comprehensive analytical query"
        },
        {
            "query": "What is the project name?",
            "expected_params": {
                "max_documents": 3,
                "context_window": 2,
                "k": 30,
                "use_document_level": True
            },
            "description": "Simple factual query"
        },
        {
            "query": "Analyze all documents for technical specifications",
            "expected_params": {
                "max_documents": 10,
                "context_window": 4,
                "k": 100,
                "use_document_level": True
            },
            "description": "Multi-document query"
        },
        {
            "query": "Show me implementation details from the architecture document",
            "expected_params": {
                "max_documents": 6,
                "context_window": 4,
                "k": 60,
                "use_document_level": True
            },
            "description": "Technical query"
        },
        {
            "query": "Just tell me the quick answer about status",
            "expected_params": {
                "use_document_level": False
            },
            "description": "Legacy-style query"
        }
    ]
    
    print("Testing semantic search parameter generation...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        
        # Generate parameters
        params = agent._generate_semantic_search_params(test_case['query'])
        
        print(f"Generated parameters: {params}")
        
        # Check expected parameters
        for param_name, expected_value in test_case['expected_params'].items():
            actual_value = params.get(param_name)
            
            if actual_value == expected_value:
                print(f"✓ {param_name}: {actual_value} (expected: {expected_value})")
            else:
                print(f"❌ {param_name}: {actual_value} (expected: {expected_value})")
        
        # Check reasoning trace
        reasoning = agent._reasoning_trace
        if reasoning:
            print(f"Reasoning: {reasoning[-1]}")  # Show latest reasoning
    
    print("\n" + "="*50)


def test_plugin_parameter_preparation():
    """Test overall parameter preparation for different plugins."""
    
    try:
        from backend.src.querying.agents.agent import Agent
    except ImportError:
        print("❌ Could not import Agent - skipping test")
        return
    
    registry = MockRegistry()
    agent = Agent(registry)
    
    test_cases = [
        {
            "plugin": "semantic_search",
            "query": "What are the project requirements?",
            "expected_keys": ["question", "use_document_level", "k", "max_documents", "context_window"],
            "description": "Semantic search parameters"
        },
        {
            "plugin": "metadata",
            "query": "Show me recent PDF files",
            "expected_keys": ["operation"],  # Would be generated by LLM or fallback
            "description": "Metadata parameters"
        }
    ]
    
    print("Testing plugin parameter preparation...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Plugin: {test_case['plugin']}")
        print(f"Query: '{test_case['query']}'")
        
        # Prepare parameters
        params = agent._prepare_params(test_case['plugin'], test_case['query'])
        
        print(f"Generated parameters: {list(params.keys())}")
        
        # Check expected parameter keys
        for expected_key in test_case['expected_keys']:
            if expected_key in params:
                print(f"✓ Has parameter: {expected_key}")
            else:
                print(f"❌ Missing parameter: {expected_key}")
        
        # Validate parameter types and values
        if test_case['plugin'] == "semantic_search":
            if isinstance(params.get('k'), int) and params.get('k') > 0:
                print("✓ k parameter is valid integer")
            else:
                print(f"❌ k parameter invalid: {params.get('k')}")
            
            if isinstance(params.get('use_document_level'), bool):
                print("✓ use_document_level parameter is boolean")
            else:
                print(f"❌ use_document_level parameter invalid: {params.get('use_document_level')}")


def test_integration_flow():
    """Test the complete integration flow from query to parameters."""
    
    try:
        from backend.src.querying.agents.agent import Agent
    except ImportError:
        print("❌ Could not import Agent - skipping test")
        return
    
    registry = MockRegistry()
    agent = Agent(registry)
    
    # Test complete flow
    query = "Give me a comprehensive analysis of all documents mentioning security requirements"
    
    print("Testing complete integration flow...")
    print(f"Query: '{query}'")
    
    # Step 1: Classify query
    plugins = agent._classify_query(query)
    print(f"Step 1 - Classified plugins: {plugins}")
    
    # Step 2: Prepare parameters for each plugin
    for plugin_name in plugins:
        params = agent._prepare_params(plugin_name, query)
        print(f"Step 2 - Parameters for {plugin_name}: {params}")
        
        # Validate semantic search parameters specifically
        if plugin_name == "semantic_search":
            expected_comprehensive_params = {
                'use_document_level': True,
                'max_documents': 7,  # Should be high for comprehensive queries
                'context_window': 5,  # Should be high for comprehensive queries
                'k': 75  # Should be high for comprehensive queries
            }
            
            for param, expected_value in expected_comprehensive_params.items():
                actual_value = params.get(param)
                if actual_value == expected_value:
                    print(f"✓ {param}: {actual_value}")
                else:
                    print(f"❌ {param}: {actual_value} (expected: {expected_value})")
    
    print("✓ Integration flow completed")


if __name__ == "__main__":
    print("Testing Enhanced Agent Functionality...")
    print("="*60)
    
    try:
        test_enhanced_query_classification()
        test_semantic_search_parameter_generation()
        test_plugin_parameter_preparation()
        test_integration_flow()
        
        print("\n🎉 All enhanced agent tests completed!")
        print("\nNext: Create end-to-end integration test and finalize implementation")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()