#!/usr/bin/env python3
"""Demo of enhanced metadata functionality as MCP-like interface.

This demo shows how the metadata functionality has been enhanced to work
like an MCP (Model Context Protocol) server interface, where natural language
is converted to structured commands by the LLM agent layer.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.src.querying.agents.factory import create_enhanced_agent
from backend.src.querying.agents.query_parser import QueryParser

def demo_query_parsing():
    """Demo how natural language is parsed into structured commands."""
    print("=== DEMO: Natural Language → Structured Commands ===\n")
    
    parser = QueryParser()
    
    test_queries = [
        "find the three latest modified presentations",
        "find all emails related to google", 
        "show me five recent documents",
        "how many PDF files do we have",
        "list Excel files from last week"
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        operation, params = parser.parse_query(query)
        print(f"  → Operation: {operation}")
        print(f"  → Parameters: {params}")
        print()

def demo_agent_processing():
    """Demo how the agent processes queries using the enhanced interface."""
    print("=== DEMO: Agent Processing with Enhanced Metadata ===\n")
    
    agent = create_enhanced_agent()
    
    test_queries = [
        "find the three latest modified presentations",
        "find all emails related to google",
        "how many files do we have",
        "show me recent documents"
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        result = agent.process_query(query)
        print(f"Response: {result}")
        print("\nReasoning:")
        print(agent.explain_reasoning())
        print("-" * 60)

def demo_mcp_like_interface():
    """Demo how this works like an MCP server interface."""
    print("=== DEMO: MCP-like Interface Behavior ===\n")
    
    print("Before: Hard-coded NLP parsing in metadata plugin")
    print("- Plugin tried to parse 'three', '3', 'III' itself")
    print("- Plugin needed to know emails are .msg files")
    print("- Brittle regex-based approach")
    print()
    
    print("After: LLM agent layer + structured metadata commands")
    print("- Agent converts 'three' → 3, 'presentations' → PPTX")
    print("- Plugin gets clean parameters: {operation: 'get_latest_files', file_type: 'PPTX', count: 3}")
    print("- Plugin focused on executing operations, not parsing language")
    print()
    
    # Show the transformation
    from backend.src.querying.agents.query_parser import create_enhanced_metadata_params
    
    examples = [
        "find the three latest modified presentations",
        "find all emails related to google"
    ]
    
    for query in examples:
        print(f"Natural Language: '{query}'")
        params = create_enhanced_metadata_params(query)
        print(f"Structured Command: {params}")
        print()

if __name__ == "__main__":
    print("🚀 Enhanced Metadata Functionality Demo")
    print("=" * 50)
    print()
    
    demo_query_parsing()
    print()
    
    demo_agent_processing()
    print()
    
    demo_mcp_like_interface()
    
    print("✅ Demo completed!")
    print("\nKey Benefits:")
    print("- Natural language understanding moved to agent layer (LLM)")
    print("- Metadata plugin simplified to structured operations")
    print("- Better separation of concerns")
    print("- More reliable and maintainable")
    print("- Works like MCP server but embedded in cli.ask")