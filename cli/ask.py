import sys
from pathlib import Path

# Add parent directory to path so we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import Agent, PluginRegistry
from agent.plugins import SemanticSearchPlugin, MetadataPlugin


def answer(question: str):
    """
    Answer user questions using the DocQuest agent framework.
    
    This function maintains backward compatibility with the original CLI interface
    while routing queries through the new plugin-based agent system.
    
    Args:
        question: User's natural language question
        
    Returns:
        Answer string
    """
    # Initialize agent with default plugins
    registry = PluginRegistry()
    registry.register(SemanticSearchPlugin())
    registry.register(MetadataPlugin())
    
    agent = Agent(registry)
    
    # Process the query through the agent
    return agent.process_query(question)


if __name__ == '__main__':
    print(answer(' '.join(sys.argv[1:])))
