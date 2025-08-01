import sys
from pathlib import Path

# Add parent directory to path so we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import create_default_agent, create_enhanced_agent

# Global agent instance - initialized on first use
_agent = None

def get_agent():
    """Get the global agent instance, creating it if necessary."""
    global _agent
    if _agent is None:
        _agent = create_enhanced_agent()  # Use enhanced agent with structured metadata
    return _agent

def answer(question: str) -> str:
    """Process a question using the DocQuest agent framework.
    
    This function maintains backward compatibility with the original CLI interface
    while using the new agent-based architecture internally.
    
    Args:
        question: The user's question
        
    Returns:
        The agent's response as a string
    """
    if not question or not question.strip():
        return "Please provide a question."
    
    try:
        agent = get_agent()
        return agent.process_query(question.strip())
    except Exception as e:
        return f"❌ Error processing query: {e}"


if __name__ == '__main__':
    print(answer(' '.join(sys.argv[1:])))
