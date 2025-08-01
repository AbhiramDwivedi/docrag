import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path so we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import create_default_agent, create_enhanced_agent

def setup_verbose_logging(verbose_level: int) -> None:
    """Setup verbose logging with configurable detail levels.
    
    Args:
        verbose_level: 0=off, 1=info, 2=debug, 3=trace
    """
    if verbose_level == 0:
        return
    
    # Import here to avoid circular imports
    from config.config import setup_verbose_logging as config_setup_verbose
    config_setup_verbose(verbose_level)


def parse_arguments() -> tuple[str, int]:
    """Parse command line arguments.
    
    Returns:
        tuple of (question, verbose_level)
    """
    # Manual parsing to support both --verbose and --verbose N
    import sys
    
    args = sys.argv[1:]
    verbose_level = 0
    question_parts = []
    
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ["--verbose", "-v"]:
            # Check if next argument is a valid verbose level
            if i + 1 < len(args) and args[i + 1] in ["1", "2", "3"]:
                verbose_level = int(args[i + 1])
                i += 2  # Skip both --verbose and the number
            else:
                verbose_level = 1  # Default to level 1
                i += 1
        elif arg.startswith("--verbose="):
            level_str = arg.split("=", 1)[1]
            if level_str in ["1", "2", "3"]:
                verbose_level = int(level_str)
            else:
                print(f"Invalid verbose level: {level_str}. Must be 1, 2, or 3.")
                sys.exit(1)
            i += 1
        elif arg in ["-h", "--help"]:
            print("""DocQuest CLI - Query your documents with natural language

usage: ask.py [-h] [--verbose [LEVEL]] [question ...]

positional arguments:
  question              The question to ask about your documents

options:
  -h, --help            show this help message and exit
  --verbose [LEVEL], -v [LEVEL]
                        Enable verbose logging (1=info, 2=debug, 3=trace)
                        Default level is 1 if no number given

Verbose levels:
  --verbose     Show agent reasoning and plugin selection (info level)
  --verbose 2   Show LLM interactions and SQL queries (debug level)  
  --verbose 3   Show full execution traces and timing (trace level)

Examples:
  python cli/ask.py "find all pdf files"
  python cli/ask.py --verbose "find all pdf files"
  python cli/ask.py --verbose 2 "find all pdf files"
  python cli/ask.py --verbose=2 "find all pdf files"
""")
            sys.exit(0)
        else:
            question_parts.append(arg)
            i += 1
    
    question = " ".join(question_parts)
    return question, verbose_level


# Global agent instance - initialized on first use
_agent = None

def get_agent():
    """Get the global agent instance, creating it if necessary."""
    global _agent
    if _agent is None:
        _agent = create_enhanced_agent()  # Use enhanced agent with structured metadata
    return _agent

def answer(question: str, verbose_level: int = 0) -> str:
    """Process a question using the DocQuest agent framework.
    
    This function maintains backward compatibility with the original CLI interface
    while using the new agent-based architecture internally.
    
    Args:
        question: The user's question
        verbose_level: Verbose logging level (0=off, 1=info, 2=debug, 3=trace)
        
    Returns:
        The agent's response as a string
    """
    if not question or not question.strip():
        return "Please provide a question."
    
    # Setup verbose logging before processing
    setup_verbose_logging(verbose_level)
    
    try:
        agent = get_agent()
        return agent.process_query(question.strip())
    except Exception as e:
        return f"❌ Error processing query: {e}"


if __name__ == '__main__':
    question, verbose_level = parse_arguments()
    
    if not question:
        print("Please provide a question.")
        sys.exit(1)
    
    print(answer(question, verbose_level))
