import sys
import argparse
import logging
from pathlib import Path
import os

# Add backend root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Fix Windows encoding issues with Unicode characters
if sys.platform == "win32":
    # Set environment variable to force UTF-8 output
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Global agent instance - initialized on first use
_agent = None

class VerboseFormatter(logging.Formatter):
    """Custom formatter for verbose output with emojis and indentation."""
    
    EMOJI_MAP = {
        'agent.classification': '🧠',
        'agent.execution': '🔧',
        'agent.synthesis': '📊',
        'plugin.metadata': '🗃️',
        'plugin.semantic': '📄',
        'llm.generation': '🤖',
        'sql.query': '💾',
        'timing': '⏱️'
    }
    
    def format(self, record):
        # Get emoji for logger name
        emoji = self.EMOJI_MAP.get(record.name, '📝')
        
        # Base indentation is none for top-level messages
        indent = ''
        
        # Format message with emoji and indentation
        if hasattr(record, 'sql_query'):
            # Special formatting for SQL queries with sub-indentation for parameters
            return f"{indent}{emoji} SQL Query: {record.sql_query}\n{indent}   Parameters: {record.sql_params}"
        elif hasattr(record, 'llm_prompt'):
            # Special formatting for LLM interactions with sub-indentation for details
            return f"{indent}{emoji} LLM {record.llm_model}...\n{indent}   Prompt: \"{record.llm_prompt[:100]}...\"\n{indent}   Response: {record.getMessage()}"
        else:
            return f"{indent}{emoji} {record.getMessage()}"

def get_agent():
    """Get the global agent instance, creating it if necessary."""
    global _agent
    if _agent is None:
        try:
            from backend.src.querying.agents.factory import create_phase3_agent
            _agent = create_phase3_agent()  # Use Phase III agent with knowledge graph
        except ImportError as e:
            # Handle missing dependencies gracefully
            print(f"Warning: Could not load agent dependencies: {e}")
            print("Please install the required dependencies with: pip install -r requirements.txt")
            return None
    return _agent

def setup_logging(verbose_level: int = 0):
    """Setup logging configuration based on verbose level.
    
    Args:
        verbose_level: 0=minimal, 1=info, 2=debug, 3=trace
    """
    # Clear existing handlers to ensure clean setup
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    if verbose_level == 0:
        # Minimal logging - only errors
        logging.basicConfig(level=logging.ERROR, format='%(message)s', force=True)
        return
    
    # Set logging levels based on verbose level
    if verbose_level == 1:
        level = logging.INFO
    elif verbose_level == 2:
        level = logging.DEBUG
    else:  # verbose_level >= 3
        level = 5  # Custom trace level below DEBUG
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    
    # Set custom formatter for our loggers
    formatter = VerboseFormatter()
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)
    
    # Configure specific logger levels
    loggers_config = {
        'agent.classification': level,
        'agent.execution': level,
        'agent.synthesis': level,
        'plugin.metadata': level,
        'plugin.semantic': level,
        'llm.generation': level if verbose_level >= 2 else logging.WARNING,
        'sql.query': level if verbose_level >= 2 else logging.WARNING,
        'timing': level if verbose_level >= 3 else logging.WARNING
    }
    
    for logger_name, logger_level in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logger_level)

def answer(question: str, verbose_level: int = 0) -> str:
    """Process a question using the DocQuest agent framework.
    
    This function maintains backward compatibility with the original CLI interface
    while using the new agent-based architecture internally.
    
    Args:
        question: The user's question
        verbose_level: Verbosity level (0=minimal, 1=info, 2=debug, 3=trace)
        
    Returns:
        The agent's response as a string
    """
    if not question or not question.strip():
        return "Please provide a question."
    
    # Setup logging for this query
    setup_logging(verbose_level)
    
    # Log the start of processing if verbose
    if verbose_level > 0:
        logger = logging.getLogger('agent.classification')
        logger.info(f"Processing query: \"{question}\"")
    
    try:
        agent = get_agent()
        if agent is None:
            return "❌ Error: Could not initialize agent. Please check dependencies."
        
        result = agent.process_query(question.strip())
        
        # Log timing information for trace level
        if verbose_level >= 3 and hasattr(agent, '_last_execution_time'):
            timing_logger = logging.getLogger('timing')
            execution_time = agent._last_execution_time
            timing_logger.info(f"Query completed (execution: {execution_time:.2f}s)")
        
        return result
    except Exception as e:
        return f"❌ Error processing query: {e}"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DocQuest CLI - Query your documents with natural language',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python cli/ask.py "find all pdf files"
  python cli/ask.py --verbose 1 "list recent documents" 
  python cli/ask.py --verbose 2 "how many files from last week"
  python cli/ask.py -v 3 "what is the compliance policy"
        '''.strip()
    )
    
    parser.add_argument(
        '--verbose', '-v',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='Verbose output level: 0=minimal, 1=info (agent reasoning), 2=debug (LLM/SQL), 3=trace (full execution)'
    )
    
    parser.add_argument(
        'question',
        nargs='*',
        help='Your question about the documents'
    )
    
    return parser.parse_args()

def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Combine question parts
    question = ' '.join(args.question) if args.question else ''
    
    if not question:
        print("Error: Please provide a question.")
        print("Use --help for usage information.")
        sys.exit(1)
    
    # Process the question with specified verbosity
    result = answer(question, args.verbose)
    
    # Print result with proper encoding for Windows
    try:
        print(result)
    except UnicodeEncodeError:
        # Fallback: replace problematic Unicode characters
        safe_result = result.encode('ascii', 'replace').decode('ascii')
        print(safe_result)

if __name__ == '__main__':
    main()
