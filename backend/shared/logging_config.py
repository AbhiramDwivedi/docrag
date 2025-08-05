"""Logging configuration for DocQuest backend."""

import logging
from typing import Dict


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
        indent = ""
        if hasattr(record, 'indent_level'):
            indent = "  " * record.indent_level
            
        # Format the message
        msg = super().format(record)
        
        # Special formatting for SQL queries
        if hasattr(record, 'sql_query'):
            msg += f"\n{indent}  📝 Query: {record.sql_query}"
            if hasattr(record, 'sql_params'):
                msg += f"\n{indent}  📋 Params: {record.sql_params}"
                
        # Special formatting for LLM interactions
        if hasattr(record, 'llm_prompt'):
            msg += f"\n{indent}  💭 Prompt: {record.llm_prompt[:100]}..."
            if hasattr(record, 'llm_model'):
                msg += f"\n{indent}  🤖 Model: {record.llm_model}"
        
        return f"{emoji} {indent}{msg}"


def setup_logging(verbose_level: int = 0):
    """Setup logging with appropriate verbosity level."""
    if verbose_level == 0:
        level = logging.WARNING
    elif verbose_level == 1:
        level = logging.INFO
    elif verbose_level == 2:
        level = logging.DEBUG
    else:
        level = logging.DEBUG
        
    logging.basicConfig(
        level=level,
        format='%(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    if verbose_level >= 2:
        # Use verbose formatter for debug and trace levels
        formatter = VerboseFormatter()
        for handler in logging.getLogger().handlers:
            handler.setFormatter(formatter)