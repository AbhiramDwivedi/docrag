"""Central configuration loader."""
from pathlib import Path
import yaml
import logging
import sys
from pydantic import BaseModel, Field, field_validator
import os
from typing import Dict, Any, Optional

CONFIG_PATH = Path(__file__).with_name("config.yaml")


class VerboseFormatter(logging.Formatter):
    """Custom formatter for verbose output with emojis and structure."""
    
    # Emoji mappings for different log levels and categories
    EMOJIS = {
        'agent.classification': '📊',
        'agent.reasoning': '🧠',
        'agent.execution': '⚙️',
        'plugin.metadata': '🔧',
        'plugin.semantic': '🔍',
        'llm.generation': '🤖',
        'llm.prompt': '💭',
        'llm.response': '📝',
        'sql.query': '🗄️',
        'timing': '⏱️',
        'error': '❌',
        'success': '✅',
        'info': 'ℹ️',
        'warning': '⚠️'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with emojis and structure."""
        # Get emoji based on logger name or level
        emoji = self._get_emoji(record)
        
        # Format the message with proper indentation
        message = record.getMessage()
        
        # Add timing information if available
        if hasattr(record, 'execution_time'):
            message += f" (execution: {record.execution_time:.2f}s)"
        
        # Format with emoji and optional indentation
        if hasattr(record, 'indent_level'):
            indent = "   " * record.indent_level
            return f"{indent}{emoji} {message}"
        else:
            return f"{emoji} {message}"
    
    def _get_emoji(self, record: logging.LogRecord) -> str:
        """Get appropriate emoji for log record."""
        # Check for specific logger name patterns
        logger_name = record.name
        
        for pattern, emoji in self.EMOJIS.items():
            if pattern in logger_name:
                return emoji
        
        # Fallback to level-based emojis
        if record.levelno >= logging.ERROR:
            return self.EMOJIS['error']
        elif record.levelno >= logging.WARNING:
            return self.EMOJIS['warning']
        else:
            return self.EMOJIS['info']


def setup_verbose_logging(verbose_level: int) -> None:
    """Setup verbose logging configuration.
    
    Args:
        verbose_level: 1=info, 2=debug, 3=trace
    """
    if verbose_level == 0:
        return
    
    # Clear any existing configuration first
    logging.getLogger().handlers.clear()
    
    # Map verbose levels to logging levels
    level_mapping = {
        1: logging.INFO,      # Agent reasoning, plugin selection
        2: logging.DEBUG,     # LLM interactions, SQL queries
        3: 5                  # TRACE level (custom level below DEBUG)
    }
    
    # Add custom TRACE level
    if not hasattr(logging, 'TRACE'):
        logging.addLevelName(5, "TRACE")
        
        def trace(self, msg, *args, **kwargs):
            if self.isEnabledFor(5):
                self._log(5, msg, args, **kwargs)
        
        logging.Logger.trace = trace
        logging.TRACE = 5
    
    # Configure root logger
    log_level = level_mapping.get(verbose_level, logging.INFO)
    
    # Create console handler with verbose formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(VerboseFormatter())
    
    # Configure specific loggers for different components
    loggers_config = {
        'agent': log_level,
        'agent.classification': log_level,
        'agent.reasoning': log_level,
        'agent.execution': log_level,
        'plugin.metadata': log_level,
        'plugin.semantic': log_level,
        'llm.generation': log_level,
        'llm.prompt': log_level,
        'llm.response': log_level,
        'sql.query': log_level,
        'timing': log_level if verbose_level >= 3 else logging.CRITICAL,  # Only show timing at trace level
    }
    
    for logger_name, level in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()  # Clear existing handlers
        logger.setLevel(level)
        logger.addHandler(console_handler)
        logger.propagate = False  # Prevent duplicate messages
    
    # Set root logger level and add handler if needed
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)


class Settings(BaseModel):
    sync_root: Path = Field(default=Path.home() / "Documents", description="Local document folder to watch and index")
    db_path: Path = Field(default=Path("data/docmeta.db"))
    vector_path: Path = Field(default=Path("data/vector.index"))
    chunk_size: int = 800
    overlap: int = 150
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    openai_api_key: Optional[str] = None

    @field_validator('sync_root', mode='before')
    @classmethod
    def expand_sync_root(cls, v: Any) -> Path:
        """Expand tilde and resolve path."""
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        return v

def load_settings(overrides: Optional[Dict[str, Any]] = None) -> Settings:
    data: Dict[str, Any] = {}
    if CONFIG_PATH.exists():
        yaml_data = yaml.safe_load(CONFIG_PATH.read_text())
        if yaml_data:
            data.update(yaml_data)
    env_map: Dict[str, Optional[str]] = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "sync_root": os.getenv("SYNC_ROOT")
    }
    data.update({k: v for k, v in env_map.items() if v})
    if overrides:
        data.update(overrides)
    return Settings(**data)

# Global settings instance - lazy loaded
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get settings instance, loading on first access."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings

# For backward compatibility - create a proxy object that behaves like the settings
class SettingsProxy:
    def __getattr__(self, name):
        return getattr(get_settings(), name)

settings = SettingsProxy()
