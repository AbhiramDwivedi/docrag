# Configuration and Shared Utilities

This directory contains configuration management, shared utilities, and common functionality used across all DocQuest components. It serves as the central hub for system settings, logging configuration, and reusable code.

## Purpose

The shared module provides centralized configuration management and common utilities to ensure consistent behavior across the entire DocQuest system, from document ingestion to query processing.

## Components

### [config.py](config.py)
Configuration management and validation system:

**Core Features**:
- **Settings Loading**: Automatic config.yaml loading with validation
- **Environment Variables**: Support for environment-based overrides
- **Type Validation**: Pydantic-based configuration validation
- **Default Values**: Sensible defaults for all configuration options
- **Hot Reloading**: Dynamic configuration updates without restart

### [config.yaml.template](config.yaml.template)
Template configuration file for safe sharing and setup:

**Purpose**: Provides a complete configuration template with:
- All available settings and their descriptions
- Safe default values for development
- Placeholder values for sensitive information (API keys)
- Documentation for each configuration section

### [logging_config.py](logging_config.py)
Advanced logging system with contextual formatting:

**Features**:
- **Contextual Loggers**: Named loggers for different components
- **Emoji Formatting**: Visual distinction between log contexts
- **Verbose Levels**: Multiple verbosity levels (0=minimal, 1=info, 2=debug, 3=trace)
- **Special Formatting**: Enhanced formatting for SQL queries and LLM interactions
- **Performance Logging**: Automatic timing and performance metrics

### [utils.py](utils.py)
Common utility functions and helpers:
- File system operations and path handling
- String processing and validation utilities
- Date/time formatting and parsing
- Error handling and retry mechanisms

## Configuration Schema

### Core Settings
```yaml
# Document Processing
sync_root: "~/Documents/MyDocuments"    # Source document directory
file_types: [".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".msg", ".eml"]

# Storage Configuration
db_path: "data/docmeta.db"              # SQLite metadata database
vector_path: "data/vector.index"        # FAISS vector index
knowledge_graph_path: "data/knowledge_graph.db"  # Knowledge graph database

# Processing Settings
chunk_size: 800                         # Text chunk size in characters
chunk_overlap: 150                      # Overlap between chunks
embed_model: "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model

# AI Integration
openai_api_key: "your-api-key-here"     # OpenAI API key
openai_model: "gpt-4o-mini"             # Default LLM model
openai_temperature: 0.1                 # LLM temperature setting
```

### Advanced Configuration
```yaml
# Performance Tuning
batch_size: 32                          # Processing batch size
max_workers: 4                          # Parallel processing workers
timeout: 30                             # Request timeout seconds

# Quality Control
similarity_threshold: 0.7               # Minimum similarity for search
confidence_threshold: 0.8               # Minimum confidence for entities
max_results: 100                        # Maximum search results

# Logging
log_level: "INFO"                       # Default log level
log_format: "detailed"                  # Log format style
enable_performance_logging: true        # Performance metrics
```

## Usage Patterns

### Configuration Loading
```python
from backend.src.shared.config import load_config

# Load configuration with validation
settings = load_config()
print(f"Document root: {settings.sync_root}")
print(f"Embedding model: {settings.embed_model}")
```

### Environment Overrides
```bash
# Override settings via environment variables
export DOCQUEST_SYNC_ROOT="/path/to/documents"
export DOCQUEST_OPENAI_API_KEY="sk-..."
export DOCQUEST_LOG_LEVEL="DEBUG"

# Settings automatically pick up environment variables
python -m backend.src.interface.cli.ask "test query"
```

### Logging Setup
```python
from backend.src.shared.logging_config import setup_logging, get_logger

# Initialize logging system
setup_logging(verbose_level=2)

# Get contextual logger
logger = get_logger("pdf.extraction")
logger.info("Processing PDF file", extra={"file_path": "doc.pdf"})

# SQL query logging with special formatting
sql_logger = get_logger("sql.query")
sql_logger.info("Executing query", extra={
    "sql_query": "SELECT * FROM chunks WHERE file_path = ?",
    "sql_params": ["document.pdf"]
})
```

### Utility Functions
```python
from backend.src.shared.utils import (
    normalize_path, 
    retry_with_backoff,
    format_file_size,
    validate_file_access
)

# Path normalization
doc_path = normalize_path("~/Documents/report.pdf")

# Retry mechanism
@retry_with_backoff(max_retries=3)
def unreliable_operation():
    # Implementation that might fail
    pass

# File utilities
size_str = format_file_size(1024*1024)  # "1.0 MB"
can_read = validate_file_access("/path/to/file.pdf")
```

## Configuration Management

### Initial Setup
```bash
# Copy template to create local configuration
cp backend/src/shared/config.yaml.template backend/src/shared/config.yaml

# Edit configuration
nano backend/src/shared/config.yaml
```

### Validation
```python
from backend.src.shared.config import ConfigSettings

# Configuration is automatically validated on load
try:
    settings = ConfigSettings.from_yaml("config.yaml")
except ValidationError as e:
    print(f"Configuration error: {e}")
```

### Environment Integration
```python
# Configuration supports environment variable overrides
# DOCQUEST_SETTING_NAME overrides setting_name in YAML

import os
os.environ["DOCQUEST_CHUNK_SIZE"] = "1000"
settings = load_config()  # chunk_size will be 1000
```

## Logging Architecture

### Contextual Loggers
```python
# Different components use specific logger names
pdf_logger = get_logger("extraction.pdf")       # 📄 PDF extraction
kg_logger = get_logger("knowledge.graph")       # 🕸️ Knowledge graph
api_logger = get_logger("interface.api")        # 🌐 API endpoints
agent_logger = get_logger("agent.classification") # 🤖 Agent system
```

### Verbose Levels
- **Level 0 (Minimal)**: Errors and critical warnings only
- **Level 1 (Info)**: General progress and status information
- **Level 2 (Debug)**: Detailed processing information
- **Level 3 (Trace)**: Full diagnostic information including timing

### Special Log Formatting
```python
# SQL queries get special formatting
logger.info("Database query", extra={
    "sql_query": "SELECT COUNT(*) FROM chunks",
    "sql_params": [],
    "execution_time": 0.023
})

# LLM interactions get enhanced formatting  
logger.info("LLM generation", extra={
    "llm_prompt": "Answer this question...",
    "llm_model": "gpt-4o-mini",
    "llm_tokens": 150,
    "llm_cost": 0.001
})
```

## Extension Points

### Custom Configuration
```python
# Add new configuration sections
class CustomSettings(ConfigSettings):
    custom_feature_enabled: bool = False
    custom_api_endpoint: str = "https://api.example.com"
    custom_timeout: int = 60
```

### Additional Utilities
```python
# Add domain-specific utilities
def custom_text_processor(text: str) -> str:
    """Domain-specific text processing"""
    return processed_text

def custom_file_validator(path: str) -> bool:
    """Custom file validation logic"""
    return is_valid
```

### Enhanced Logging
```python
# Custom log formatters
class CustomLogFormatter(logging.Formatter):
    """Domain-specific log formatting"""
    def format(self, record):
        # Custom formatting logic
        return formatted_message
```

## Security Considerations

### API Key Management
- API keys stored in local config.yaml (git-ignored)
- Environment variable support for CI/CD
- Template file contains safe placeholder values
- No sensitive data committed to version control

### Configuration Validation
- Type checking for all configuration values
- Range validation for numeric settings
- Path existence checking for file system settings
- API key format validation

### Access Control
- Configuration files have appropriate file permissions
- Sensitive settings are clearly marked
- Default values are safe for development

## Links

- **Main Configuration**: [config.yaml.template](config.yaml.template)
- **System Architecture**: [../../docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)
- **Ingestion Settings**: [../ingestion/README.md](../ingestion/README.md)
- **Query Settings**: [../querying/README.md](../querying/README.md)
- **Interface Settings**: [../interface/README.md](../interface/README.md)