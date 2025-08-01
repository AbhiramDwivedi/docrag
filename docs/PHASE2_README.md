# DocQuest Phase 2: Enhanced Metadata + Database Schema

## Overview

Phase 2 transforms DocQuest from basic semantic search into an intelligent document analysis system with rich metadata capabilities. This enables sophisticated queries like "Find emails from John last week" and "Show newest Excel files larger than 1MB".

## New Capabilities

### 🔍 Advanced Query Types

**Email Analysis:**
- `"emails from john@example.com last week"`
- `"emails about budget meeting"`
- `"messages from team leads this month"`

**File Filtering:**
- `"PDF files larger than 1MB"`
- `"Excel files modified yesterday"`
- `"documents created this week"`

**Multi-Step Queries:**
- `"latest email about project Alpha and related files"`
- `"budget documents from last quarter"`

### 📊 Enhanced Database Schema

The enhanced schema includes:

- **Files Table**: Comprehensive file metadata (size, dates, types, paths)
- **Email Metadata Table**: Sender, recipients, subjects, dates, attachments
- **Efficient Indexing**: Fast queries on type, date, size, sender
- **JSON Support**: Flexible additional metadata storage

### 🔄 Backward Compatibility

Phase 2 maintains full backward compatibility:
- All Phase 1 queries continue to work
- Existing CLI and API interfaces unchanged  
- Graceful degradation when enhanced schema unavailable
- Safe migration path from basic to enhanced schema

## Installation & Migration

### Option 1: Fresh Installation
For new installations, enhanced functionality is enabled by default.

### Option 2: Upgrade from Phase 1
```bash
# Create backup and migrate existing database
python ingest/migrate_to_enhanced.py

# Use enhanced ingestion for new files
python ingest/enhanced_ingest.py --mode incremental
```

## Usage Examples

### Enhanced Ingestion
```bash
# Ingest with rich metadata capture
python ingest/enhanced_ingest.py --mode full

# Process specific file types with metadata
python ingest/enhanced_ingest.py --file-type pdf

# Migrate existing database
python ingest/enhanced_ingest.py --migrate
```

### Advanced Queries
```bash
# Email queries
python -m cli.ask "emails from alice@company.com last week"
python -m cli.ask "messages about project deadline"

# File filtering
python -m cli.ask "PDF files larger than 5MB"
python -m cli.ask "Excel files modified this month"

# Multi-step queries  
python -m cli.ask "latest email about budget and related documents"
```

## API Integration

The enhanced functionality integrates seamlessly with existing APIs:

```python
from agent.factory import create_default_agent

agent = create_default_agent()

# Enhanced queries automatically route to appropriate plugins
response = agent.process_query("emails from john last week")
print(response)

# Get reasoning trace for debugging
reasoning = agent.explain_reasoning()
print(reasoning)
```

## Database Schema Details

### Files Table
```sql
CREATE TABLE files (
    file_path TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_extension TEXT NOT NULL,
    file_size INTEGER,
    created_time REAL,
    modified_time REAL,
    accessed_time REAL,
    file_type TEXT,
    chunk_count INTEGER DEFAULT 0,
    ingestion_time REAL,
    metadata_json TEXT
);
```

### Email Metadata Table
```sql
CREATE TABLE email_metadata (
    file_path TEXT PRIMARY KEY,
    message_id TEXT,
    thread_id TEXT,
    sender_email TEXT,
    sender_name TEXT,
    recipients TEXT,  -- JSON array
    cc_recipients TEXT,  -- JSON array
    bcc_recipients TEXT,  -- JSON array
    subject TEXT,
    email_date REAL,
    has_attachments BOOLEAN DEFAULT 0,
    attachment_count INTEGER DEFAULT 0,
    conversation_index TEXT,
    importance TEXT,
    flags TEXT,  -- JSON array
    FOREIGN KEY (file_path) REFERENCES files (file_path)
);
```

## Plugin Architecture

### Enhanced Metadata Plugin

The metadata plugin now supports:
- **Email Analysis**: Sender/recipient filtering, subject search
- **Date Filtering**: Temporal queries with natural language
- **Size Filtering**: File size comparisons  
- **Type Filtering**: File type and extension filtering
- **Statistical Analysis**: Collection summaries and breakdowns

### Multi-Step Query Planning

The agent now coordinates multiple plugins:
- **Query Classification**: Intelligent routing to appropriate plugins
- **Multi-Plugin Execution**: Coordinate metadata and content searches
- **Response Synthesis**: Combine results from multiple sources
- **Reasoning Traces**: Transparent execution planning

## Performance Considerations

### Database Indexes
Efficient queries are enabled by indexes on:
- File extensions and types
- Modification and creation dates
- File sizes
- Email senders and dates
- Email subjects

### Query Optimization
- Metadata queries are fast (< 1 second)
- Large collections supported via pagination
- Efficient JSON parsing for flexible metadata
- Connection pooling for concurrent access

## Testing

Run the comprehensive test suite:
```bash
# Test enhanced functionality
python -m pytest tests/test_enhanced_metadata.py -v

# Test backward compatibility
python -m pytest tests/test_agent_framework.py -v

# Demo enhanced capabilities
python demo_phase2.py
```

## Troubleshooting

### Common Issues

**"No document database found"**
- Run `python ingest/enhanced_ingest.py` to create database
- Check database path in `config/config.py`

**"Plugin not found"**
- Ensure plugins are registered in `agent/factory.py`
- Check plugin import paths

**Migration errors**
- Backup created automatically before migration
- Check file permissions on database directory
- Verify sufficient disk space

### Debug Mode
```python
# Enable reasoning traces
agent = create_default_agent()
response = agent.process_query("your query")
reasoning = agent.explain_reasoning()
print(reasoning)
```

## What's Next: Phase 3

Phase 3 will add:
- **Document Relationship Analysis**: Cross-reference detection and clustering
- **Advanced LLM Planning**: Sophisticated multi-step reasoning
- **Comprehensive Reporting**: Rich analytical reports and summaries

## Contributing

See `CONTRIBUTING.md` for guidelines on contributing to Phase 2 enhancements.

## License

MIT License - see `LICENSE` file for details.