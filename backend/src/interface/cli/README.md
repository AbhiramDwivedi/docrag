# Command-Line Interface

This directory implements the primary command-line interface for DocQuest, providing direct access to document querying capabilities with rich output formatting and citation support.

## Purpose

The CLI serves as the main interactive interface for querying indexed documents, designed for developers, researchers, and power users who need direct, scriptable access to the document retrieval system.

## Components

### [ask.py](ask.py)
Main CLI application for document querying:

**Core Features**:
- **Natural Language Queries**: Ask questions in plain English
- **Citation Support**: Automatic source attribution with file paths and unit IDs
- **Context Display**: Show relevant document chunks used for answers
- **Multiple Output Formats**: Text, JSON, and markdown formatting
- **Progress Tracking**: Real-time feedback during processing
- **Error Handling**: Graceful handling of edge cases and failures

**Key Functions**:
```python
def main(question: str, verbose: bool = False)  # Main entry point
def process_query(question: str) -> QueryResult  # Query processing
def format_response(result: QueryResult) -> str  # Output formatting
```

## Usage Patterns

### Basic Querying
```bash
# Simple question answering
python -m backend.src.interface.cli.ask "What are the project requirements?"

# Multi-word questions (automatic quoting)
python -m backend.src.interface.cli.ask What is the system architecture?
```

### Advanced Options
```bash
# Verbose output with processing details
python -m backend.src.interface.cli.ask "Show me the budget data" --verbose

# JSON output for programmatic use
python -m backend.src.interface.cli.ask "List the key findings" --format json

# Pipe input for batch processing
echo "What are the risks?" | python -m backend.src.interface.cli.ask
```

### Integration Examples
```bash
# Save results to file
python -m backend.src.interface.cli.ask "Summarize the report" > summary.txt

# Process multiple questions
for q in "What is the budget?" "What are the timelines?" "Who are the stakeholders?"; do
    echo "Q: $q"
    python -m backend.src.interface.cli.ask "$q"
    echo "---"
done

# Generate documentation
python -m backend.src.interface.cli.ask "What are the API endpoints?" \
    --format markdown >> api_docs.md
```

## Output Formats

### Default Text Format
```
Answer: The project requirements include data processing capabilities, 
real-time querying, and multi-format document support.

Sources:
- requirements.pdf:page_2
- project_spec.docx:section_1
- technical_notes.txt:line_45
```

### JSON Format
```json
{
    "question": "What are the requirements?",
    "answer": "The project requirements include...",
    "sources": [
        {"file": "requirements.pdf", "unit": "page_2", "confidence": 0.92},
        {"file": "project_spec.docx", "unit": "section_1", "confidence": 0.87}
    ],
    "processing_time": 1.23,
    "context_chunks": 5
}
```

### Markdown Format
```markdown
## Query: What are the requirements?

**Answer:** The project requirements include data processing capabilities, 
real-time querying, and multi-format document support.

### Sources
- [requirements.pdf:page_2](file://requirements.pdf)
- [project_spec.docx:section_1](file://project_spec.docx)
- [technical_notes.txt:line_45](file://technical_notes.txt)
```

## Configuration

### CLI Settings in `../../shared/config.yaml`
```yaml
cli:
  default_format: "text"              # Default output format
  max_context_length: 8000           # Maximum context for LLM
  citation_style: "file:unit"         # Citation format
  show_confidence: false             # Display confidence scores
  color_output: true                 # Colored terminal output
  progress_bar: true                 # Show processing progress
```

### Environment Variables
```bash
export DOCQUEST_VERBOSE=1           # Enable verbose mode by default
export DOCQUEST_FORMAT=json         # Set default output format
export DOCQUEST_NO_COLOR=1          # Disable colored output
```

## Error Handling

### Common Scenarios

**No Relevant Documents Found**:
```
No relevant information found for your query.

Suggestions:
- Try rephrasing your question
- Check if documents are properly indexed
- Use broader search terms
```

**Configuration Issues**:
```
Error: OpenAI API key not configured.

To fix this:
1. Add your API key to config/config.yaml
2. Set OPENAI_API_KEY environment variable
3. Run: python setup_openai.py
```

**Index Not Found**:
```
Error: No document index found.

To create an index:
1. Run: python -m backend.src.ingestion.pipeline
2. Point sync_root to your document folder in config.yaml
3. Ensure documents are accessible
```

## Performance Characteristics

### Response Times
- **Typical queries**: 1-3 seconds
- **Complex queries**: 3-7 seconds  
- **First run**: Additional 2-5 seconds for model loading
- **Cached results**: Sub-second responses

### Resource Usage
- **Memory**: ~200MB baseline + document index size
- **CPU**: Burst usage during embedding generation
- **Disk I/O**: Index file access and result caching
- **Network**: OpenAI API calls only

## Development Patterns

### Adding New Features
```python
# Extend CLI with new options
@click.option('--new-feature', help='Description of new feature')
def main(question: str, new_feature: bool = False):
    if new_feature:
        # Implement new functionality
        pass
```

### Custom Output Formatters
```python
def custom_formatter(result: QueryResult) -> str:
    """Implement domain-specific output formatting"""
    return f"Custom format: {result.answer}"

# Register formatter
FORMATTERS['custom'] = custom_formatter
```

### Testing Patterns
```python
def test_cli_basic_query():
    """Test basic CLI functionality"""
    result = runner.invoke(cli, ["What is the test?"])
    assert result.exit_code == 0
    assert "answer" in result.output.lower()
```

## Integration Points

### Python API
```python
# Programmatic usage
from backend.src.interface.cli.ask import process_query

result = process_query("What are the requirements?")
print(f"Answer: {result.answer}")
print(f"Sources: {result.sources}")
```

### Shell Integration
```bash
# Add to ~/.bashrc or ~/.zshrc
alias dq='python -m backend.src.interface.cli.ask'
alias dqv='python -m backend.src.interface.cli.ask --verbose'

# Usage
dq "What is the project status?"
dqv "Show me the technical details"
```

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Extract Requirements
  run: |
    python -m backend.src.interface.cli.ask \
      "What are the system requirements?" \
      --format json > requirements.json
```

## Extension Guidelines

### Adding Query Types
1. **Specialized queries**: Domain-specific question patterns
2. **Multi-step queries**: Complex analysis workflows
3. **Interactive mode**: Conversational query sessions
4. **Batch processing**: Multiple questions in single invocation

### Output Enhancements
1. **Rich formatting**: Terminal colors and styling
2. **Interactive elements**: Clickable links and menus
3. **Progress visualization**: Real-time processing feedback
4. **Export options**: Direct integration with other tools

## Links

- **Interface Architecture**: [../README.md](../README.md)
- **Query Processing**: [../../querying/README.md](../../querying/README.md)
- **Configuration**: [../../shared/README.md](../../shared/README.md)
- **System Architecture**: [../../../docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md)