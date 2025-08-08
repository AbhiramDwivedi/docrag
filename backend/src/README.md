# Source Code

This directory contains the main source code for DocQuest, organized following standard Python project layout. All application logic, from document ingestion to query processing, is implemented here.

## Architecture Overview

DocQuest follows a modular architecture with clear separation of concerns:

```
src/
├── ingestion/          # Document processing pipeline
├── querying/          # Query processing and agent framework  
├── interface/         # User interfaces (CLI, API)
├── shared/           # Configuration and shared utilities
├── cli/              # Legacy CLI components (being phased out)
└── main.py           # Application entry point
```

## Key Components

### [ingestion/](ingestion/)
Complete document processing pipeline from file extraction to vector storage:
- **extractors/** - Modular document extractors for PDF, DOCX, PPTX, XLSX, TXT, and email formats
- **processors/** - Text chunking and embedding generation
- **storage/** - Vector database and knowledge graph storage
- **pipeline.py** - Orchestrates the full ingestion workflow

### [querying/](querying/)
Query processing and multi-agent framework:
- **agents/** - Plugin-based agent system for extensible query processing
- **search/** - Vector search and retrieval components
- **api.py** - FastAPI web service endpoints

### [interface/](interface/)
User-facing interfaces:
- **cli/** - Command-line interface for document querying

### [shared/](shared/)
Configuration and utilities shared across components:
- Configuration management and validation
- Logging setup and utilities
- Common helper functions

## Data Flow

1. **Document Ingestion**: Files → extractors → processors → storage
2. **Query Processing**: User query → vector search → context retrieval → LLM processing → response
3. **Configuration**: shared/config.py manages all system settings

## Development Patterns

### Adding New File Types
1. Create extractor in `ingestion/extractors/`
2. Register in `ingestion/extractors/__init__.py`
3. Add tests in `../tests/`
4. Update documentation

### Extending Query Processing
1. Create plugin in `querying/agents/plugins/`
2. Implement Plugin interface
3. Register with agent framework
4. Add integration tests

### Configuration Changes
1. Update `shared/config.yaml.template`
2. Modify `shared/config.py` validation
3. Update dependent components
4. Document new settings

## Key Dependencies

- **Vector Storage**: FAISS + SQLite for hybrid storage
- **Embeddings**: sentence-transformers for semantic encoding
- **Document Processing**: Specialized libraries per format (PyMuPDF, python-docx, etc.)
- **LLM Integration**: OpenAI API for question answering
- **Web Framework**: FastAPI for API endpoints

## Links

- **Architecture Details**: [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)
- **Configuration**: [shared/README.md](shared/README.md)
- **Testing**: [../tests/](../tests/)
- **Documentation**: [../docs/](../docs/)