# Contributing to DocQuest

Thanks for your interest in contributing! This guide will help you get started.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd docquest
   ```

2. **Setup development environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   cd backend
   pip install -e .[dev]  # Install with development dependencies
   cd ..
   python setup.py  # Post-clone setup
   ```

3. **Configure for development**
   ```bash
   # Update config/config.yaml with your document folder path
   # Add your OpenAI API key
   python setup_openai.py
   ```

## Coding Guidelines

Please follow the guidelines in `.github/copilot-instructions.md`:

- **Python ≥ 3.11** with type hints and dataclasses
- **Black formatting** (88 char line length)
- **Google-style docstrings**
- **pathlib.Path** for file operations
- **pytest** for testing

## Project Structure

```
├── api/                 # FastAPI web interface
├── cli/                 # Command-line tools
├── config/              # Configuration management
├── ingest/              # Document processing pipeline
├── watcher/             # File system monitoring
└── tests/               # Unit tests (TODO)
```

## Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding guidelines
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**
   ```bash
   # Test imports
   python -c "from backend.ingestion.extractor import extract_text"
   
   # Test ingestion (with test documents)
   python -m backend.ingestion.pipeline --mode full
   
   # Test CLI
   python -m interface.cli.ask "test query"
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Add: brief description of your change"
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**

## Security Considerations

- **Never commit API keys** - they're in .gitignore for a reason
- **Use the template files** for configuration examples
- **Test with sample documents** rather than sensitive personal data

## Common Development Tasks

### Adding New File Types
1. Update `ingest/extractor.py` with new extraction method
2. Add file extension to supported types
3. Test with sample files
4. Update documentation

### Improving Vector Search
1. Modify `ingest/vector_store.py` for index improvements
2. Update `ingest/embed.py` for embedding model changes
3. Test search quality with sample queries

### Adding CLI Features
1. Extend `cli/ask.py` for new functionality
2. Add argument parsing if needed
3. Update help text and documentation

## Testing

Currently testing is manual, but we welcome contributions for:
- Unit tests with pytest
- Integration tests for document processing
- Performance benchmarks

## Documentation

- Update README.md for user-facing changes
- Update this CONTRIBUTING.md for developer changes
- Add docstrings for new functions/classes
- Update .github/copilot-instructions.md for coding standards

## Questions?

- Check existing issues and discussions
- Create a new issue for bugs or feature requests
- Follow the project's coding guidelines

Thank you for contributing! 🎉
