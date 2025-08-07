# DocQuest: Document Retrieval-Augmented Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A local RAG pipeline that **quests through your document collections** to find answers using vector search and AI. DocQuest embarks on intelligent journeys through personal files, team folders, or any local document repository, discovering the information you seek with natural language queries.

## 🚀 Quick Start

```bash
# Install DocQuest package
pip install -e .

# Use convenient commands
dq-ingest ./documents/          # Process documents
dq-ask "What are the reports?"   # Query documents
dq-serve                        # Start API server
```

## 📋 Features

- **Document Ingestion**: PDF, DOCX, PPTX, TXT, XLSX support
- **Vector Search**: Semantic document retrieval
- **Knowledge Graph**: Entity and relationship extraction
- **Multi-Agent Framework**: Pluggable query processing
- **CLI Interface**: Simple command-line tools
- **API Server**: RESTful endpoints for integration

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/AbhiramDwivedi/docrag.git
cd docrag/backend

# Install in development mode
pip install -e .

# Create configuration
cp src/shared/config.yaml.template src/shared/config.yaml
# Edit config.yaml with your settings
```

## 📖 Usage

### Command Line Interface

```bash
# Process documents
dq-ingest /path/to/documents

# Query documents
dq-ask "What are the quarterly results?"

# Start file watcher
dq-watch /path/to/documents

# Rebuild knowledge graph
dq-kg-build

# Start API server
dq-serve
```

### Configuration

Edit `src/shared/config.yaml`:

```yaml
sync_root: "~/Documents/MyDocuments"
db_path: "data/docmeta.db"
vector_path: "data/vector.index"
knowledge_graph_path: "data/knowledge_graph.db"
embed_model: "sentence-transformers/all-MiniLM-L6-v2"
openai_api_key: "your-api-key-here"
```

## 🏗️ Architecture

DocQuest uses a modular architecture:

- **Ingestion**: Document extraction and processing
- **Storage**: Vector database and knowledge graph
- **Querying**: Multi-agent query processing
- **Interface**: CLI and API endpoints

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_kg_implementation_fixes.py
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Follow the coding guidelines in `.github/copilot-instructions.md`
2. Place tests in `backend/tests/`
3. Place demos in `backend/examples/`
4. Use conventional commit messages
5. Ensure CI passes before creating PRs
