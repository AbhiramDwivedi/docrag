# SP-RAG: SharePoint Retrieval-Augmented Generation

A local RAG pipeline that syncs SharePoint documents via OneDrive and enables natural language querying using vector search and OpenAI.

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd localfsmc
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure SharePoint sync path
# Edit config/config.yaml and update sync_root to your OneDrive SharePoint folder
# Example: "~/Company Name - SharePoint Site"

# 4. Setup OpenAI API key
python setup_openai.py

# 5. Index your documents
python -m ingest.ingest --mode full

# 6. Ask questions
python -m cli.ask "What documents are available?"

# 7. Start web API (optional)
uvicorn api.app:app --reload
```

## � Project Structure

```
├── api/                 # FastAPI web interface
├── cli/                 # Command-line interface
├── config/              # Configuration files
├── ingest/              # Document processing pipeline
│   ├── extractor.py     # Multi-format text extraction
│   ├── chunker.py       # Text chunking with NLTK
│   ├── embed.py         # Sentence transformer embeddings
│   ├── vector_store.py  # FAISS + SQLite storage
│   └── ingest.py        # Main ingestion CLI
└── watcher/             # File system monitoring
```

## 🔧 Configuration

### SharePoint Sync Setup
1. Sync your SharePoint document library via OneDrive
2. Update `config/config.yaml`:
   ```yaml
   sync_root: "~/Your Company - SharePoint Site"
   ```

### OpenAI API Key Setup
**Option 1: Secure setup (recommended)**
```bash
python setup_openai.py
```

**Option 2: Manual setup**
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Edit `config/config.yaml`:
   ```yaml
   openai_api_key: "your-actual-api-key-here"
   ```

**Option 3: Environment variable**
```bash
export OPENAI_API_KEY="your-api-key"
```

## 📄 Supported File Types

- **PDF**: Text extraction via PyMuPDF
- **Word Documents**: .docx via python-docx
- **PowerPoint**: .pptx via python-pptx  
- **Excel**: .xlsx via pandas + openpyxl
- **Text Files**: .txt plain text

## 💡 Usage Examples

### Document Ingestion
```bash
# Full re-index
python -m ingest.ingest --mode full

# Incremental update
python -m ingest.ingest --mode incremental

# Process specific file types
python -m ingest.ingest --file-type pdf
python -m ingest.ingest --file-type xlsx
```

### Querying Documents
```bash
# CLI queries
python -m cli.ask "What is the PCI compliance scope?"
python -m cli.ask "Show me budget information from Excel files"
python -m cli.ask "What are the project requirements?"

# Web API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What documents mention security requirements?"}'
```

## 🛠 Technical Details

### Vector Search
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Vector Store**: FAISS IndexFlatL2 with SQLite metadata
- **Chunking**: 800 characters with 150 character overlap
- **Search**: Retrieves top-8 relevant chunks for context

### Document Processing
- **Excel Limits**: 50MB max file size, 5 sheets max, 500 rows/sheet
- **Error Handling**: Graceful handling of permission errors and malformed files
- **Progress Tracking**: Rich progress bars with file counts and timing
- **Deduplication**: SHA1-based chunk IDs prevent duplicate processing

## 🔐 Security

- API keys stored locally in `config/config.yaml` (git-ignored)
- Template configuration provided for safe sharing
- No sensitive data committed to version control
- Secure input prompts for API key setup

## 🚨 Important Notes

1. **First Run**: Copy `config/config.yaml.template` to `config/config.yaml` if missing
2. **SharePoint Path**: Use the local OneDrive sync folder path, not SharePoint URLs
3. **File Permissions**: Ensure read access to SharePoint documents
4. **API Limits**: OpenAI API usage charges apply for question answering

## 🛟 Troubleshooting

### Common Issues
- **"No module named 'ingest'"**: Run commands from project root directory
- **"Permission denied"**: Check file access permissions in OneDrive
- **"API key not configured"**: Run `python setup_openai.py`
- **"No relevant information found"**: Try different query phrasing or re-index documents

### File Processing Issues
- **Excel files stuck**: Large files are automatically limited (50MB, 5 sheets)
- **NLTK errors**: Punkt data downloaded automatically on first run
- **Import errors**: Ensure virtual environment is activated

## 📜 License

This project follows the coding guidelines in `.github/copilot-instructions.md`.
