# DocQuest: Document Retrieval-Augmented Generation

<div align="center">
  <img src="assets/icon.png" alt="DocQuest Logo" width="150" height="150">
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A local RAG pipeline that **quests through your document collections** to find answers using vector search and AI. DocQuest embarks on intelligent journeys through personal files, team folders, or any local document repository, discovering the information you seek with natural language queries.

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd docrag
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. Navigate to backend directory
cd backend

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your config file
copy shared\config.yaml.template shared\config.yaml  # Windows
# cp shared/config.yaml.template shared/config.yaml  # macOS/Linux

# 5. Configure document folder and API key
# Edit shared/config.yaml:
#   - Update sync_root to your document folder
#   - Add your OpenAI API key

# 6. Index your documents
python -m ingestion.pipeline --mode full

# 7. Ask questions
python -m interface.cli.ask "What documents are available?"

# 8. Start web API (optional)
uvicorn querying.api:app --reload
```

## 📁 Project Structure

```
docrag/
├── .github/             # GitHub workflows and CI/CD
├── backend/             # ALL backend functionality
│   ├── data/            # Data storage (databases, indexes)
│   ├── docs/            # Documentation
│   │   ├── ARCHITECTURE.md  # System architecture and data flow
│   │   ├── EXCEL_PROCESSING.md # Excel processing details
│   │   └── PDF_PROCESSING.md # LangChain + AI PDF processing
│   ├── examples/        # Usage examples and demos
│   ├── ingestion/       # Document processing pipeline
│   │   ├── extractors/  # Modular document extractors
│   │   │   ├── base.py  # Abstract extractor interface
│   │   │   ├── pdf_extractor.py    # Advanced PDF + AI image analysis
│   │   │   ├── docx_extractor.py   # Word document processing
│   │   │   ├── pptx_extractor.py   # PowerPoint processing
│   │   │   ├── xlsx_simple_extractor.py # Excel processing
│   │   │   └── txt_extractor.py    # Plain text processing
│   │   ├── extractor.py # Main extraction interface (factory)
│   │   ├── chunker.py   # Text chunking with NLTK
│   │   ├── embed.py     # Sentence transformer embeddings
│   │   ├── storage/     # FAISS + SQLite storage
│   │   └── pipeline.py  # Main ingestion CLI
│   ├── interface/       # User interfaces
│   │   └── cli/         # Command-line interface
│   ├── querying/        # Search and retrieval
│   │   ├── agents/      # Intelligent agent framework
│   │   └── api.py       # FastAPI web interface
│   ├── shared/          # Configuration and utilities
│   ├── tests/           # All test files
│   ├── watcher/         # File system monitoring
│   └── requirements.txt # Python dependencies
├── README.md            # This file
└── package.json         # Node.js dependencies (minimal)
```

[📊 View detailed system architecture →](backend/docs/ARCHITECTURE.md)

## 🔧 Configuration

### Initial Setup
After cloning, create your configuration file:
```bash
# Copy the template
copy backend\shared\config.yaml.template backend\shared\config.yaml  # Windows
cp backend/shared/config.yaml.template backend/shared/config.yaml    # macOS/Linux
```

### Document Folder Setup
1. Choose any folder containing documents you want to search
2. Update `backend/shared/config.yaml`:
   ```yaml
   sync_root: "~/Documents/MyDocuments"
   # or any path like "C:/Work/ProjectDocs" or "~/Dropbox/Research"
   ```

### OpenAI API Key Setup
**Option 1: Direct config file (recommended)**
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Edit `backend/shared/config.yaml`:
   ```yaml
   openai_api_key: "your-actual-api-key-here"
   ```

**Option 2: Environment variable**
```bash
export OPENAI_API_KEY="your-api-key"
```

## 📄 Supported File Types

- **PDF**: **🚀 Enhanced with LangChain + AI image analysis**
  - Advanced text extraction via LangChain (fallback to PyMuPDF)
  - GPT-4 Vision analysis of images and diagrams
  - Architecture diagrams → structured markdown conversion
  - Smart image type detection and filtering
  - **Vector graphics support**: Renders pages to capture draw.io diagrams and flowcharts
  - [📄 See detailed PDF processing features →](docs/PDF_PROCESSING.md)
- **Word Documents**: .docx via python-docx
- **PowerPoint**: .pptx via python-pptx  
- **Excel**: .xlsx via pandas + openpyxl with **enhanced processing**
  - Smart sheet prioritization by meaningful names
  - Complete data extraction with relationship preservation
  - Empty sheet filtering and intelligent fallback strategies
  - Detailed processing logs and progress tracking
  - [📊 See detailed Excel processing features →](docs/EXCEL_PROCESSING.md)
- **Email Messages**: **🆕 .msg (Outlook) and .eml files**
  - Thread-level extraction maintaining conversation context
  - Complete metadata extraction (sender, recipients, dates)
  - Aggressive signature and quote removal for clean content
  - Rich attachment information and message type detection
  - [📧 See detailed email processing features →](docs/EMAIL_PROCESSING.md)
- **Text Files**: .txt plain text

### 🔄 Unsupported Formats → PDF Conversion

For unsupported file formats, **export to PDF** for optimal processing:

**Draw.io Diagrams**: 
- `.drawio` files → Export as PDF
- **Vector graphics fully supported**: Our enhanced PDF processor renders pages to capture diagrams, flowcharts, and technical schematics

**Other Formats**:
- Visio diagrams → Export as PDF
- CAD drawings → Export as PDF  
- Image files with text → Convert to PDF
- Legacy formats → Export/convert to PDF

**Why PDF?** Our PDF extractor features:
- Advanced text extraction with multiple fallback methods
- AI-powered image analysis for diagrams and charts
- Vector graphics rendering for draw.io and technical diagrams
- Intelligent filtering to focus on meaningful content

## 💡 Usage Examples

### Document Ingestion
```bash
# Navigate to backend directory first
cd backend

# Full re-index
python -m ingestion.pipeline --mode full

# Incremental update (default)
python -m ingestion.pipeline --mode incremental

# Process specific file types
python -m ingestion.pipeline --file-type pdf
python -m ingestion.pipeline --file-type xlsx
python -m ingestion.pipeline --file-type docx

# Process specific files with enhanced options
python -m ingestion.pipeline --file-type xlsx --target "quarterly_report.xlsx" --all-sheets
```

### Command Line Options
- `--mode {full,incremental}`: Processing mode (default: incremental)
- `--file-type FILE_TYPE`: Process only specific file types (pdf, xlsx, docx, pptx, txt, msg, eml)
- `--target TARGET`: Process specific file by name (useful with --all-sheets)
- `--all-sheets`: Process ALL sheets in Excel files (removes 15-sheet limit)

### Querying Documents
```bash
# Navigate to backend directory first
cd backend

# CLI queries
python -m interface.cli.ask "What is the PCI compliance scope?"
python -m interface.cli.ask "Show me budget information from Excel files"
python -m interface.cli.ask "What are the project requirements?"

# Web API
uvicorn querying.api:app --reload
# Then in another terminal:
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
- **Modular Architecture**: Separate extractor classes for each file type (PDF, DOCX, PPTX, XLSX, TXT)
- **Advanced PDF Processing**: LangChain integration with GPT-4 Vision for image analysis
- **Smart Image Filtering**: 7-layer filtering system to identify meaningful diagrams vs decorative images
- **Processing Limits**: Configurable file size and content limits per format
- **Error Handling**: Graceful handling of permission errors and malformed files
- **Progress Tracking**: Rich progress bars with file counts and timing
- **Deduplication**: SHA1-based chunk IDs prevent duplicate processing

## 🔐 Security

- API keys stored locally in `backend/shared/config.yaml` (git-ignored)
- Template configuration provided for safe sharing
- No sensitive data committed to version control
- Secure input prompts for API key setup

## 🚨 Important Notes

1. **First Run**: Copy `backend/shared/config.yaml.template` to `backend/shared/config.yaml` and configure it
2. **Document Path**: Use any local folder path containing your documents
3. **File Permissions**: Ensure read access to your document folders
4. **API Limits**: OpenAI API usage charges apply for question answering
5. **Security**: Your `backend/shared/config.yaml` is git-ignored and stays local

## 🛟 Troubleshooting

### Common Issues
- **"No module named 'backend'"**: Run commands from project root directory
- **"Permission denied"**: Check file access permissions in your document folder
- **"API key not configured"**: Set API key in `backend/shared/config.yaml`
- **"No relevant information found"**: Try different query phrasing or re-index documents

### File Processing Issues
- **Large files**: Processing limits vary by file type (see format-specific docs)
- **NLTK errors**: Punkt data downloaded automatically on first run
- **Import errors**: Ensure virtual environment is activated

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 DocQuest Contributors
