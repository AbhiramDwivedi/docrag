# Document Extractors

This directory contains modular document extractors for different file formats. Each extractor is specialized for a specific document type, providing isolated processing logic and format-specific optimizations.

## Architecture

The extractor system uses a **factory pattern** for routing documents to appropriate processors:

```python
# Factory automatically routes to correct extractor
extractor = ExtractorFactory.get_extractor(file_path)
chunks = extractor.extract(file_path)
```

## Available Extractors

### [base.py](base.py)
Abstract base class defining the extractor interface:
- Standardized `extract()` method signature
- Common error handling patterns
- Unit ID generation utilities

### [pdf_extractor.py](pdf_extractor.py) 🚀 **Enhanced**
Advanced PDF processing with AI capabilities:
- **LangChain integration** for robust text extraction
- **GPT-4 Vision analysis** of images and diagrams  
- **7-layer image filtering** to identify meaningful content
- **Vector graphics support** via page rendering
- Fallback to PyMuPDF for compatibility

### [docx_extractor.py](docx_extractor.py)
Microsoft Word document processing:
- Paragraph-by-paragraph extraction using python-docx
- Preserves document structure and formatting context
- Handles tables and embedded content

### [pptx_extractor.py](pptx_extractor.py)
PowerPoint presentation processing:
- Slide-by-slide text extraction
- Title and content separation
- Speaker notes inclusion

### [xlsx_simple_extractor.py](xlsx_simple_extractor.py) 📊 **Enhanced**
Excel spreadsheet processing with intelligent features:
- **Smart sheet prioritization** by meaningful names
- **Empty sheet filtering** and fallback strategies
- Complete data extraction with relationship preservation
- Configurable processing limits and progress tracking

### [txt_extractor.py](txt_extractor.py)
Plain text file processing:
- UTF-8 encoding with fallback detection
- Simple, reliable text extraction
- Minimal preprocessing for clean content

### [email_extractor.py](email_extractor.py) 📧 **New**
Email message processing:
- **Outlook (.msg)** and standard (.eml) support
- Thread-level extraction maintaining conversation context
- Metadata extraction (sender, recipients, dates)
- Signature and quote removal for clean content

## Factory Pattern

### [__init__.py](__init__.py)
Central factory for extractor routing:
```python
def get_extractor(file_path: str) -> BaseExtractor:
    """Routes file to appropriate extractor based on extension"""
```

Benefits:
- **Isolation**: Issues with one format don't affect others
- **Testability**: Each extractor can be unit tested independently  
- **Maintainability**: Clear separation of concerns per document type
- **Extensibility**: New formats added by creating new extractor classes

## Usage Patterns

### Direct Extractor Usage
```python
from backend.src.ingestion.extractors import PDFExtractor

extractor = PDFExtractor()
chunks = extractor.extract("document.pdf")
# Returns: List[Tuple[unit_id, text]]
```

### Factory-Based Usage (Recommended)
```python
from backend.src.ingestion.extractors import ExtractorFactory

extractor = ExtractorFactory.get_extractor("document.pdf")
chunks = extractor.extract("document.pdf")
```

### Pipeline Integration
```python
# Called automatically by pipeline.py
chunks = extract_text(file_path)  # Uses factory internally
```

## Extension Guide

### Adding New File Format Support

1. **Create Extractor Class**:
```python
from .base import BaseExtractor

class NewFormatExtractor(BaseExtractor):
    def extract(self, file_path: str) -> List[Tuple[str, str]]:
        # Implementation here
        return chunks
```

2. **Register in Factory**:
```python
# In __init__.py
EXTRACTORS = {
    '.newext': NewFormatExtractor,
    # ... existing extractors
}
```

3. **Add Tests**:
```python
# In ../../tests/test_new_format_extractor.py
def test_new_format_extraction():
    # Test implementation
```

4. **Update Documentation**:
- Add format details to this README
- Update processing documentation in [../../docs/](../../docs/)

### Enhancing Existing Extractors

- **Performance**: Optimize for specific document characteristics
- **Features**: Add format-specific metadata extraction  
- **Quality**: Improve text cleaning and structure preservation
- **Robustness**: Handle edge cases and malformed files

## Processing Characteristics

| Format | Complexity | AI Features | Metadata | Performance |
|--------|------------|-------------|----------|-------------|
| PDF    | High       | ✅ Vision   | Rich     | Variable    |
| XLSX   | Medium     | ❌          | Rich     | Fast        |
| DOCX   | Low        | ❌          | Basic    | Fast        |
| PPTX   | Low        | ❌          | Basic    | Fast        |
| Email  | Medium     | ❌          | Rich     | Fast        |
| TXT    | Minimal    | ❌          | None     | Very Fast   |

## Configuration

Key settings in `../../shared/config.yaml`:
- Processing limits per file type
- AI feature toggles for PDF processing
- Timeout and memory limits
- Quality vs. speed trade-offs

## Links

- **Base Architecture**: [../README.md](../README.md)
- **PDF Processing Details**: [../../docs/PDF_PROCESSING.md](../../docs/PDF_PROCESSING.md)
- **Excel Processing Details**: [../../docs/EXCEL_PROCESSING.md](../../docs/EXCEL_PROCESSING.md)
- **Email Processing Details**: [../../docs/EMAIL_PROCESSING.md](../../docs/EMAIL_PROCESSING.md)
- **Testing**: [../../tests/](../../tests/)