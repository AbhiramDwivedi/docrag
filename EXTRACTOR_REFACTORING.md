# Modular Extractor Architecture

## Overview

DocQuest now uses a modular extractor architecture where each document type has its own specialized extractor class. This design provides better maintainability, testability, and extensibility compared to the previous monolithic approach.

## Architecture Benefits

### ✅ **Maintainability**
- Each file type is self-contained in its own extractor class
- Clear separation of concerns between different document formats
- Easier to modify or debug specific extractors without affecting others

### ✅ **Testability** 
- Individual extractors can be unit tested in isolation
- Mock testing easier with clear interfaces
- Focused test suites for each document type

### ✅ **Extensibility**
- Adding new file types requires only creating a new extractor class
- Factory pattern makes registration automatic
- Plugin-like architecture for future enhancements

### ✅ **Error Isolation**
- Issues with one file type don't affect others
- Graceful degradation when specific extractors fail
- Better error reporting and handling

## New Architecture

### Base Architecture
- **`extractors/base.py`**: Abstract `BaseExtractor` class defining the common interface
- **`extractors/__init__.py`**: Factory functions and extractor registry

### Specialized Extractors
- **`extractors/pdf_extractor.py`**: Advanced PDF processing with LangChain + AI image analysis
- **`extractors/docx_extractor.py`**: Word document text extraction
- **`extractors/pptx_extractor.py`**: PowerPoint slide-by-slide processing
- **`extractors/xlsx_simple_extractor.py`**: Excel spreadsheet processing (simplified, error-resistant)
- **`extractors/txt_extractor.py`**: Plain text file processing

### Main Interface
- **`extractor.py`**: Simplified factory that delegates to appropriate extractors
- Maintains backward compatibility with existing imports

## Key Features

### Modular Design
- Each file type has its own dedicated extractor class
- Clear separation of concerns
- Easy to add new file type support
- Independent testing of each extractor

### Enhanced PDF Processing (Preserved)
- LangChain integration with UnstructuredPDFLoader and PyMuPDFLoader fallbacks
- GPT-4 Vision image analysis for diagrams and charts
- 7-layer intelligent image filtering system
- Adaptive rate limiting to prevent OpenAI throttling
- Per-page API limits (3 images max per page)
- Hash-based image deduplication

### Factory Pattern
```python
from backend.ingestion.extractors import get_extractor

extractor = get_extractor(Path("document.pdf"))
if extractor:
    units = extractor.extract(Path("document.pdf"))
```

### Backward Compatibility
- Existing code using `from backend.ingestion.extractor import extract_text` continues to work
- API remains the same: `extract_text(path) -> List[Unit]`

## Supported File Types
- `.pdf` - Advanced processing with AI image analysis
- `.docx` - Word documents
- `.pptx` - PowerPoint presentations  
- `.xlsx`, `.xls` - Excel spreadsheets
- `.txt` - Plain text files

## Benefits Achieved

1. **Maintainability**: Each extractor is self-contained and easier to modify
2. **Testability**: Individual extractors can be unit tested in isolation
3. **Extensibility**: Adding new file types requires only creating a new extractor class
4. **Error Isolation**: Issues with one file type don't affect others
5. **Code Clarity**: Smaller, focused classes are easier to understand
6. **Preserved Features**: All enhanced PDF processing capabilities retained

## Migration Notes

- Old `extractor.py` backed up as `extractor_old.py`
- New modular architecture is fully backward compatible
- All existing functionality preserved
- Enhanced error handling and graceful degradation

## Testing Status

✅ Modular architecture tested and working  
✅ Factory pattern tested  
✅ Backward compatibility verified  
✅ Import structure validated  

The refactoring is complete and ready for use!
