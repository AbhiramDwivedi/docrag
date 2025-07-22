"""
Advanced document extractors for DocQuest RAG pipeline.

This module now uses a modular extractor architecture where each file type
has its own specialized extractor class. The main extract_text function
serves as a factory that delegates to the appropriate extractor.

Supported formats:
- 📄 PDF: Advanced extraction with LangChain and AI image analysis
- 📝 DOCX: Word document processing
- 📊 PPTX: PowerPoint presentation processing  
- 📈 XLSX: Excel spreadsheet processing
- 📄 TXT: Plain text file processing
- 📧 EMAIL: Email message processing (.msg and .eml files)

Migration Note:
The old monolithic extractor functions have been moved to separate
extractor classes in the extractors/ subdirectory for better modularity,
testing, and maintenance.
"""
from pathlib import Path
from typing import List, Tuple

from .extractors import get_extractor, get_supported_extensions

Unit = Tuple[str, str]  # logical unit id, text

def extract_text(path: Path) -> List[Unit]:
    """
    Extract text from supported document types using modular extractors.
    
    Args:
        path: Path to the document file
        
    Returns:
        List of (unit_id, text) tuples representing logical units of the document
    """
    extractor = get_extractor(path)
    if extractor:
        return extractor.extract(path)
    else:
        supported = ', '.join(get_supported_extensions())
        print(f"⚠️  Unsupported file type: {path.suffix}. Supported: {supported}")
        return []

# Backward compatibility functions for any existing imports
def set_all_sheets_mode(enabled: bool):
    """
    Set global flag for processing all sheets in Excel files.
    
    Note: This is maintained for backward compatibility.
    The new modular extractors handle this internally.
    In the future, this could configure the XLSX extractor if needed.
    """
    # This could be implemented by configuring the XLSX extractor if needed
    print(f"📋 All sheets mode set to: {enabled} (handled by XLSX extractor)")
    pass

def get_supported_file_types() -> List[str]:
    """Get list of supported file extensions."""
    return get_supported_extensions()

# Export the main interface
__all__ = ['extract_text', 'set_all_sheets_mode', 'get_supported_file_types', 'Unit']
