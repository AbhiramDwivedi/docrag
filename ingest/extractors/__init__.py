"""
Extractor modules for different file types.

This package provides specialized extractors for various document formats:
- PDF: Advanced extraction with LangChain and AI image analysis
- DOCX: Word document processing
- PPTX: PowerPoint presentation processing  
- XLSX: Excel spreadsheet processing
- TXT: Plain text file processing
- EMAIL: Email message processing (.msg and .eml)
"""

from .base import BaseExtractor, Unit
from .pdf_extractor import PDFExtractor
from .docx_extractor import DOCXExtractor
from .pptx_extractor import PPTXExtractor
from .xlsx_simple_extractor import XLSXExtractor
from .txt_extractor import TXTExtractor
from .email_extractor import EmailExtractor

from pathlib import Path
from typing import Dict, Type, Optional

# Registry of all available extractors
EXTRACTORS: Dict[str, Type[BaseExtractor]] = {
    '.pdf': PDFExtractor,
    '.docx': DOCXExtractor,
    '.pptx': PPTXExtractor,
    '.xlsx': XLSXExtractor,
    '.xls': XLSXExtractor,
    '.txt': TXTExtractor,
    '.msg': EmailExtractor,
    '.eml': EmailExtractor,
}

def get_extractor(file_path: Path) -> Optional[BaseExtractor]:
    """Get the appropriate extractor for a file based on its extension."""
    extension = file_path.suffix.lower()
    extractor_class = EXTRACTORS.get(extension)
    
    if extractor_class:
        return extractor_class()
    return None

def get_supported_extensions() -> list[str]:
    """Get list of all supported file extensions."""
    return list(EXTRACTORS.keys())

__all__ = [
    'BaseExtractor',
    'Unit', 
    'PDFExtractor',
    'DOCXExtractor',
    'PPTXExtractor',
    'XLSXExtractor',
    'TXTExtractor',
    'EmailExtractor',
    'get_extractor',
    'get_supported_extensions',
    'EXTRACTORS'
]
