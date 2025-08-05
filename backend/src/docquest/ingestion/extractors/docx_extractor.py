"""
DOCX extractor for Word documents.

Features:
- Paragraph-based text extraction
- Preserves document structure
- Handles formatting gracefully
"""
from pathlib import Path
from typing import List

from .base import BaseExtractor, Unit

try:
    import docx
except ImportError:
    docx = None

class DOCXExtractor(BaseExtractor):
    """Word document extractor using python-docx."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".docx"]
    
    def extract(self, path: Path) -> List[Unit]:
        """Extract text from DOCX using python-docx."""
        try:
            import docx
            
            print(f"📝 Processing Word document: {path.name}")
            
            doc = docx.Document(str(path))
            text_parts: List[str] = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            
            full_text = "\n".join(text_parts)
            
            if full_text.strip():
                print(f"   ✅ Extracted {len(full_text)} characters")
                return [("document", full_text)]
            else:
                print(f"   ⚠️  No text content found")
                return []
                
        except Exception as e:
            self._log_error(path, e)
            return []
            return []
