"""
TXT extractor for plain text files.

Features:
- UTF-8 text processing
- Simple and efficient
- Handles encoding gracefully
"""
from pathlib import Path
from typing import List

from .base import BaseExtractor, Unit

class TXTExtractor(BaseExtractor):
    """Plain text file extractor."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".txt"]
    
    def extract(self, path: Path) -> List[Unit]:
        """Extract text from plain text files."""
        try:
            print(f"📄 Processing text file: {path.name}")
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if content.strip():
                print(f"   ✅ Extracted {len(content)} characters")
                return [("document", content)]
            else:
                print(f"   ⚠️  File is empty")
                return []
                
        except Exception as e:
            self._log_error(path, e)
            return []
