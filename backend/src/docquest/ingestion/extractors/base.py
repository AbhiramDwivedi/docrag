"""
Base extractor interface for DocQuest document processing.

Defines the common interface that all document extractors must implement.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

Unit = Tuple[str, str]  # logical unit id, text

class BaseExtractor(ABC):
    """Abstract base class for document extractors."""
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of file extensions this extractor supports."""
        pass
    
    @abstractmethod
    def extract(self, path: Path) -> List[Unit]:
        """Extract text units from the given file path."""
        pass
    
    def can_extract(self, path: Path) -> bool:
        """Check if this extractor can handle the given file."""
        return path.suffix.lower() in self.supported_extensions
    
    def _log_error(self, path: Path, error: Exception) -> None:
        """Log extraction error in a consistent format."""
        print(f"Error extracting {path.suffix.upper()} {path}: {error}")
