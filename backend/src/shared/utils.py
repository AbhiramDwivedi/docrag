"""Shared utilities for DocQuest backend."""

from pathlib import Path
from typing import List, Dict, Any
import hashlib


def get_file_hash(path: Path) -> str:
    """Generate a unique hash for a file path."""
    return hashlib.sha1(str(path).encode()).hexdigest()[:10]


def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path("data")


def get_default_paths() -> Dict[str, str]:
    """Get default file paths for DocQuest data."""
    from shared.config import get_settings
    settings = get_settings()
    
    return {
        "vector_path": str(settings.vector_path),
        "db_path": str(settings.db_path),
        "knowledge_graph_path": str(settings.knowledge_graph_path)
    }