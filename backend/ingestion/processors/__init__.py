"""Text processing and chunking components."""

from .chunker import chunk_text
from .embedder import embed_texts, get_model

__all__ = ['chunk_text', 'embed_texts', 'get_model']