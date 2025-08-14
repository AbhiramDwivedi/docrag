from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np

@lru_cache
def get_model(name):
    return SentenceTransformer(name)

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings for cosine similarity.
    
    Args:
        embeddings: 2D array of embeddings to normalize
        
    Returns:
        Normalized embeddings with unit length
    """
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def embed_texts(texts, model_name, normalize=True):
    """Generate embeddings for texts with optional normalization.
    
    Args:
        texts: List of texts to embed
        model_name: Name of the sentence transformer model
        normalize: Whether to normalize embeddings for cosine similarity (default: True)
        
    Returns:
        Embeddings array, normalized if normalize=True
    """
    model = get_model(model_name)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    
    if normalize:
        embeddings = normalize_embeddings(embeddings)
    
    return embeddings
