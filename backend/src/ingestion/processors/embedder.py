from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np
from typing import List, Optional, Dict, Any

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

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model-specific configuration for text formatting and processing.
    
    Args:
        model_name: Name of the sentence transformer model
        
    Returns:
        Dictionary with model configuration including prefix requirements
    """
    # Model-specific configurations
    model_configs = {
        # E5 models require specific prefixes for queries and passages
        "intfloat/e5-base-v2": {
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "requires_prefixes": True,
            "pooling": "mean"
        },
        "intfloat/e5-small-v2": {
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "requires_prefixes": True,
            "pooling": "mean"
        },
        "intfloat/e5-large-v2": {
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "requires_prefixes": True,
            "pooling": "mean"
        },
        # BGE models work well with no prefixes but benefit from specific instructions
        "BAAI/bge-small-en-v1.5": {
            "query_prefix": "",
            "passage_prefix": "",
            "requires_prefixes": False,
            "pooling": "cls"
        },
        "BAAI/bge-base-en-v1.5": {
            "query_prefix": "",
            "passage_prefix": "",
            "requires_prefixes": False,
            "pooling": "cls"
        },
        "BAAI/bge-large-en-v1.5": {
            "query_prefix": "",
            "passage_prefix": "",
            "requires_prefixes": False,
            "pooling": "cls"
        },
        # GTE models work well without prefixes
        "thenlper/gte-base": {
            "query_prefix": "",
            "passage_prefix": "",
            "requires_prefixes": False,
            "pooling": "mean"
        },
        "thenlper/gte-small": {
            "query_prefix": "",
            "passage_prefix": "",
            "requires_prefixes": False,
            "pooling": "mean"
        },
        "thenlper/gte-large": {
            "query_prefix": "",
            "passage_prefix": "",
            "requires_prefixes": False,
            "pooling": "mean"
        }
    }
    
    # Default configuration for unknown models (backward compatibility)
    default_config = {
        "query_prefix": "",
        "passage_prefix": "",
        "requires_prefixes": False,
        "pooling": "mean"
    }
    
    return model_configs.get(model_name, default_config)

def format_texts_for_model(texts: List[str], model_name: str, text_type: str = "passage") -> List[str]:
    """Format texts according to model-specific requirements.
    
    Args:
        texts: List of texts to format
        model_name: Name of the embedding model
        text_type: Type of text - "query" or "passage"
        
    Returns:
        List of formatted texts with appropriate prefixes
    """
    config = get_model_config(model_name)
    
    if not config["requires_prefixes"]:
        return texts
    
    if text_type == "query":
        prefix = config["query_prefix"]
    elif text_type == "passage":
        prefix = config["passage_prefix"]
    else:
        prefix = config["passage_prefix"]  # Default to passage prefix
    
    return [prefix + text for text in texts]

def embed_texts(texts: List[str], model_name: str, normalize: bool = True, text_type: str = "passage") -> np.ndarray:
    """Generate embeddings for texts with optional normalization and model-specific formatting.
    
    Args:
        texts: List of texts to embed
        model_name: Name of the sentence transformer model
        normalize: Whether to normalize embeddings for cosine similarity (default: True)
        text_type: Type of text - "query" or "passage" for model-specific formatting
        
    Returns:
        Embeddings array, normalized if normalize=True
    """
    # Format texts according to model requirements
    formatted_texts = format_texts_for_model(texts, model_name, text_type)
    
    model = get_model(model_name)
    embeddings = model.encode(formatted_texts, batch_size=32, show_progress_bar=False)
    
    if normalize:
        embeddings = normalize_embeddings(embeddings)
    
    return embeddings
