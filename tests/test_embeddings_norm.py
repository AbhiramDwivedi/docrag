"""Test embedding normalization and cosine similarity conversion."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add backend src to path
backend_root = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_root / "src"))


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings for cosine similarity."""
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def cosine_similarity_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between normalized embeddings using dot product."""
    # Embeddings should already be normalized
    return np.dot(embeddings1, embeddings2.T)


class TestEmbeddingNormalization:
    """Test vector normalization and cosine similarity."""
    
    def test_embedding_normalization(self):
        """Test that embeddings are properly normalized."""
        # Create sample embeddings
        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.1, 0.2, 0.3]
        ], dtype=np.float32)
        
        normalized = normalize_embeddings(embeddings)
        
        # Check that all vectors have unit length
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0, 1.0], decimal=6)
        
    def test_cosine_similarity_with_normalized_vectors(self):
        """Test cosine similarity computation with normalized vectors."""
        # Create sample vectors
        v1 = np.array([[1.0, 0.0, 0.0]])
        v2 = np.array([[0.0, 1.0, 0.0]])
        v3 = np.array([[1.0, 0.0, 0.0]])
        
        # These are already normalized
        embeddings = np.vstack([v1, v2, v3])
        
        # Compute similarity matrix
        similarity = cosine_similarity_matrix(embeddings, embeddings)
        
        # Check expected similarities
        assert similarity[0, 0] == pytest.approx(1.0)  # v1 with itself
        assert similarity[0, 1] == pytest.approx(0.0)  # v1 with v2 (orthogonal)
        assert similarity[0, 2] == pytest.approx(1.0)  # v1 with v3 (identical)
        assert similarity[1, 1] == pytest.approx(1.0)  # v2 with itself
        
    def test_normalization_preserves_relative_similarity(self):
        """Test that normalization preserves relative similarity ordering."""
        # Create vectors with known relationships
        base = np.array([1.0, 1.0, 0.0])
        similar = np.array([1.1, 1.1, 0.1])  # Similar to base
        different = np.array([0.0, 0.0, 1.0])  # Different from base
        
        embeddings = np.array([base, similar, different])
        normalized = normalize_embeddings(embeddings)
        
        # Compute similarities
        sim_matrix = cosine_similarity_matrix(normalized, normalized)
        
        # Base should be more similar to 'similar' than to 'different'
        sim_base_similar = sim_matrix[0, 1]
        sim_base_different = sim_matrix[0, 2]
        
        assert sim_base_similar > sim_base_different
        
    def test_deterministic_normalization(self):
        """Test that normalization is deterministic."""
        embeddings = np.random.rand(5, 10).astype(np.float32)
        
        # Normalize twice
        norm1 = normalize_embeddings(embeddings)
        norm2 = normalize_embeddings(embeddings)
        
        # Should be identical
        np.testing.assert_array_equal(norm1, norm2)
        
    def test_zero_vector_handling(self):
        """Test handling of zero vectors in normalization."""
        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],  # Zero vector
            [4.0, 5.0, 6.0]
        ], dtype=np.float32)
        
        # Should handle zero vectors gracefully (return zero or NaN)
        normalized = normalize_embeddings(embeddings)
        
        # Check non-zero vectors are normalized
        assert abs(np.linalg.norm(normalized[0]) - 1.0) < 1e-6
        assert abs(np.linalg.norm(normalized[2]) - 1.0) < 1e-6
        
        # Zero vector should remain zero or become NaN
        zero_norm = np.linalg.norm(normalized[1])
        assert zero_norm == 0.0 or np.isnan(zero_norm)