"""Performance smoke tests for retrieval system."""

import time
import json
import sys
from pathlib import Path
from typing import Dict, Any

import pytest

# Add backend src to path and import build script
backend_root = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_root / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_test_artifacts import dummy_embed_texts
from test_golden_retrieval import search_vector_index, search_fts5_index


class TestPerformanceSmoke:
    """Performance smoke tests with generous bounds."""
    
    @pytest.fixture(scope="class")
    def test_artifacts_path(self):
        """Path to test artifacts."""
        return Path(__file__).parent.parent / "tests/fixtures/artifacts_v1"
    
    @pytest.fixture(scope="class")
    def artifacts_exist(self, test_artifacts_path):
        """Check if test artifacts exist."""
        vector_path = test_artifacts_path / "vector.index"
        db_path = test_artifacts_path / "docmeta.db"
        metadata_path = test_artifacts_path / "metadata.json"
        
        if not all(p.exists() for p in [vector_path, db_path, metadata_path]):
            pytest.skip("Test artifacts not found. Run 'python scripts/build_test_artifacts.py' first.")
        
        return True
    
    def test_artifact_size_bounds(self, test_artifacts_path, artifacts_exist):
        """Test that artifact sizes are reasonable for a small test corpus."""
        # Check file sizes
        vector_path = test_artifacts_path / "vector.index"
        db_path = test_artifacts_path / "docmeta.db"
        metadata_path = test_artifacts_path / "metadata.json"
        
        vector_size = vector_path.stat().st_size
        db_size = db_path.stat().st_size
        metadata_size = metadata_path.stat().st_size
        
        # Vector index should be reasonable size for small corpus
        # 17 chunks * 384 dimensions * 4 bytes + overhead
        expected_vector_size = 17 * 384 * 4  # ~26KB
        assert vector_size < expected_vector_size * 2, f"Vector index too large: {vector_size} bytes"
        
        # Database should be reasonable
        assert db_size < 500_000, f"Database too large: {db_size} bytes"  # 500KB limit
        
        # Metadata should be small
        assert metadata_size < 10_000, f"Metadata too large: {metadata_size} bytes"  # 10KB limit
        
        # Total artifact size should be under 1MB
        total_size = vector_size + db_size + metadata_size
        assert total_size < 1_000_000, f"Total artifacts too large: {total_size} bytes"
    
    def test_embedding_generation_latency(self):
        """Test embedding generation latency."""
        test_texts = [
            "This is a short test document.",
            "Here is another document with different content.",
            "The third document contains various words and phrases."
        ]
        
        start_time = time.time()
        embeddings = dummy_embed_texts(test_texts)
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Should be very fast for dummy embeddings
        assert latency < 1.0, f"Embedding generation too slow: {latency:.3f}s"
        
        # Check output shape
        assert embeddings.shape == (3, 384)
    
    def test_vector_search_latency(self, test_artifacts_path, artifacts_exist):
        """Test vector search latency."""
        vector_path = test_artifacts_path / "vector.index"
        db_path = test_artifacts_path / "docmeta.db"
        
        query = "test query for performance measurement"
        
        # Warm up
        search_vector_index(query, vector_path, db_path, k=10)
        
        # Measure latency
        start_time = time.time()
        results = search_vector_index(query, vector_path, db_path, k=10)
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Should be fast for small corpus (generous bound)
        assert latency < 2.0, f"Vector search too slow: {latency:.3f}s"
        
        # Should return results
        assert len(results) > 0
    
    def test_lexical_search_latency(self, test_artifacts_path, artifacts_exist):
        """Test lexical search latency."""
        db_path = test_artifacts_path / "docmeta.db"
        
        query = "Acme"
        
        # Warm up
        search_fts5_index(query, db_path, k=10)
        
        # Measure latency
        start_time = time.time()
        results = search_fts5_index(query, db_path, k=10)
        end_time = time.time()
        
        latency = end_time - start_time
        
        if results:  # Only test if FTS5 is available
            # Should be very fast for small corpus
            assert latency < 1.0, f"Lexical search too slow: {latency:.3f}s"
            assert len(results) > 0
        else:
            pytest.skip("FTS5 not available")
    
    def test_multiple_queries_latency(self, test_artifacts_path, artifacts_exist):
        """Test latency for multiple queries."""
        vector_path = test_artifacts_path / "vector.index"
        db_path = test_artifacts_path / "docmeta.db"
        
        queries = [
            "what is Acme?",
            "tell me about BOLT",
            "find Globex information",
            "Contoso partnership details",
            "Initech performance metrics"
        ]
        
        start_time = time.time()
        
        for query in queries:
            results = search_vector_index(query, vector_path, db_path, k=5)
            assert len(results) > 0
        
        end_time = time.time()
        total_latency = end_time - start_time
        avg_latency = total_latency / len(queries)
        
        # Average latency should be reasonable
        assert avg_latency < 1.0, f"Average query latency too high: {avg_latency:.3f}s"
        assert total_latency < 5.0, f"Total latency too high: {total_latency:.3f}s"
    
    def test_index_loading_time(self, test_artifacts_path, artifacts_exist):
        """Test index loading time."""
        import faiss
        
        vector_path = test_artifacts_path / "vector.index"
        
        start_time = time.time()
        index = faiss.read_index(str(vector_path))
        end_time = time.time()
        
        loading_time = end_time - start_time
        
        # Should load quickly for small index
        assert loading_time < 1.0, f"Index loading too slow: {loading_time:.3f}s"
        
        # Verify index loaded correctly
        assert index.ntotal > 0
        assert index.d == 384
    
    def test_database_connection_time(self, test_artifacts_path, artifacts_exist):
        """Test database connection and query time."""
        import sqlite3
        
        db_path = test_artifacts_path / "docmeta.db"
        
        start_time = time.time()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]
        conn.close()
        end_time = time.time()
        
        query_time = end_time - start_time
        
        # Should be very fast
        assert query_time < 0.5, f"Database query too slow: {query_time:.3f}s"
        assert count > 0
    
    def test_memory_usage_bounds(self, test_artifacts_path, artifacts_exist):
        """Test that memory usage is reasonable."""
        import psutil
        import os
        import faiss
        import sqlite3
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Load index and database
        vector_path = test_artifacts_path / "vector.index"
        db_path = test_artifacts_path / "docmeta.db"
        
        index = faiss.read_index(str(vector_path))
        conn = sqlite3.connect(str(db_path))
        
        # Perform some operations
        import numpy as np
        query_embedding = dummy_embed_texts(["test query"])[0]
        scores, indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), 5)
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        cursor.fetchone()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        conn.close()
        
        # Memory increase should be reasonable for small corpus
        # Allow up to 100MB increase (very generous)
        assert memory_increase < 100_000_000, f"Memory usage too high: {memory_increase} bytes"
    
    def test_concurrent_query_performance(self, test_artifacts_path, artifacts_exist):
        """Test performance with simulated concurrent queries."""
        vector_path = test_artifacts_path / "vector.index"
        db_path = test_artifacts_path / "docmeta.db"
        
        queries = [
            "Acme corporation",
            "BOLT protocol", 
            "Globex investments",
            "Contoso solutions",
            "Initech automation"
        ] * 4  # 20 total queries
        
        start_time = time.time()
        
        # Simulate concurrent queries (sequential for simplicity)
        for query in queries:
            results = search_vector_index(query, vector_path, db_path, k=3)
            assert len(results) > 0
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle multiple queries efficiently
        queries_per_second = len(queries) / total_time
        assert queries_per_second > 5, f"Query throughput too low: {queries_per_second:.1f} qps"
    
    def test_percentile_latency_bounds(self, test_artifacts_path, artifacts_exist):
        """Test p95 latency bounds."""
        vector_path = test_artifacts_path / "vector.index"
        db_path = test_artifacts_path / "docmeta.db"
        
        query = "performance test query"
        latencies = []
        
        # Run multiple queries to get latency distribution
        for _ in range(20):
            start_time = time.time()
            results = search_vector_index(query, vector_path, db_path, k=5)
            end_time = time.time()
            latencies.append(end_time - start_time)
            assert len(results) > 0
        
        # Calculate p95 latency
        latencies.sort()
        p95_latency = latencies[int(0.95 * len(latencies))]
        
        # P95 should be under generous bound
        assert p95_latency < 3.0, f"P95 latency too high: {p95_latency:.3f}s"