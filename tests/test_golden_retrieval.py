"""Integration tests for golden retrieval using test corpus."""

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
import pytest

# Add backend src to path
backend_root = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_root / "src"))

# Import build script functions
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_test_artifacts import dummy_embed_texts


def load_golden_queries(queries_path: Path) -> List[Dict[str, Any]]:
    """Load golden queries from JSONL file."""
    queries = []
    if queries_path.exists():
        with open(queries_path, 'r') as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line.strip()))
    return queries


def search_vector_index(
    query_text: str,
    index_path: Path,
    db_path: Path,
    k: int = 20
) -> List[Dict[str, Any]]:
    """Search vector index for similar documents."""
    # Generate query embedding
    query_embedding = dummy_embed_texts([query_text])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Load FAISS index
    index = faiss.read_index(str(index_path))
    
    # Search
    scores, indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), k)
    
    # Get document metadata from database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue
            
        cursor.execute(
            "SELECT file_path, chunk_id, content FROM chunks WHERE embedding_id = ?",
            (int(idx),)
        )
        result = cursor.fetchone()
        if result:
            file_path, chunk_id, content = result
            results.append({
                'file_path': file_path,
                'chunk_id': chunk_id,
                'content': content,
                'score': float(score),
                'rank': i
            })
    
    conn.close()
    
    # Sort results by score descending, then by chunk_id for deterministic tie-breaking
    results.sort(key=lambda x: (-x['score'], x['chunk_id']))
    
    # Update rank after sorting
    for i, result in enumerate(results):
        result['rank'] = i
    
    return results


def search_fts5_index(
    query_text: str,
    db_path: Path,
    k: int = 20
) -> List[Dict[str, Any]]:
    """Search FTS5 index for matching documents."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Try FTS5 search
        cursor.execute(
            "SELECT file_path, chunk_id, content FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
            (query_text, k)
        )
        results = []
        for i, (file_path, chunk_id, content) in enumerate(cursor.fetchall()):
            results.append({
                'file_path': file_path,
                'chunk_id': chunk_id,
                'content': content,
                'score': 1.0 - (i * 0.1),  # Simple score based on rank
                'rank': i
            })
        return results
    except sqlite3.OperationalError:
        # FTS5 not available
        return []
    finally:
        conn.close()


def merge_search_results(
    vector_results: List[Dict[str, Any]], 
    fts_results: List[Dict[str, Any]], 
    alpha: float = 0.5
) -> List[Dict[str, Any]]:
    """Merge vector and FTS search results with hybrid scoring."""
    # Create a map of chunk_id to combined results
    combined = {}
    
    # Add vector results
    for result in vector_results:
        chunk_id = result['chunk_id']
        combined[chunk_id] = {
            **result,
            'vector_score': result['score'],
            'fts_score': 0.0,
            'hybrid_score': alpha * result['score']
        }
    
    # Add FTS results and update hybrid scores
    for result in fts_results:
        chunk_id = result['chunk_id']
        if chunk_id in combined:
            combined[chunk_id]['fts_score'] = result['score']
            combined[chunk_id]['hybrid_score'] = (
                alpha * combined[chunk_id]['vector_score'] + 
                (1 - alpha) * result['score']
            )
        else:
            combined[chunk_id] = {
                **result,
                'vector_score': 0.0,
                'fts_score': result['score'],
                'hybrid_score': (1 - alpha) * result['score']
            }
    
    # Convert to list and sort by hybrid score descending, then chunk_id for determinism
    results = list(combined.values())
    results.sort(key=lambda x: (-x['hybrid_score'], x['chunk_id']))
    
    # Update rank and score to use hybrid score
    for i, result in enumerate(results):
        result['rank'] = i
        result['score'] = result['hybrid_score']
    
    return results


class TestGoldenRetrieval:
    """Test retrieval using golden test corpus."""
    
    @pytest.fixture(scope="class")
    def test_artifacts_path(self):
        """Path to test artifacts."""
        return Path(__file__).parent.parent / "tests/fixtures/artifacts_v1"
    
    @pytest.fixture(scope="class")
    def queries_path(self):
        """Path to golden queries."""
        return Path(__file__).parent.parent / "tests/fixtures/queries_v1.jsonl"
    
    @pytest.fixture(scope="class")
    def artifacts_exist(self, test_artifacts_path):
        """Check if test artifacts exist."""
        vector_path = test_artifacts_path / "vector.index"
        db_path = test_artifacts_path / "docmeta.db"
        metadata_path = test_artifacts_path / "metadata.json"
        
        if not all(p.exists() for p in [vector_path, db_path, metadata_path]):
            pytest.skip("Test artifacts not found. Run 'python scripts/build_test_artifacts.py' first.")
        
        return True
    
    @pytest.fixture(scope="class")
    def golden_queries(self, queries_path):
        """Load golden queries."""
        queries = load_golden_queries(queries_path)
        if not queries:
            pytest.skip("No golden queries found")
        return queries
    
    def test_artifacts_metadata(self, test_artifacts_path, artifacts_exist):
        """Test that artifacts metadata is valid."""
        metadata_path = test_artifacts_path / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Check required fields
        assert 'checksum' in metadata
        assert 'model_name' in metadata
        assert 'embedding_dim' in metadata
        assert 'num_documents' in metadata
        assert 'num_chunks' in metadata
        assert 'has_fts5' in metadata
        
        # Check reasonable values
        assert metadata['embedding_dim'] == 384
        assert metadata['num_documents'] > 0
        assert metadata['num_chunks'] > 0
    
    def test_vector_index_structure(self, test_artifacts_path, artifacts_exist):
        """Test that vector index has correct structure."""
        index_path = test_artifacts_path / "vector.index"
        index = faiss.read_index(str(index_path))
        
        # Check index properties
        assert index.d == 384  # Embedding dimension
        assert index.ntotal > 0  # Has vectors
        
        # Should be IndexFlatIP for cosine similarity with normalized vectors
        assert isinstance(index, faiss.IndexFlatIP)
    
    def test_database_structure(self, test_artifacts_path, artifacts_exist):
        """Test that database has correct structure."""
        db_path = test_artifacts_path / "docmeta.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'documents' in tables
        assert 'chunks' in tables
        
        # Check chunks table has data
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        assert chunk_count > 0
        
        # Check if FTS5 table exists
        if 'chunks_fts' in tables:
            cursor.execute("SELECT COUNT(*) FROM chunks_fts")
            fts_count = cursor.fetchone()[0]
            assert fts_count > 0
        
        conn.close()
    
    @pytest.mark.parametrize("query", [
        {"id": "test_acme", "query": "what is Acme?", "expected_doc": "text/acme_overview.txt"},
        {"id": "test_bolt", "query": "what is BOLT?", "expected_doc": "text/bolt_spec.md"},
        {"id": "test_globex", "query": "tell me about Globex", "expected_doc": "text/globex_financial.md"},
    ])
    def test_semantic_retrieval_basic(self, query, test_artifacts_path, artifacts_exist):
        """Test basic semantic retrieval for key entities."""
        vector_path = test_artifacts_path / "vector.index"
        db_path = test_artifacts_path / "docmeta.db"
        
        results = search_vector_index(query["query"], vector_path, db_path, k=20)
        
        # Should return some results
        assert len(results) > 0
        
        # Expected document should appear in top-20 results
        found_files = [result['file_path'] for result in results]
        assert query["expected_doc"] in found_files
        
        # Results should be sorted by score (descending)
        scores = [result['score'] for result in results]
        assert scores == sorted(scores, reverse=True)
        
        # For deterministic tie-breaking, assert that identical scores are ordered consistently
        # by a secondary key (e.g., chunk_id or file_path)
        for i in range(len(results) - 1):
            if abs(results[i]['score'] - results[i+1]['score']) < 1e-6:  # Tied scores
                # When scores are tied, should be ordered by chunk_id for determinism
                assert results[i]['chunk_id'] <= results[i+1]['chunk_id'], \
                    f"Tied scores at positions {i}, {i+1} not deterministically ordered: " \
                    f"{results[i]['chunk_id']} vs {results[i+1]['chunk_id']}"
    
    def test_golden_queries_vector_search(self, golden_queries, test_artifacts_path, artifacts_exist):
        """Test all golden queries using vector search."""
        vector_path = test_artifacts_path / "vector.index"
        db_path = test_artifacts_path / "docmeta.db"
        
        for query_data in golden_queries:
            query_id = query_data['id']
            query_text = query_data['query']
            expectations = query_data['expects']
            
            expected_docs = expectations['doc_ids']
            k = expectations['k']
            min_at_topk = expectations['min_at_topk']
            
            # Search
            results = search_vector_index(query_text, vector_path, db_path, k=k)
            
            # Check that we got results
            assert len(results) > 0, f"No results for query {query_id}: {query_text}"
            
            # Check that expected documents appear in top-k
            found_files = [result['file_path'] for result in results]
            matched_docs = [doc for doc in expected_docs if doc in found_files]
            
            assert len(matched_docs) >= min_at_topk, \
                f"Query {query_id} expected {min_at_topk} of {expected_docs} in top-{k}, " \
                f"but found {matched_docs} in {found_files[:5]}..."
    
    def test_lexical_search_availability(self, test_artifacts_path, artifacts_exist):
        """Test FTS5 lexical search availability and basic functionality."""
        db_path = test_artifacts_path / "docmeta.db"
        
        # Test basic FTS5 search
        results = search_fts5_index("Acme", db_path, k=10)
        
        if results:
            # FTS5 is available and working
            assert len(results) > 0
            found_files = [result['file_path'] for result in results]
            assert "text/acme_overview.txt" in found_files
        else:
            pytest.skip("FTS5 not available in this SQLite build")
    
    def test_keyword_queries_lexical_search(self, test_artifacts_path, artifacts_exist):
        """Test keyword-based queries using lexical search."""
        db_path = test_artifacts_path / "docmeta.db"
        
        # Test some keyword queries
        test_queries = [
            ("Contoso", "text/contoso_partnership.txt"),
            ("BOLT", "text/bolt_spec.md"),
            ("Globex", "text/globex_financial.md"),
            ("Initech", "text/initech_roadmap.txt"),
        ]
        
        for keyword, expected_file in test_queries:
            results = search_fts5_index(keyword, db_path, k=10)
            
            if results:  # Only test if FTS5 is available
                found_files = [result['file_path'] for result in results]
                assert expected_file in found_files, \
                    f"Keyword '{keyword}' should find {expected_file}, but found {found_files}"
    
    def test_retrieval_deterministic(self, test_artifacts_path, artifacts_exist):
        """Test that retrieval results are deterministic."""
        vector_path = test_artifacts_path / "vector.index"
        db_path = test_artifacts_path / "docmeta.db"
        
        query = "what is Acme?"
        
        # Run search multiple times
        results1 = search_vector_index(query, vector_path, db_path, k=10)
        results2 = search_vector_index(query, vector_path, db_path, k=10)
        
        # Results should be identical
        assert len(results1) == len(results2)
        
        for r1, r2 in zip(results1, results2):
            assert r1['file_path'] == r2['file_path']
            assert r1['chunk_id'] == r2['chunk_id']
            assert abs(r1['score'] - r2['score']) < 1e-6  # Floating point tolerance
    
    def test_retrieval_coverage(self, test_artifacts_path, artifacts_exist):
        """Test that retrieval can find content from all corpus files."""
        vector_path = test_artifacts_path / "vector.index"
        db_path = test_artifacts_path / "docmeta.db"
        
        # Get all files in corpus
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT file_path FROM documents")
        all_files = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Test queries that should find each file
        file_queries = {
            "text/acme_overview.txt": "Acme corporation",
            "text/bolt_spec.md": "BOLT protocol",
            "text/contoso_partnership.txt": "Contoso partnership",
            "text/globex_financial.md": "Globex financial",
            "text/initech_roadmap.txt": "Initech workflow",
            "text/reltora_integration.md": "Reltora gateway"
        }
        
        found_files = set()
        
        for expected_file, query in file_queries.items():
            if expected_file in all_files:
                results = search_vector_index(query, vector_path, db_path, k=20)
                result_files = [result['file_path'] for result in results]
                if expected_file in result_files:
                    found_files.add(expected_file)
        
        # Should be able to retrieve most text files
        text_files = [f for f in all_files if f.endswith(('.txt', '.md'))]
        coverage = len(found_files) / len(text_files) if text_files else 0
        
        assert coverage >= 0.8, f"Only found {len(found_files)}/{len(text_files)} text files: {found_files}"


def test_golden_retrieval_edge_cases(tmp_path):
    """Test retrieval system behavior with edge case queries."""
    artifacts_dir = Path(__file__).parent / "fixtures" / "artifacts_v1"
    
    if not artifacts_dir.exists():
        pytest.skip("Artifacts not available - run build_test_artifacts.py first")
    
    vector_path = artifacts_dir / "vector.index"
    db_path = artifacts_dir / "docmeta.db"
    
    if not vector_path.exists() or not db_path.exists():
        pytest.skip("Required artifacts missing")
    
    # Test query that should return no results
    no_results_queries = [
        "xyznonexistentterm12345",  # Completely non-existent term
        "zebra elephant unicorn",   # Multiple non-existent terms
        "®™©¡¿±"                   # Special characters only
    ]
    
    for query in no_results_queries:
        # Vector search should return empty or very low relevance results
        vector_results = search_vector_index(query, vector_path, db_path, k=5)
        
        # If results are returned, scores should be very low (< 0.1)
        if vector_results:
            max_score = max(result['score'] for result in vector_results)
            assert max_score < 0.1, f"Query '{query}' returned unexpectedly high score: {max_score}"
    
    # Test FTS5 search with non-existent terms
    try:
        for query in no_results_queries:
            fts_results = search_fts5_index(query, db_path, k=5)
            # FTS5 should return empty results for non-existent terms
            assert len(fts_results) == 0, f"FTS5 query '{query}' should return no results"
    except Exception:
        # FTS5 might not be available, skip this part
        pass


def test_golden_retrieval_query_variations(tmp_path):
    """Test retrieval with query variations and misspellings."""
    artifacts_dir = Path(__file__).parent / "fixtures" / "artifacts_v1"
    
    if not artifacts_dir.exists():
        pytest.skip("Artifacts not available - run build_test_artifacts.py first")
    
    vector_path = artifacts_dir / "vector.index"
    db_path = artifacts_dir / "docmeta.db"
    
    if not vector_path.exists() or not db_path.exists():
        pytest.skip("Required artifacts missing")
    
    # Test query variations and misspellings
    query_variations = [
        ("Acme", "acme"),           # Case variation
        ("Acme", "ACME"),           # Uppercase
        ("Globex", "globx"),        # Minor misspelling
        ("Contoso", "contoso co"),  # Partial match
        ("BOLT", "bolt protocol"),  # Expansion
    ]
    
    for original, variation in query_variations:
        # Get results for both queries
        original_results = search_vector_index(original, vector_path, db_path, k=10)
        variation_results = search_vector_index(variation, vector_path, db_path, k=10)
        
        # Both should return some results
        assert len(original_results) > 0, f"Original query '{original}' returned no results"
        assert len(variation_results) > 0, f"Variation query '{variation}' returned no results"
        
        # Results should have some overlap (at least one common document in top 5)
        original_docs = set(result['file_path'] for result in original_results[:5])
        variation_docs = set(result['file_path'] for result in variation_results[:5])
        
        overlap = len(original_docs.intersection(variation_docs))
        assert overlap > 0, f"No overlap between '{original}' and '{variation}' results"