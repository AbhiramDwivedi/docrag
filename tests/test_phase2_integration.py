"""Integration test demonstrating Phase 2 hybrid search functionality."""

import pytest
import tempfile
import sqlite3
from pathlib import Path
import sys

# Add the backend source to Python path
backend_src = Path(__file__).parent.parent / "backend" / "src"
sys.path.insert(0, str(backend_src))

# Import our implementations
from shared.hybrid_search import classify_query_intent, merge_search_results


def create_test_database_with_fts5():
    """Create a test database with FTS5 index and sample data."""
    # Check FTS5 availability first
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE VIRTUAL TABLE test_fts USING fts5(content)")
        conn.close()
    except sqlite3.OperationalError:
        pytest.skip("FTS5 not available in this SQLite build")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create main chunks table
    cursor.execute("""
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            content TEXT,
            file_path TEXT,
            chunk_index INTEGER,
            current INTEGER DEFAULT 1
        )
    """)
    
    # Create files table for metadata
    cursor.execute("""
        CREATE TABLE files (
            file_path TEXT PRIMARY KEY,
            file_name TEXT,
            document_title TEXT,
            document_id TEXT
        )
    """)
    
    # Create FTS5 table
    cursor.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            content,
            file_path UNINDEXED
        )
    """)
    
    # Sample test data representing different document types
    test_data = [
        {
            'chunk_id': 'acme_001',
            'content': 'Acme Corporation is a leading data platform company that provides advanced analytics solutions for enterprise customers. Our platform integrates with multiple data sources.',
            'file_path': 'acme_overview.txt',
            'file_name': 'acme_overview.txt',
            'document_title': 'Acme Corporation Overview',
            'document_id': 'acme_001'
        },
        {
            'chunk_id': 'bolt_001', 
            'content': 'BOLT is a high-performance protocol for real-time data streaming between distributed systems. It supports both TCP and UDP transport mechanisms.',
            'file_path': 'bolt_protocol_spec.md',
            'file_name': 'bolt_protocol_spec.md',
            'document_title': 'BOLT Protocol Specification',
            'document_id': 'bolt_001'
        },
        {
            'chunk_id': 'globex_001',
            'content': 'Globex Financial Services specializes in technology sector investments and M&A advisory services. We focus on data-driven companies and emerging technologies.',
            'file_path': 'globex_financial_report.pdf',
            'file_name': 'globex_financial_report.pdf',
            'document_title': 'Globex Financial Services Report',
            'document_id': 'globex_001'
        },
        {
            'chunk_id': 'ml_concepts_001',
            'content': 'Machine learning algorithms require large datasets to train effectively. The quality of training data significantly impacts model performance and accuracy.',
            'file_path': 'machine_learning_guide.md',
            'file_name': 'machine_learning_guide.md',
            'document_title': 'Machine Learning Concepts Guide',
            'document_id': 'ml_001'
        },
        {
            'chunk_id': 'data_quality_001',
            'content': 'Data quality is crucial for business analytics. Poor data quality leads to incorrect insights and can negatively impact business decision making processes.',
            'file_path': 'data_quality_best_practices.pdf',
            'file_name': 'data_quality_best_practices.pdf',
            'document_title': 'Data Quality Best Practices',
            'document_id': 'dq_001'
        }
    ]
    
    # Insert test data
    for doc in test_data:
        # Insert into main tables
        cursor.execute(
            "INSERT INTO chunks (chunk_id, content, file_path, chunk_index) VALUES (?, ?, ?, ?)",
            (doc['chunk_id'], doc['content'], doc['file_path'], 0)
        )
        cursor.execute(
            "INSERT OR REPLACE INTO files (file_path, file_name, document_title, document_id) VALUES (?, ?, ?, ?)",
            (doc['file_path'], doc['file_name'], doc['document_title'], doc['document_id'])
        )
        
        # Insert into FTS5 table
        cursor.execute(
            "INSERT INTO chunks_fts (chunk_id, content, file_path) VALUES (?, ?, ?)",
            (doc['chunk_id'], doc['content'], doc['file_path'])
        )
    
    conn.commit()
    conn.close()
    
    return db_path


def simulate_semantic_search(query: str, db_path: str):
    """Simulate semantic search results (normally would use embeddings)."""
    # For testing, we'll simulate semantic search with keyword relevance
    # In reality, this would use vector embeddings and cosine similarity
    
    query_lower = query.lower()
    simulated_results = []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT chunk_id, content FROM chunks WHERE current = 1")
    all_chunks = cursor.fetchall()
    
    for chunk_id, content in all_chunks:
        content_lower = content.lower()
        
        # Simple relevance scoring based on keyword overlap
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        overlap = len(query_words.intersection(content_words))
        
        if overlap > 0:
            # Simulate cosine similarity (0-1 range)
            similarity = overlap / (len(query_words) + len(content_words) - overlap)
            simulated_results.append((chunk_id, similarity))
    
    conn.close()
    
    # Sort by similarity and return top results
    simulated_results.sort(key=lambda x: x[1], reverse=True)
    return simulated_results[:10]


def perform_lexical_search(query: str, db_path: str):
    """Perform actual FTS5 lexical search."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Use FTS5 with BM25 scoring
        cursor.execute("""
            SELECT chunk_id, bm25(chunks_fts) as score
            FROM chunks_fts 
            WHERE chunks_fts MATCH ? 
            ORDER BY bm25(chunks_fts) 
            LIMIT 10
        """, (query,))
        
        results = cursor.fetchall()
        
        # Convert BM25 scores to positive values (FTS5 BM25 returns negative scores)
        # Lower (more negative) BM25 scores indicate better matches
        results = [(chunk_id, abs(score)) for chunk_id, score in results]
        
        return results
        
    except sqlite3.OperationalError as e:
        print(f"FTS5 query failed: {e}")
        return []
    finally:
        conn.close()


class TestPhase2Integration:
    """Integration tests for Phase 2 hybrid search functionality."""
    
    @pytest.fixture
    def test_db(self):
        """Create test database with sample data."""
        db_path = create_test_database_with_fts5()
        yield db_path
        Path(db_path).unlink(missing_ok=True)
    
    def test_query_routing_lexical_primary(self, test_db):
        """Test that keyword queries route to lexical search."""
        query = "find files containing ACME"
        
        # Test query classification
        intent = classify_query_intent(query)
        assert intent["strategy"] == "lexical_primary"
        assert intent["lexical_score"] > 0
        
        # Test lexical search execution
        lexical_results = perform_lexical_search("ACME", test_db)
        assert len(lexical_results) > 0
        
        # Verify it finds the ACME document
        chunk_ids = [chunk_id for chunk_id, _ in lexical_results]
        assert "acme_001" in chunk_ids
    
    def test_query_routing_semantic_primary(self, test_db):
        """Test that complex queries route to semantic search."""
        query = "What is the relationship between data quality and business outcomes?"
        
        # Test query classification
        intent = classify_query_intent(query)
        assert intent["strategy"] == "semantic_primary"
        assert intent["semantic_score"] > 0
        
        # Test semantic search execution
        semantic_results = simulate_semantic_search(query, test_db)
        assert len(semantic_results) > 0
        
        # Should find documents about data quality and business
        chunk_ids = [chunk_id for chunk_id, _ in semantic_results]
        assert any("data_quality" in chunk_id or "ml_concepts" in chunk_id for chunk_id in chunk_ids)
    
    def test_query_routing_hybrid(self, test_db):
        """Test that proper noun queries route to hybrid search."""
        query = "BOLT protocol"
        
        # Test query classification
        intent = classify_query_intent(query)
        assert intent["strategy"] == "hybrid"
        assert len(intent["proper_nouns"]) > 0
        assert "BOLT" in intent["proper_nouns"]
        
        # Test both search methods
        semantic_results = simulate_semantic_search(query, test_db)
        lexical_results = perform_lexical_search("BOLT protocol", test_db)
        
        # Both should find the BOLT document
        semantic_chunk_ids = [chunk_id for chunk_id, _ in semantic_results]
        lexical_chunk_ids = [chunk_id for chunk_id, _ in lexical_results]
        
        assert "bolt_001" in semantic_chunk_ids or "bolt_001" in lexical_chunk_ids
    
    def test_hybrid_search_execution(self, test_db):
        """Test full hybrid search execution and result merging."""
        query = "ACME data platform"
        
        # Execute both search methods
        semantic_results = simulate_semantic_search(query, test_db)
        lexical_results = perform_lexical_search("ACME data platform", test_db)
        
        # Merge results
        merged_results = merge_search_results(
            semantic_results, lexical_results,
            dense_weight=0.6, lexical_weight=0.4
        )
        
        # Verify merging
        assert len(merged_results) > 0
        
        # Should find ACME document (contains both ACME and "data platform")
        chunk_ids = [chunk_id for chunk_id, _ in merged_results]
        assert "acme_001" in chunk_ids
        
        # Verify deduplication (no duplicate chunk_ids)
        assert len(chunk_ids) == len(set(chunk_ids))
        
        # Verify sorting (scores should be in descending order)
        scores = [score for _, score in merged_results]
        assert scores == sorted(scores, reverse=True)
    
    def test_acceptance_criteria_keyword_search(self, test_db):
        """Acceptance criteria: 'find files containing [keyword]' uses lexical search."""
        queries = [
            "find files containing ACME",
            "search for documents with BOLT",
            "files containing Globex"
        ]
        
        for query in queries:
            intent = classify_query_intent(query)
            assert intent["strategy"] == "lexical_primary", f"Query '{query}' should use lexical search"
            
            # Extract the actual keyword for search
            keyword = None
            if "ACME" in query:
                keyword = "ACME"
            elif "BOLT" in query:
                keyword = "BOLT"
            elif "Globex" in query:
                keyword = "Globex"
            
            if keyword:
                lexical_results = perform_lexical_search(keyword, test_db)
                assert len(lexical_results) > 0, f"Should find documents containing '{keyword}'"
    
    def test_acceptance_criteria_proper_noun_hybrid(self, test_db):
        """Acceptance criteria: proper noun queries improve recall with hybrid."""
        proper_noun_queries = [
            "ACME",
            "BOLT", 
            "Globex"
        ]
        
        for query in proper_noun_queries:
            # Test semantic-only results
            semantic_results = simulate_semantic_search(query, test_db)
            semantic_count = len(semantic_results)
            
            # Test lexical-only results
            lexical_results = perform_lexical_search(query, test_db)
            lexical_count = len(lexical_results)
            
            # Test hybrid results
            merged_results = merge_search_results(
                semantic_results, lexical_results,
                dense_weight=0.6, lexical_weight=0.4
            )
            hybrid_count = len(merged_results)
            
            # Hybrid should provide equal or better recall
            assert hybrid_count >= max(semantic_count, lexical_count), \
                f"Hybrid search for '{query}' should improve recall: semantic={semantic_count}, lexical={lexical_count}, hybrid={hybrid_count}"
    
    def test_fts5_index_population_and_search(self, test_db):
        """Test that FTS5 index is properly populated and searchable."""
        # Verify FTS5 table exists and has data
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM chunks_fts")
        fts_count = cursor.fetchone()[0]
        assert fts_count > 0, "FTS5 table should contain indexed documents"
        
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunks_count = cursor.fetchone()[0]
        assert fts_count == chunks_count, "FTS5 table should have same number of documents as chunks table"
        
        # Test various FTS5 search features
        test_searches = [
            ("Acme", "acme_001"),
            ("BOLT", "bolt_001"),
            ("Globex", "globex_001"),
            ('"data platform"', "acme_001"),  # Phrase search
            ("protocol OR financial", ["bolt_001", "globex_001"]),  # Boolean OR
        ]
        
        for search_query, expected in test_searches:
            cursor.execute(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts)",
                (search_query,)
            )
            results = [row[0] for row in cursor.fetchall()]
            
            if isinstance(expected, str):
                assert expected in results, f"Search '{search_query}' should find {expected}"
            else:
                assert any(exp in results for exp in expected), f"Search '{search_query}' should find one of {expected}"
        
        conn.close()
    
    def test_no_content_search_in_metadata(self):
        """Acceptance criteria: No content search remains in metadata plugin."""
        # Read the metadata plugin source to verify removal
        metadata_plugin_path = Path(__file__).parent.parent / "backend" / "src" / "querying" / "agents" / "plugins" / "metadata_commands.py"
        
        with open(metadata_plugin_path, 'r') as f:
            content = f.read()
        
        # Verify the method is removed
        assert "_find_files_by_content" not in content, "find_files_by_content method should be removed"
        assert "find_files_by_content" not in content or content.count("find_files_by_content") <= 1, \
            "References to find_files_by_content should be minimal or removed"