"""Tests for query constraint extraction."""

import pytest
from backend.src.shared.constraints import ConstraintExtractor, QueryConstraints


class TestConstraintExtractor:
    """Test cases for ConstraintExtractor."""
    
    def test_extract_count_digits(self):
        """Test extracting numeric counts from queries."""
        test_cases = [
            ("List 5 latest files", 5),
            ("Show 10 recent documents", 10),
            ("Get first 3 presentations", 3),
            ("Find top 7 spreadsheets", 7),
            ("Display 15 newest pdfs", 15),
            ("List the latest 8 documents", 8)
        ]
        
        for query, expected_count in test_cases:
            constraints = ConstraintExtractor.extract(query)
            assert constraints.count == expected_count, f"Failed for query: {query}"
    
    def test_extract_count_words(self):
        """Test extracting word-based counts from queries."""
        test_cases = [
            ("List five latest files", 5),
            ("Show ten recent documents", 10),
            ("Get first three presentations", 3),
            ("Find the top seven spreadsheets", 7),
            ("Display fifteen newest pdfs", 15),
            ("Show me twenty docs", 20)
        ]
        
        for query, expected_count in test_cases:
            constraints = ConstraintExtractor.extract(query)
            assert constraints.count == expected_count, f"Failed for query: {query}"
    
    def test_count_bounds_checking(self):
        """Test that count extraction respects bounds."""
        # Test minimum bound
        constraints = ConstraintExtractor.extract("Show 0 files")
        assert constraints.count == 1  # Should be clamped to minimum
        
        # Test maximum bound
        constraints = ConstraintExtractor.extract("List 200 documents")
        assert constraints.count == 100  # Should be clamped to maximum
        
        # Test negative numbers
        constraints = ConstraintExtractor.extract("Show -5 files")
        assert constraints.count is None  # Invalid, should not extract
    
    def test_no_count_extraction(self):
        """Test queries without explicit counts."""
        queries = [
            "List all files",
            "Show recent documents",
            "Find presentations about project",
            "Get latest spreadsheets"
        ]
        
        for query in queries:
            constraints = ConstraintExtractor.extract(query)
            assert constraints.count is None, f"Should not extract count from: {query}"
    
    def test_extract_file_types_single(self):
        """Test extracting single file type constraints."""
        test_cases = [
            ("List 5 decks", ["PPTX", "PPT"]),
            ("Show recent docs", ["DOCX", "DOC"]),
            ("Find spreadsheets", ["XLSX", "XLS"]),
            ("Get latest pdfs", ["PDF"]),
            ("Display emails", ["MSG"]),
            ("List text files", ["TXT"])
        ]
        
        for query, expected_types in test_cases:
            constraints = ConstraintExtractor.extract(query)
            assert sorted(constraints.file_types) == sorted(expected_types), f"Failed for query: {query}"
    
    def test_extract_file_types_plural(self):
        """Test extracting file types with plural forms."""
        test_cases = [
            ("List presentations", ["PPTX", "PPT"]),
            ("Show documents", ["DOCX", "DOC"]),
            ("Find workbooks", ["XLSX", "XLS"]),
            ("Get messages", ["MSG"])
        ]
        
        for query, expected_types in test_cases:
            constraints = ConstraintExtractor.extract(query)
            assert sorted(constraints.file_types) == sorted(expected_types), f"Failed for query: {query}"
    
    def test_extract_file_types_multiple(self):
        """Test extracting multiple file types from single query."""
        constraints = ConstraintExtractor.extract("Show docs and spreadsheets")
        expected = sorted(["DOCX", "DOC", "XLSX", "XLS"])
        assert sorted(constraints.file_types) == expected
        
        constraints = ConstraintExtractor.extract("List pdfs and presentations")
        expected = sorted(["PDF", "PPTX", "PPT"])
        assert sorted(constraints.file_types) == expected
    
    def test_no_file_type_extraction(self):
        """Test queries without file type constraints."""
        queries = [
            "List 5 latest files",
            "Show recent items",
            "Find something about project",
            "Get newest things"
        ]
        
        for query in queries:
            constraints = ConstraintExtractor.extract(query)
            assert constraints.file_types == [], f"Should not extract file types from: {query}"
    
    def test_recency_constraint_detection(self):
        """Test detection of time/recency constraints."""
        recency_queries = [
            "List latest files",
            "Show recent documents", 
            "Find newest presentations",
            "Get last week's files",
            "Display current spreadsheets"
        ]
        
        for query in recency_queries:
            constraints = ConstraintExtractor.extract(query)
            assert constraints.has_recency_constraint, f"Should detect recency in: {query}"
        
        non_recency_queries = [
            "List all files",
            "Show files about project",
            "Find presentations",
            "Get spreadsheets"
        ]
        
        for query in non_recency_queries:
            constraints = ConstraintExtractor.extract(query)
            assert not constraints.has_recency_constraint, f"Should not detect recency in: {query}"
    
    def test_content_terms_extraction(self):
        """Test extraction of content search terms."""
        test_cases = [
            ("List 5 decks about project roadmaps", ["project", "roadmaps"]),
            ("Show recent docs discussing budget planning", ["discussing", "budget", "planning"]),
            ("Find presentations on marketing strategy", ["marketing", "strategy"]),
            ("Get spreadsheets containing financial data", ["containing", "financial", "data"]),
            ("Latest files about quarterly review", ["quarterly", "review"])
        ]
        
        for query, expected_terms in test_cases:
            constraints = ConstraintExtractor.extract(query)
            # Check that at least the key terms are present
            found_terms = set(constraints.content_terms)
            expected_terms_set = set(expected_terms)
            assert expected_terms_set.issubset(found_terms) or len(found_terms.intersection(expected_terms_set)) > 0, \
                f"Failed for query: {query}. Expected terms {expected_terms}, got {constraints.content_terms}"
    
    def test_content_filter_detection(self):
        """Test detection of content filtering requirements."""
        content_queries = [
            "List 5 decks about project roadmaps",
            "Show docs discussing budget",
            "Find presentations on strategy", 
            "Get files containing project info"
        ]
        
        for query in content_queries:
            constraints = ConstraintExtractor.extract(query)
            assert constraints.has_content_filter, f"Should detect content filter in: {query}"
        
        metadata_only_queries = [
            "List 5 latest decks",
            "Show recent docs",
            "Find presentations",
            "Get 10 newest files"
        ]
        
        for query in metadata_only_queries:
            constraints = ConstraintExtractor.extract(query)
            assert not constraints.has_content_filter, f"Should not detect content filter in: {query}"
    
    def test_complex_multi_constraint_queries(self):
        """Test complex queries with multiple constraints."""
        # Test the example from requirements
        constraints = ConstraintExtractor.extract("List 5 latest decks that discuss project roadmaps")
        
        assert constraints.count == 5
        assert sorted(constraints.file_types) == sorted(["PPTX", "PPT"])
        assert constraints.has_recency_constraint == True
        assert constraints.has_content_filter == True
        assert len(constraints.content_terms) > 0
        assert "project" in constraints.content_terms or "roadmaps" in constraints.content_terms
        
        # Another complex example
        constraints = ConstraintExtractor.extract("Show 3 recent spreadsheets about budget")
        
        assert constraints.count == 3
        assert sorted(constraints.file_types) == sorted(["XLSX", "XLS"])
        assert constraints.has_recency_constraint == True
        assert constraints.has_content_filter == True
        assert "budget" in constraints.content_terms
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty query
        constraints = ConstraintExtractor.extract("")
        assert constraints.count is None
        assert constraints.file_types == []
        assert not constraints.has_content_filter
        
        # None query
        constraints = ConstraintExtractor.extract(None)
        assert constraints.count is None
        assert constraints.file_types == []
        assert not constraints.has_content_filter
        
        # Non-string query
        constraints = ConstraintExtractor.extract(123)
        assert constraints.count is None
        assert constraints.file_types == []
        assert not constraints.has_content_filter
    
    def test_success_criteria_examples(self):
        """Test examples from the success criteria table."""
        test_cases = [
            {
                "query": "List 5 latest decks",
                "expected_count": 5,
                "expected_file_types": ["PPTX", "PPT"],
                "expected_content_filter": False
            },
            {
                "query": "Show 3 recent spreadsheets about budget",
                "expected_count": 3,
                "expected_file_types": ["XLSX", "XLS"],
                "expected_content_filter": True
            },
            {
                "query": "Latest 10 files",
                "expected_count": 10,
                "expected_file_types": [],
                "expected_content_filter": False
            },
            {
                "query": "Decks about strategy",
                "expected_count": None,
                "expected_file_types": ["PPTX", "PPT"],
                "expected_content_filter": True
            }
        ]
        
        for test_case in test_cases:
            constraints = ConstraintExtractor.extract(test_case["query"])
            
            assert constraints.count == test_case["expected_count"], \
                f"Count mismatch for '{test_case['query']}': expected {test_case['expected_count']}, got {constraints.count}"
            
            assert sorted(constraints.file_types) == sorted(test_case["expected_file_types"]), \
                f"File types mismatch for '{test_case['query']}': expected {test_case['expected_file_types']}, got {constraints.file_types}"
            
            assert constraints.has_content_filter == test_case["expected_content_filter"], \
                f"Content filter mismatch for '{test_case['query']}': expected {test_case['expected_content_filter']}, got {constraints.has_content_filter}"


if __name__ == "__main__":
    # Run basic tests to validate functionality
    extractor = ConstraintExtractor()
    
    # Test basic extraction
    result = extractor.extract("List 5 latest decks that discuss project roadmaps")
    print(f"Test result: count={result.count}, file_types={result.file_types}, "
          f"content_terms={result.content_terms}, has_content_filter={result.has_content_filter}")
    
    # Test metadata-only query
    result = extractor.extract("Show 10 recent presentations")
    print(f"Metadata-only result: count={result.count}, file_types={result.file_types}, "
          f"has_content_filter={result.has_content_filter}")
    
    print("Basic constraint extraction tests completed")