"""Tests for query constraint extraction functionality."""

import pytest
from backend.src.shared.constraints import (
    ConstraintExtractor, 
    QueryConstraints,
    SortBy,
    SortOrder
)


class TestConstraintExtractor:
    """Test cases for ConstraintExtractor functionality."""
    
    def test_extract_count_from_digits(self):
        """Test extracting count from digit numbers."""
        # Single digit
        result = ConstraintExtractor.extract("list 5 files")
        assert result.count == 5
        
        # Double digit
        result = ConstraintExtractor.extract("show me 15 documents")
        assert result.count == 15
        
        # At safe upper bound
        result = ConstraintExtractor.extract("get 100 items")
        assert result.count == 100
        
        # Beyond safe bound (should be clipped)
        result = ConstraintExtractor.extract("list 500 files")
        assert result.count == 100
        
        # Zero (should be set to 1)
        result = ConstraintExtractor.extract("list 0 files") 
        assert result.count == 1
    
    def test_extract_count_from_words(self):
        """Test extracting count from number words."""
        test_cases = [
            ("one latest file", 1),
            ("three recent documents", 3), 
            ("ten newest pdfs", 10),
            ("twenty spreadsheets", 20)
        ]
        
        for query, expected in test_cases:
            result = ConstraintExtractor.extract(query)
            assert result.count == expected, f"Failed for query: {query}"
    
    def test_extract_count_none_when_missing(self):
        """Test that count is None when not specified."""
        result = ConstraintExtractor.extract("latest files")
        assert result.count is None
        
        result = ConstraintExtractor.extract("show me documents")
        assert result.count is None
    
    def test_extract_file_types_single(self):
        """Test extracting single file types."""
        test_cases = [
            ("latest pdf", ["PDF"]),
            ("recent docs", ["DOC", "DOCX"]),
            ("newest presentations", ["PPT", "PPTX"]),
            ("excel files", ["XLS", "XLSX"]),
            ("text documents", ["TXT"])
        ]
        
        for query, expected in test_cases:
            result = ConstraintExtractor.extract(query)
            assert result.file_types == expected, f"Failed for query: {query}"
    
    def test_extract_file_types_multiple(self):
        """Test extracting multiple file types from query."""
        result = ConstraintExtractor.extract("list pdfs and docs")
        # Should include both PDF and Word document types
        expected_types = set(result.file_types)
        assert "PDF" in expected_types
        assert "DOC" in expected_types or "DOCX" in expected_types
    
    def test_extract_file_types_synonyms(self):
        """Test file type synonym mapping."""
        synonym_tests = [
            # PDF
            ("pdf files", ["PDF"]),
            ("pdfs", ["PDF"]),
            
            # Word docs
            ("word documents", ["DOC", "DOCX"]),
            ("docx files", ["DOCX"]),
            
            # Presentations
            ("decks", ["PPT", "PPTX"]),
            ("powerpoint slides", ["PPT", "PPTX"]),
            
            # Spreadsheets  
            ("spreadsheets", ["XLS", "XLSX"]),
            ("xlsx files", ["XLSX"])
        ]
        
        for query, expected in synonym_tests:
            result = ConstraintExtractor.extract(query)
            assert result.file_types == expected, f"Failed for query: {query}"
    
    def test_recency_detection(self):
        """Test detection of recency terms."""
        recency_queries = [
            "latest files",
            "newest documents", 
            "recent presentations",
            "most recent pdfs",
            "show me updated spreadsheets"
        ]
        
        for query in recency_queries:
            result = ConstraintExtractor.extract(query)
            assert result.recency is True, f"Failed to detect recency in: {query}"
            
        # Non-recency queries
        non_recency_queries = [
            "all files",
            "documents about budget",
            "find spreadsheets"
        ]
        
        for query in non_recency_queries:
            result = ConstraintExtractor.extract(query)
            assert result.recency is False, f"False positive recency in: {query}"
    
    def test_content_terms_extraction(self):
        """Test extraction of meaningful content terms."""
        # Simple content terms
        result = ConstraintExtractor.extract("latest files about budget")
        assert "budget" in result.content_terms
        
        # Multiple content terms
        result = ConstraintExtractor.extract("recent docs about project planning")
        expected_terms = {"project", "planning"}
        actual_terms = set(result.content_terms)
        assert expected_terms.issubset(actual_terms)
        
        # Content terms with stopwords removed
        result = ConstraintExtractor.extract("find documents that mention the quarterly report")
        content_terms = set(result.content_terms)
        # Should include meaningful terms
        assert "quarterly" in content_terms
        assert "report" in content_terms
        # Should not include stopwords
        assert "the" not in content_terms
        assert "that" not in content_terms
    
    def test_content_terms_empty_for_metadata_only(self):
        """Test that content terms are empty for metadata-only queries."""
        metadata_only_queries = [
            "list 10 latest files",
            "show newest pdfs", 
            "get recent documents"
        ]
        
        for query in metadata_only_queries:
            result = ConstraintExtractor.extract(query)
            assert len(result.content_terms) == 0, f"Unexpected content terms in: {query}"
    
    def test_complex_query_extraction(self):
        """Test extraction from complex multi-constraint queries."""
        query = "list 5 latest pdfs that mention budget planning"
        result = ConstraintExtractor.extract(query)
        
        assert result.count == 5
        assert result.file_types == ["PDF"]
        assert result.recency is True
        assert "budget" in result.content_terms
        assert "planning" in result.content_terms
    
    def test_defaults_are_set(self):
        """Test that default values are properly set."""
        result = ConstraintExtractor.extract("simple query")
        
        assert result.sort_by == SortBy.MODIFIED_TIME
        assert result.sort_order == SortOrder.DESC
        assert isinstance(result.file_types, list)
        assert isinstance(result.content_terms, list)
    
    def test_deterministic_ordering(self):
        """Test that results are deterministically ordered."""
        query = "docs and spreadsheets about budget"
        
        # Extract multiple times and verify consistent ordering
        results = [ConstraintExtractor.extract(query) for _ in range(5)]
        
        first_result = results[0]
        for result in results[1:]:
            assert result.file_types == first_result.file_types
            assert result.content_terms == first_result.content_terms
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty query
        result = ConstraintExtractor.extract("")
        assert result.count is None
        assert len(result.file_types) == 0
        assert len(result.content_terms) == 0
        
        # Query with only stopwords
        result = ConstraintExtractor.extract("the and of")
        assert len(result.content_terms) == 0
        
        # Query with special characters
        result = ConstraintExtractor.extract("find docs about AI/ML research!")
        content_terms = set(result.content_terms)
        assert "ai" in content_terms or "ml" in content_terms or "research" in content_terms