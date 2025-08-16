"""Query constraint extraction for multi-constraint query processing.

This module provides utilities to parse natural language constraints from queries
and extract structured parameters for metadata and content filtering.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Dict, Any, Union

logger = logging.getLogger(__name__)


@dataclass
class QueryConstraints:
    """Structured representation of query constraints."""
    
    # Numeric constraints
    count: Optional[int] = None
    
    # File type constraints
    file_types: List[str] = None
    
    # Content constraints
    content_terms: List[str] = None
    has_content_filter: bool = False
    
    # Time constraints (detected but used for validation)
    has_recency_constraint: bool = False
    
    def __post_init__(self):
        """Initialize default values."""
        if self.file_types is None:
            self.file_types = []
        if self.content_terms is None:
            self.content_terms = []


class ConstraintExtractor:
    """Extract structured constraints from natural language queries."""
    
    # Number word to digit mapping
    NUMBER_WORDS = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
        'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10
    }
    
    # File type synonym mappings
    FILE_TYPE_SYNONYMS = {
        'deck': ['PPTX', 'PPT'],
        'decks': ['PPTX', 'PPT'],
        'presentation': ['PPTX', 'PPT'],
        'presentations': ['PPTX', 'PPT'],
        'slides': ['PPTX', 'PPT'],
        'powerpoint': ['PPTX', 'PPT'],
        
        'doc': ['DOCX', 'DOC'],
        'docs': ['DOCX', 'DOC'],
        'document': ['DOCX', 'DOC'],
        'documents': ['DOCX', 'DOC'],
        'word': ['DOCX', 'DOC'],
        
        'spreadsheet': ['XLSX', 'XLS'],
        'spreadsheets': ['XLSX', 'XLS'],
        'excel': ['XLSX', 'XLS'],
        'workbook': ['XLSX', 'XLS'],
        'workbooks': ['XLSX', 'XLS'],
        
        'pdf': ['PDF'],
        'pdfs': ['PDF'],
        
        'email': ['MSG'],
        'emails': ['MSG'],
        'message': ['MSG'],
        'messages': ['MSG'],
        
        'text': ['TXT'],
        'txt': ['TXT']
    }
    
    # Time/recency keywords
    RECENCY_KEYWORDS = {
        'latest', 'recent', 'newest', 'new', 'last', 'current',
        'today', 'yesterday', 'this', 'past'
    }
    
    # Constraint keywords to remove when extracting content terms
    CONSTRAINT_KEYWORDS = {
        # Count-related
        'list', 'show', 'get', 'find', 'give', 'display',
        'first', 'top', 'initial', 'latest', 'recent', 'newest', 'last',
        
        # File type related (values from FILE_TYPE_SYNONYMS)
        'deck', 'decks', 'presentation', 'presentations', 'slides', 'powerpoint',
        'doc', 'docs', 'document', 'documents', 'word',
        'spreadsheet', 'spreadsheets', 'excel', 'workbook', 'workbooks',
        'pdf', 'pdfs', 'email', 'emails', 'message', 'messages',
        'text', 'txt', 'file', 'files',
        
        # Connector words
        'that', 'which', 'with', 'containing', 'about', 'on', 'regarding',
        'discussing', 'related', 'to', 'for', 'in', 'of', 'and', 'or',
        'the', 'a', 'an', 'is', 'are', 'has', 'have', 'had'
    }
    
    # Safe bounds for extracted numbers
    MIN_COUNT = 1
    MAX_COUNT = 100
    DEFAULT_COUNT = 10
    
    @classmethod
    def extract(cls, query: str) -> QueryConstraints:
        """Extract structured constraints from a natural language query.
        
        Args:
            query: Natural language query string
            
        Returns:
            QueryConstraints with extracted parameters
        """
        # Handle various invalid input types gracefully
        if query is None:
            return QueryConstraints()
        
        if not isinstance(query, str):
            logger.debug(f"Non-string query input: {type(query)}, returning empty constraints")
            return QueryConstraints()
        
        # Handle empty or whitespace-only strings
        if not query.strip():
            return QueryConstraints()
        
        query_lower = query.lower().strip()
        constraints = QueryConstraints()
        
        # Extract count constraint
        constraints.count = cls._extract_count(query_lower)
        
        # Extract file type constraints
        constraints.file_types = cls._extract_file_types(query_lower)
        
        # Check for recency constraints
        constraints.has_recency_constraint = cls._has_recency_constraint(query_lower)
        
        # Extract content terms (after removing constraint keywords)
        content_terms = cls._extract_content_terms(query_lower)
        constraints.content_terms = content_terms
        constraints.has_content_filter = len(content_terms) > 0
        
        logger.debug(f"Extracted constraints from '{query}': "
                    f"count={constraints.count}, file_types={constraints.file_types}, "
                    f"content_terms={constraints.content_terms}, "
                    f"has_content_filter={constraints.has_content_filter}, "
                    f"has_recency_constraint={constraints.has_recency_constraint}")
        
        return constraints
    
    @classmethod
    def explain_constraints(cls, query: str) -> str:
        """Provide human-readable explanation of extracted constraints.
        
        Args:
            query: Natural language query string
            
        Returns:
            Human-readable explanation of what constraints were detected
        """
        constraints = cls.extract(query)
        
        parts = []
        
        if constraints.count:
            parts.append(f"Count: {constraints.count}")
            
        if constraints.file_types:
            parts.append(f"File types: {', '.join(constraints.file_types)}")
            
        if constraints.content_terms:
            parts.append(f"Content search terms: {', '.join(constraints.content_terms)}")
            
        if constraints.has_recency_constraint:
            parts.append("Time constraint: Recent/latest files requested")
            
        if not parts:
            return "No specific constraints detected - will use default search parameters"
            
        return "Detected constraints: " + "; ".join(parts)
    
    @classmethod
    def _extract_count(cls, query: str) -> Optional[int]:
        """Extract numeric count from query.
        
        Args:
            query: Lowercase query string
            
        Returns:
            Extracted count within safe bounds, or None if not found
        """
        # Pattern for positive digits only (reject negative numbers)
        digit_patterns = [
            r'\b(\d+)\s*(?:latest|recent|newest|new|first|top|files?|documents?|items?|results?|presentations?|spreadsheets?|docs?|pdfs?)\b',
            r'\b(?:(?:first|top|latest|recent|newest|show|list|get|find)\s+)?(\d+)(?!\s*[\w-])\b',
            r'\b(\d+)\s+(?:of\s+)?(?:the\s+)?(?:latest|recent|newest|new|first|top)\b'
        ]
        
        for pattern in digit_patterns:
            match = re.search(pattern, query)
            if match:
                try:
                    count_str = match.group(1)
                    # Ensure we didn't match a negative number
                    full_match = match.group(0)
                    if '-' in query[max(0, match.start()-2):match.start()]:
                        continue  # Skip negative numbers
                    
                    count = int(count_str)
                    # Accept any non-negative number, clamp will handle bounds
                    if count >= 0:
                        return cls._clamp_count(count)
                except (ValueError, IndexError):
                    continue
        
        # Pattern for number words - handle different contexts
        number_word_patterns = [
            # Handle "get/find first X" specifically (where X is the actual number we want)
            r'\b(?:get|find)\s+first\s+(' + '|'.join([k for k in cls.NUMBER_WORDS.keys() if k not in ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']]) + r')\s',
            # Handle various contexts with optional words in between
            r'\b(?:show|list|get|find|display)(?:\s+me)?(?:\s+the)?(?:\s+top)?\s+(' + '|'.join(cls.NUMBER_WORDS.keys()) + r')(?:\s+(?:latest|recent|newest|new|files?|documents?|items?|results?|spreadsheets?|presentations?|docs?|pdfs?))',
            # Standard pattern: number followed by type/qualifier
            r'\b(' + '|'.join(cls.NUMBER_WORDS.keys()) + r')\s+(?:latest|recent|newest|new|files?|documents?|items?|results?)\b'
        ]
        
        for pattern in number_word_patterns:
            match = re.search(pattern, query)
            if match:
                word = match.group(1)
                if word in cls.NUMBER_WORDS:
                    count = cls.NUMBER_WORDS[word]
                    return cls._clamp_count(count)
        
        # No count found
        return None
    
    @classmethod
    def _extract_file_types(cls, query: str) -> List[str]:
        """Extract file type constraints from query.
        
        Args:
            query: Lowercase query string
            
        Returns:
            List of uppercase file extensions with modern formats prioritized
        """
        file_types = set()
        
        for synonym, extensions in cls.FILE_TYPE_SYNONYMS.items():
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + re.escape(synonym) + r'\b', query):
                file_types.update(extensions)
        
        # Convert to sorted list with modern format prioritization
        result = list(file_types)
        result.sort(key=cls._get_format_priority)
        return result
    
    @classmethod
    def _get_format_priority(cls, file_type: str) -> tuple:
        """Get priority for file type sorting (modern formats first).
        
        Args:
            file_type: File extension like 'XLSX', 'XLS'
            
        Returns:
            Tuple for sorting (lower values = higher priority)
        """
        # Priority mapping: modern formats get lower numbers (higher priority)
        priority_map = {
            # Office formats - modern first
            'DOCX': (1, 1), 'DOC': (1, 2),
            'XLSX': (2, 1), 'XLS': (2, 2), 
            'PPTX': (3, 1), 'PPT': (3, 2),
            
            # Other formats
            'PDF': (4, 1),
            'MSG': (5, 1),
            'TXT': (6, 1)
        }
        
        return priority_map.get(file_type, (99, 1))  # Unknown formats last
    
    @classmethod
    def _has_recency_constraint(cls, query: str) -> bool:
        """Check if query has time/recency constraints.
        
        Args:
            query: Lowercase query string
            
        Returns:
            True if recency keywords are found
        """
        for keyword in cls.RECENCY_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', query):
                return True
        return False
    
    @classmethod
    def _extract_content_terms(cls, query: str) -> List[str]:
        """Extract content search terms by removing constraint keywords.
        
        Args:
            query: Lowercase query string
            
        Returns:
            List of meaningful content terms
        """
        # Start with the original query
        words = query.split()
        
        # Remove constraint keywords
        content_words = []
        for word in words:
            # Remove punctuation and normalize
            clean_word = re.sub(r'[^\w\s]', '', word).strip()
            
            # Skip empty words, numbers, and constraint keywords
            if (clean_word and 
                not clean_word.isdigit() and 
                clean_word not in cls.CONSTRAINT_KEYWORDS and
                clean_word not in cls.NUMBER_WORDS and
                len(clean_word) > 2):  # Skip very short words
                content_words.append(clean_word)
        
        # Remove duplicate words while preserving order
        seen = set()
        unique_words = []
        for word in content_words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        
        return unique_words
    
    @classmethod
    def _clamp_count(cls, count: int) -> int:
        """Clamp count to safe bounds.
        
        Args:
            count: Raw extracted count
            
        Returns:
            Count clamped to [MIN_COUNT, MAX_COUNT]
        """
        if count < cls.MIN_COUNT:
            logger.debug(f"Count {count} below minimum, using {cls.MIN_COUNT}")
            return cls.MIN_COUNT
        elif count > cls.MAX_COUNT:
            logger.debug(f"Count {count} above maximum, using {cls.MAX_COUNT}")
            return cls.MAX_COUNT
        else:
            return count


# Configuration helper functions
def get_default_count() -> int:
    """Get default count for queries without explicit count."""
    try:
        from shared.config import get_settings
        settings = get_settings()
        return settings.default_query_count
    except:
        return ConstraintExtractor.DEFAULT_COUNT


def get_max_count() -> int:
    """Get maximum allowed count for safety."""
    try:
        from shared.config import get_settings
        settings = get_settings()
        return settings.max_query_count
    except:
        return ConstraintExtractor.MAX_COUNT


def get_content_filtering_multiplier() -> int:
    """Get multiplier for widening metadata queries when content filtering."""
    try:
        from shared.config import get_settings
        settings = get_settings()
        return settings.content_filtering_multiplier
    except:
        return 3  # Default fallback


def validate_constraint_results(constraints: QueryConstraints, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate if search results meet the extracted constraints.
    
    Args:
        constraints: Original query constraints
        results: Search results to validate
        
    Returns:
        Dictionary with validation results and suggestions
    """
    validation = {
        "meets_constraints": True,
        "issues": [],
        "suggestions": [],
        "result_count": len(results)
    }
    
    # Check count constraint
    if constraints.count and len(results) < constraints.count:
        validation["meets_constraints"] = False
        validation["issues"].append(f"Requested {constraints.count} items but only found {len(results)}")
        validation["suggestions"].append("Try broadening your search terms or removing some constraints")
    
    # Check file type constraints
    if constraints.file_types and results:
        result_file_types = set()
        for result in results:
            file_path = result.get("file", "") or result.get("document_path", "")
            if file_path:
                # Extract extension
                ext = file_path.split('.')[-1].upper() if '.' in file_path else ""
                if ext:
                    result_file_types.add(ext)
        
        requested_types = set(constraints.file_types)
        if not requested_types.intersection(result_file_types):
            validation["meets_constraints"] = False
            validation["issues"].append(f"Requested file types {constraints.file_types} but found {list(result_file_types)}")
            validation["suggestions"].append("Try using different file type terms or check if the requested file types exist in your document collection")
    
    # Check content filtering effectiveness
    if constraints.has_content_filter and constraints.content_terms:
        if not results:
            validation["meets_constraints"] = False
            validation["issues"].append(f"No results found matching content terms: {constraints.content_terms}")
            validation["suggestions"].append("Try using different keywords or synonyms")
    
    return validation