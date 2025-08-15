"""Query constraint extraction utility for complex metadata queries.

This module provides utilities to extract structured constraints from natural
language queries, enabling precise metadata filtering and content searching.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from enum import Enum


class SortBy(Enum):
    """Available sort criteria."""
    MODIFIED_TIME = "modified_time"


class SortOrder(Enum):
    """Available sort orders."""
    DESC = "desc"
    ASC = "asc"


@dataclass
class QueryConstraints:
    """Structured representation of query constraints.
    
    Attributes:
        count: Requested number of items, None if not specified
        file_types: List of canonical file extensions (uppercase)
        recency: Whether query includes recency terms like "latest", "newest"
        content_terms: Meaningful content terms after removing constraints/stopwords
        sort_by: Sort criteria (default: modified_time)
        sort_order: Sort order (default: desc)
    """
    count: Optional[int] = None
    file_types: List[str] = field(default_factory=list)
    recency: bool = False
    content_terms: List[str] = field(default_factory=list)
    sort_by: SortBy = SortBy.MODIFIED_TIME
    sort_order: SortOrder = SortOrder.DESC


class ConstraintExtractor:
    """Extracts structured constraints from natural language queries."""
    
    # Number word mappings (one through twenty for safety)
    NUMBER_WORDS = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
    }
    
    # File type synonym mappings to canonical extensions
    FILE_TYPE_SYNONYMS = {
        # PDF documents
        'pdf': ['PDF'], 'pdfs': ['PDF'],
        
        # Word documents - be more specific
        'docs': ['DOCX', 'DOC'], 'docx': ['DOCX'], 'doc': ['DOC'],
        'word': ['DOCX', 'DOC'], 'word documents': ['DOCX', 'DOC'],
        'docx files': ['DOCX'], 
        
        # PowerPoint presentations
        'deck': ['PPTX', 'PPT'], 'decks': ['PPTX', 'PPT'],
        'presentation': ['PPTX', 'PPT'], 'presentations': ['PPTX', 'PPT'],
        'ppt': ['PPTX', 'PPT'], 'ppts': ['PPTX', 'PPT'],
        'pptx': ['PPTX'], 'powerpoint': ['PPTX', 'PPT'],
        'powerpoint slides': ['PPTX', 'PPT'],
        'slide': ['PPTX', 'PPT'], 'slides': ['PPTX', 'PPT'],
        
        # Excel spreadsheets
        'spreadsheet': ['XLSX', 'XLS'], 'spreadsheets': ['XLSX', 'XLS'],
        'excel': ['XLSX', 'XLS'], 'excel files': ['XLSX', 'XLS'],
        'xlsx': ['XLSX'], 'xlsx files': ['XLSX'], 'xls': ['XLS'],
        
        # Text files - be specific
        'txt': ['TXT'], 'text files': ['TXT'],
        
        # Email files
        'email': ['EMAIL'], 'emails': ['EMAIL'], 'msg': ['EMAIL'], 'eml': ['EMAIL'],
        
        # Generic fallbacks (less specific, processed last)
        'document': ['DOCX', 'DOC'], 'documents': ['DOCX', 'DOC'],
        'text': ['TXT'], 'text documents': ['TXT'],
    }
    
    # Recency indicator terms
    RECENCY_TERMS = {
        'latest', 'newest', 'recent', 'recently', 'new', 'most recent',
        'last', 'current', 'up-to-date', 'updated'
    }
    
    # Common stopwords (basic set for filtering)
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'would', 'i', 'me', 'my',
        'we', 'us', 'our', 'you', 'your', 'they', 'them', 'their',
        'this', 'these', 'those', 'what', 'which', 'who', 'when', 'where',
        'why', 'how', 'can', 'could', 'should', 'would', 'may', 'might',
        'must', 'shall', 'about', 'above', 'across', 'after', 'against',
        'along', 'among', 'around', 'before', 'behind', 'below', 'beneath',
        'beside', 'between', 'beyond', 'during', 'except', 'inside',
        'into', 'near', 'off', 'onto', 'outside', 'over', 'through',
        'throughout', 'under', 'until', 'up', 'upon', 'within', 'without'
    }
    
    # Constraint-related terms that should be removed from content
    CONSTRAINT_KEYWORDS = {
        'list', 'show', 'find', 'get', 'give', 'latest', 'newest', 'recent',
        'files', 'file', 'documents', 'document', 'items', 'results',
        'that', 'which', 'mention', 'mentions', 'about', 'containing',
        'related', 'regarding', 'concerning', 'with', 'include', 'includes'
    }
    
    @classmethod
    def extract(cls, query: str) -> QueryConstraints:
        """Extract structured constraints from a natural language query.
        
        Args:
            query: Natural language query string
            
        Returns:
            QueryConstraints object with extracted constraints
        """
        constraints = QueryConstraints()
        
        # Normalize query for processing
        normalized_query = query.lower().strip()
        tokens = re.findall(r'\b\w+\b', normalized_query)
        
        # Extract count
        constraints.count = cls._extract_count(tokens)
        
        # Extract file types
        constraints.file_types = cls._extract_file_types(tokens)
        
        # Check for recency terms
        constraints.recency = cls._has_recency_terms(tokens)
        
        # Extract content terms (after removing constraints and stopwords)
        constraints.content_terms = cls._extract_content_terms(tokens)
        
        return constraints
    
    @classmethod
    def _extract_count(cls, tokens: List[str]) -> Optional[int]:
        """Extract requested count from tokens.
        
        Args:
            tokens: List of normalized tokens
            
        Returns:
            Extracted count or None if not found, bounded to safe range (1-100)
        """
        for i, token in enumerate(tokens):
            # Check for digit numbers (but skip if it starts with -)
            if token.isdigit():
                count = int(token)
                # Ensure safe bounds
                return max(1, min(100, count))
            
            # Check for number words
            if token in cls.NUMBER_WORDS:
                return cls.NUMBER_WORDS[token]
        
        return None
    
    @classmethod 
    def _extract_file_types(cls, tokens: List[str]) -> List[str]:
        """Extract file types from tokens using synonym mapping.
        
        Args:
            tokens: List of normalized tokens
            
        Returns:
            List of canonical file extensions (uppercase)
        """
        file_types = set()
        text = ' '.join(tokens)
        
        # Process multi-word phrases first (more specific)
        phrase_matches = []
        for phrase, extensions in cls.FILE_TYPE_SYNONYMS.items():
            if ' ' in phrase and phrase in text:
                phrase_matches.append((phrase, extensions))
                file_types.update(extensions)
        
        # Remove matched phrases from consideration for individual tokens
        remaining_text = text
        for phrase, _ in phrase_matches:
            remaining_text = remaining_text.replace(phrase, '')
        
        remaining_tokens = re.findall(r'\b\w+\b', remaining_text)
        
        # Check individual tokens only if they weren't part of phrases
        for token in remaining_tokens:
            if token in cls.FILE_TYPE_SYNONYMS:
                file_types.update(cls.FILE_TYPE_SYNONYMS[token])
        
        return sorted(list(file_types))  # Sort for deterministic ordering
    
    @classmethod
    def _has_recency_terms(cls, tokens: List[str]) -> bool:
        """Check if query contains recency terms.
        
        Args:
            tokens: List of normalized tokens
            
        Returns:
            True if recency terms found
        """
        return any(token in cls.RECENCY_TERMS for token in tokens)
    
    @classmethod
    def _extract_content_terms(cls, tokens: List[str]) -> List[str]:
        """Extract meaningful content terms after removing constraints and stopwords.
        
        Args:
            tokens: List of normalized tokens
            
        Returns:
            List of content terms in stable order
        """
        content_terms = []
        
        for token in tokens:
            # Skip if it's a stopword
            if token in cls.STOPWORDS:
                continue
                
            # Skip if it's a constraint keyword
            if token in cls.CONSTRAINT_KEYWORDS:
                continue
                
            # Skip if it's a number (already processed)
            if token.isdigit() or token in cls.NUMBER_WORDS:
                continue
                
            # Skip if it's a file type synonym (already processed)
            if token in cls.FILE_TYPE_SYNONYMS:
                continue
                
            # Skip if it's a recency term (already processed)
            if token in cls.RECENCY_TERMS:
                continue
                
            # Add as content term
            content_terms.append(token)
        
        return content_terms