"""Enhanced query parser for converting natural language to structured metadata commands."""

import re
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class QueryParser:
    """Parses natural language queries and converts them to structured metadata commands."""
    
    def __init__(self):
        """Initialize the query parser."""
        # File type mappings
        self.file_type_map = {
            # Presentations
            'presentation': 'PPTX', 'presentations': 'PPTX', 'powerpoint': 'PPTX',
            'ppt': 'PPTX', 'pptx': 'PPTX', 'slides': 'PPTX',
            
            # Documents  
            'document': 'DOCX', 'documents': 'DOCX', 'doc': 'DOC', 'docx': 'DOCX',
            'word': 'DOCX', 'text': 'TXT', 'txt': 'TXT',
            
            # Spreadsheets
            'spreadsheet': 'XLSX', 'spreadsheets': 'XLSX', 'excel': 'XLSX',
            'xls': 'XLS', 'xlsx': 'XLSX',
            
            # PDFs
            'pdf': 'PDF', 'pdfs': 'PDF',
            
            # Emails
            'email': 'MSG', 'emails': 'MSG', 'mail': 'MSG', 'message': 'MSG',
            'msg': 'MSG'
        }
        
        # Number mappings
        self.number_map = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5,
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5
        }
        
        # Time filter mappings
        self.time_map = {
            'recent': 'recent', 'recently': 'recent', 'latest': 'recent', 'newest': 'recent',
            'last week': 'last_week', 'this week': 'this_week', 'past week': 'last_week',
            'last month': 'last_month', 'this month': 'this_month', 'past month': 'last_month',
            'yesterday': 'yesterday', 'today': 'today'
        }
    
    def parse_query(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """Parse natural language query into operation and parameters.
        
        Args:
            question: Natural language question
            
        Returns:
            Tuple of (operation, parameters_dict)
        """
        question_lower = question.lower().strip()
        
        # Determine the operation type
        operation = self._determine_operation(question_lower)
        
        # Extract parameters based on operation
        params = self._extract_parameters(question_lower, operation)
        params["operation"] = operation
        
        logger.debug(f"Parsed query '{question}' -> operation='{operation}', params={params}")
        
        return operation, params
    
    def _determine_operation(self, question: str) -> str:
        """Determine the primary operation from the question."""
        
        # Latest/recent files operations
        if any(keyword in question for keyword in ['latest', 'recent', 'newest', 'last', 'modified']):
            if any(keyword in question for keyword in ['show', 'list', 'find', 'get']):
                return 'get_latest_files'
        
        # Content search operations  
        if any(keyword in question for keyword in ['about', 'related to', 'containing', 'mentions']):
            return 'find_files_by_content'
        
        # Count operations
        if any(keyword in question for keyword in ['how many', 'count', 'number of']):
            return 'get_file_count'
        
        # File types operations
        if any(keyword in question for keyword in ['file types', 'what types', 'types of files']):
            return 'get_file_types'
        
        # Statistics operations
        if any(keyword in question for keyword in ['statistics', 'stats', 'summary', 'overview']):
            return 'get_file_stats'
        
        # List operations (default for "show me", "list", etc.)
        if any(keyword in question for keyword in ['show me', 'list', 'find', 'get']):
            return 'get_latest_files'
        
        # Default to stats for general questions
        return 'get_file_stats'
    
    def _extract_parameters(self, question: str, operation: str) -> Dict[str, Any]:
        """Extract parameters from the question based on operation type."""
        params = {}
        
        # Extract file type
        file_type = self._extract_file_type(question)
        if file_type:
            params["file_type"] = file_type
        
        # Extract count
        count = self._extract_count(question)
        if count:
            params["count"] = count
        
        # Extract time filter
        time_filter = self._extract_time_filter(question)
        if time_filter:
            params["time_filter"] = time_filter
        
        # Extract keywords for content search
        if operation == 'find_files_by_content':
            keywords = self._extract_keywords(question)
            if keywords:
                params["keywords"] = keywords
        
        return params
    
    def _extract_file_type(self, question: str) -> Optional[str]:
        """Extract file type from the question."""
        question_lower = question.lower()
        
        # Look for file type keywords
        for keyword, file_type in self.file_type_map.items():
            if keyword in question_lower:
                return file_type
        
        # Look for file extensions
        ext_pattern = r'\b\w+\.(pdf|docx?|xlsx?|pptx?|txt|msg)\b'
        match = re.search(ext_pattern, question_lower)
        if match:
            ext = match.group(1).upper()
            return ext
        
        return None
    
    def _extract_count(self, question: str) -> Optional[int]:
        """Extract count/number from the question."""
        # Look for explicit numbers first
        number_pattern = r'\b(\d+)\b'
        matches = re.findall(number_pattern, question)
        if matches:
            return int(matches[0])  # Take the first number found
        
        # Look for word numbers with word boundaries - sort by length desc to match longer words first
        question_lower = question.lower()
        sorted_words = sorted(self.number_map.keys(), key=len, reverse=True)
        
        for word in sorted_words:
            # Use word boundaries to avoid partial matches like 'i' in 'list'
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, question_lower):
                return self.number_map[word]
        
        return None
    
    def _extract_time_filter(self, question: str) -> Optional[str]:
        """Extract time filter from the question."""
        # Look for time phrases (longer phrases first to avoid conflicts)
        time_phrases = sorted(self.time_map.keys(), key=len, reverse=True)
        
        for phrase in time_phrases:
            if phrase in question:
                return self.time_map[phrase]
        
        return None
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords for content search."""
        keywords = []
        
        # Look for content after "about", "related to", etc.
        patterns = [
            r'about\s+([^,\.!?]+)',
            r'related to\s+([^,\.!?]+)', 
            r'containing\s+([^,\.!?]+)',
            r'mentions\s+([^,\.!?]+)',
            r'regarding\s+([^,\.!?]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                # Clean up the extracted text
                keyword = match.strip().lower()
                # Remove common stop words and clean up
                keyword = re.sub(r'\b(the|a|an|and|or|of|in|on|at|to|for|with|by)\b', '', keyword)
                keyword = keyword.strip()
                if keyword and len(keyword) > 1:
                    keywords.append(keyword)
        
        return keywords


def create_enhanced_metadata_params(question: str) -> Dict[str, Any]:
    """Create structured metadata parameters from natural language question.
    
    This function acts as the LLM intelligence layer that converts natural language
    to structured metadata commands.
    
    Args:
        question: Natural language question
        
    Returns:
        Parameters dictionary for metadata plugin
    """
    parser = QueryParser()
    operation, params = parser.parse_query(question)
    
    # Add the operation to the parameters dictionary
    params["operation"] = operation
    
    return params