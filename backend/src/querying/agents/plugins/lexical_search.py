"""Lexical search plugin using SQLite FTS5 for keyword and exact text matching.

This plugin provides lexical search capabilities using SQLite's FTS5 full-text
search extension, offering BM25 scoring and exact/prefix matching for keyword
queries that complement semantic search.
"""

import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from ..plugin import Plugin, PluginInfo

# Try different import approaches for the dependencies
try:
    from shared.config import settings
except ImportError:
    try:
        from src.shared.config import settings
    except ImportError:
        import os
        backend_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(backend_root))
        from src.shared.config import settings

logger = logging.getLogger(__name__)


class LexicalSearchPlugin(Plugin):
    """Plugin for lexical search using SQLite FTS5.
    
    This plugin provides keyword-based search functionality with:
    - BM25 scoring via FTS5
    - Exact phrase matching
    - Prefix matching
    - Boolean query operators (AND, OR, NOT)
    - Integration with hybrid search workflows
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the lexical search plugin.
        
        Args:
            db_path: Path to SQLite database, defaults to config setting
        """
        self.db_path = db_path or getattr(settings, 'db_path', 'data/docmeta.db')
        self._fts5_available = None
        self._index_exists = None
        
    def get_info(self) -> PluginInfo:
        """Get plugin information."""
        return PluginInfo(
            name="lexical_search",
            description="Lexical search using SQLite FTS5 for keyword and exact text matching",
            version="1.0.0",
            capabilities=["search", "lexical", "fts5", "bm25"],
            parameters={
                "query": {"type": "string", "required": True, "description": "Search query"},
                "limit": {"type": "integer", "default": 20, "description": "Maximum results"},
                "exact_phrase": {"type": "boolean", "default": False, "description": "Use exact phrase matching"},
                "prefix_match": {"type": "boolean", "default": False, "description": "Enable prefix matching"}
            }
        )
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute lexical search with comprehensive input validation.
        
        Args:
            params: Search parameters including query, limit, etc.
            
        Returns:
            Search results with lexical scores and metadata
        """
        # Input validation and sanitization
        query = params.get("query", "")
        if not isinstance(query, str):
            return {
                "response": "Invalid query parameter: must be a string.",
                "sources": [],
                "metadata": {"error": "invalid_query_type", "results_count": 0}
            }
        
        query = query.strip()
        if not query:
            return {
                "response": "No search query provided.",
                "sources": [],
                "metadata": {"error": "empty_query", "results_count": 0}
            }
        
        # Validate and sanitize limit parameter
        limit = params.get("limit", 20)
        try:
            limit = int(limit)
            if limit <= 0 or limit > 100:  # Reasonable limits
                limit = 20
        except (ValueError, TypeError):
            limit = 20
        
        # Validate boolean parameters
        exact_phrase = bool(params.get("exact_phrase", False))
        prefix_match = bool(params.get("prefix_match", False))
        
        try:
            # Check FTS5 availability and index existence
            if not self._check_fts5_availability():
                logger.warning("FTS5 not available, lexical search disabled")
                return {
                    "response": "Lexical search not available (FTS5 not supported).",
                    "sources": [],
                    "metadata": {"error": "fts5_unavailable", "results_count": 0}
                }
            
            if not self._check_index_exists():
                logger.warning("FTS5 index not found, lexical search disabled")
                return {
                    "response": "Lexical search index not available.",
                    "sources": [],
                    "metadata": {"error": "index_missing", "results_count": 0}
                }
            
            # Perform lexical search
            try:
                results = self._search_fts5(query, limit, exact_phrase, prefix_match)
            except Exception as e:
                logger.error(f"Database connection error during search: {e}")
                return {
                    "response": "Database connection error occurred during search.",
                    "sources": [],
                    "metadata": {"error": "database_connection_error", "results_count": 0, "query": query}
                }
            
            if not results:
                return {
                    "response": f"No documents found containing '{query}'.",
                    "sources": [],
                    "metadata": {"results_count": 0, "query": query}
                }
            
            # Format results
            sources = []
            for chunk_id, score, content, file_path in results:
                sources.append({
                    "chunk_id": chunk_id,
                    "score": score,
                    "content": content[:500] + "..." if len(content) > 500 else content,
                    "file_path": file_path,
                    "search_type": "lexical"
                })
            
            response = f"Found {len(results)} documents containing '{query}' using lexical search."
            
            return {
                "response": response,
                "sources": sources,
                "metadata": {
                    "results_count": len(results),
                    "query": query,
                    "search_type": "lexical",
                    "exact_phrase": exact_phrase,
                    "prefix_match": prefix_match
                }
            }
            
        except Exception as e:
            logger.error(f"Lexical search failed: {e}", exc_info=True)
            return {
                "response": f"Lexical search failed: {str(e)}",
                "sources": [],
                "metadata": {"error": str(e), "results_count": 0}
            }
    
    def _check_fts5_availability(self) -> bool:
        """Check if SQLite was compiled with FTS5 support."""
        if self._fts5_available is not None:
            return self._fts5_available
            
        try:
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            cursor.execute("CREATE VIRTUAL TABLE test_fts USING fts5(content)")
            conn.close()
            self._fts5_available = True
            return True
        except sqlite3.OperationalError:
            self._fts5_available = False
            return False
    
    def _check_index_exists(self) -> bool:
        """Check if the FTS5 index table exists."""
        if self._index_exists is not None:
            return self._index_exists
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
            )
            result = cursor.fetchone()
            conn.close()
            self._index_exists = result is not None
            return self._index_exists
        except Exception as e:
            logger.warning(f"Could not check FTS5 index existence: {e}")
            self._index_exists = False
            return False
    
    def _search_fts5(self, query: str, limit: int, exact_phrase: bool = False, 
                     prefix_match: bool = False) -> List[Tuple[str, float, str, str]]:
        """Perform FTS5 search and return results with BM25 scores.
        
        Args:
            query: Search query
            limit: Maximum number of results
            exact_phrase: Whether to use exact phrase matching
            prefix_match: Whether to enable prefix matching
            
        Returns:
            List of (chunk_id, score, content, file_path) tuples
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare FTS5 query with sanitization
            fts_query = self._prepare_fts5_query(query, exact_phrase, prefix_match)
            
            if not fts_query:  # Empty after sanitization
                logger.warning(f"Query became empty after sanitization: '{query}'")
                return []
            
            # Execute FTS5 search with BM25 ranking
            cursor.execute("""
                SELECT chunk_id, bm25(chunks_fts) as score, content, file_path
                FROM chunks_fts 
                WHERE chunks_fts MATCH ? 
                ORDER BY bm25(chunks_fts) 
                LIMIT ?
            """, (fts_query, limit))
            
            results = cursor.fetchall()
            
            # Convert BM25 scores to positive values (FTS5 BM25 returns negative scores)
            # Lower (more negative) BM25 scores indicate better matches
            results = [(chunk_id, abs(score), content, file_path) 
                      for chunk_id, score, content, file_path in results]
            
            return results
            
        except sqlite3.OperationalError as e:
            # Database connection errors should be re-raised to be handled at execute level
            if "unable to open database" in str(e) or "no such file or directory" in str(e):
                logger.error(f"Database connection error: {e}")
                raise e
            logger.error(f"FTS5 query failed: {e}")
            logger.error(f"Original query: '{query}', Prepared query: '{fts_query if 'fts_query' in locals() else 'N/A'}'")
            return []
        except sqlite3.Error as e:
            # Database connection errors should be re-raised to be handled at execute level
            if "no such file or directory" in str(e) or "unable to open database" in str(e):
                logger.error(f"Database connection error: {e}")
                raise e
            logger.error(f"Database error during FTS5 search: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during FTS5 search: {e}", exc_info=True)
            return []
        finally:
            if conn:
                conn.close()
    
    def _prepare_fts5_query(self, query: str, exact_phrase: bool = False, 
                           prefix_match: bool = False) -> str:
        """Prepare query string for FTS5 MATCH syntax with input sanitization.
        
        Args:
            query: Original search query
            exact_phrase: Whether to use exact phrase matching
            prefix_match: Whether to enable prefix matching
            
        Returns:
            FTS5-formatted query string, properly sanitized
        """
        # Input validation
        if not query or not isinstance(query, str):
            return ""
        
        # Remove potentially dangerous characters and limit length
        query = query.strip()[:1000]  # Limit query length to prevent DoS
        
        # Comprehensive sanitization for FTS5 special characters
        query = self._sanitize_fts5_input(query)
        
        if not query:  # Return empty if nothing left after sanitization
            return ""
        
        if exact_phrase:
            # Exact phrase search - double quotes are already escaped
            return f'"{query}"'
        elif prefix_match:
            # Prefix matching - add * to each term
            terms = query.split()
            sanitized_terms = []
            for term in terms:
                if term and term.isalnum():  # Only allow alphanumeric terms for prefix
                    sanitized_terms.append(f"{term}*")
            return " ".join(sanitized_terms) if sanitized_terms else ""
        else:
            # Standard search - return sanitized query
            return query
    
    def _sanitize_fts5_input(self, query: str) -> str:
        """Sanitize input to prevent FTS5 injection and other security issues.
        
        Args:
            query: Raw query string
            
        Returns:
            Sanitized query string safe for FTS5 use
        """
        import re
        
        # Escape double quotes (FTS5 special character)
        query = query.replace('"', '""')
        
        # Remove or escape other potentially dangerous FTS5 characters
        # FTS5 special characters: " * ( ) [ ] { } ^ ~ : + - 
        dangerous_chars = {
            '*': '',  # Remove wildcard unless explicitly added for prefix
            '(': '',  # Remove grouping
            ')': '', 
            '[': '',  # Remove column filters
            ']': '',
            '{': '',  # Remove proximity operators  
            '}': '',
            '^': '',  # Remove column filters
            '~': '',  # Remove proximity operators
            ':': '',  # Remove column specifiers
            '+': '',  # Remove required terms (can cause issues)
            '\\': '', # Remove escape characters
            ';': '',  # Remove SQL statement terminators
            '--': '', # Remove SQL comments
            '/*': '', # Remove SQL comments
            '*/': '', # Remove SQL comments
        }
        
        for char, replacement in dangerous_chars.items():
            query = query.replace(char, replacement)
        
        # Remove excessive whitespace and normalize
        query = ' '.join(query.split())
        
        # Remove dangerous SQL keywords (case-insensitive)
        # Apply this after character removal to catch compound cases like "test;DROP" -> "testDROP"
        sql_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 
            'TRUNCATE', 'EXEC', 'EXECUTE', 'UNION', 'SELECT', 'FROM',
            'WHERE', 'HAVING', 'ORDER', 'GROUP', 'INTO', 'VALUES',
            'TABLE', 'DATABASE', 'SCHEMA', 'INDEX', 'VIEW', 'TRIGGER'
        ]
        
        for keyword in sql_keywords:
            # Use word boundaries to avoid removing legitimate words containing these
            pattern = r'\b' + re.escape(keyword) + r'\b'
            query = re.sub(pattern, '', query, flags=re.IGNORECASE)
        
        # Remove excessive whitespace again after keyword removal
        query = ' '.join(query.split())
        
        # Additional safety: only allow alphanumeric, spaces, and basic punctuation
        # This regex allows letters, numbers, spaces, hyphens, underscores, periods, and escaped quotes
        query = re.sub(r'[^a-zA-Z0-9\s\-_."]', '', query)
        
        return query.strip()
    
    def search_raw(self, query: str, limit: int = 20) -> List[Tuple[str, float]]:
        """Raw search method for hybrid search integration.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of (chunk_id, score) tuples for hybrid merging
        """
        results = self._search_fts5(query, limit)
        return [(chunk_id, score) for chunk_id, score, _, _ in results]
    
    def create_index(self, force_rebuild: bool = False) -> bool:
        """Create or rebuild the FTS5 search index.
        
        Args:
            force_rebuild: Whether to rebuild existing index
            
        Returns:
            True if successful, False otherwise
        """
        if not self._check_fts5_availability():
            logger.error("Cannot create index: FTS5 not available")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Drop existing index if rebuilding
            if force_rebuild:
                cursor.execute("DROP TABLE IF EXISTS chunks_fts")
            
            # Create FTS5 virtual table
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    content,
                    file_path UNINDEXED
                )
            """)
            
            # Populate from chunks table
            cursor.execute("""
                INSERT OR REPLACE INTO chunks_fts (chunk_id, content, file_path)
                SELECT chunk_id, content, file_path 
                FROM chunks 
                WHERE current = 1
            """)
            
            conn.commit()
            conn.close()
            
            # Reset cache
            self._index_exists = None
            
            logger.info("FTS5 index created/updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create FTS5 index: {e}")
            return False
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters for lexical search.
        
        Args:
            params: Dictionary of parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        # Check if query parameter exists and is a string
        query = params.get("query")
        if not isinstance(query, str):
            return False
        
        # Check query length (basic DoS protection)
        if len(query.strip()) == 0 or len(query) > 1000:
            return False
        
        # Validate limit parameter if provided
        limit = params.get("limit")
        if limit is not None:
            try:
                limit_int = int(limit)
                if limit_int <= 0 or limit_int > 100:
                    return False
            except (ValueError, TypeError):
                return False
        
        # Validate boolean parameters
        for param in ["exact_phrase", "prefix_match"]:
            value = params.get(param)
            if value is not None and not isinstance(value, bool):
                return False
        
        return True