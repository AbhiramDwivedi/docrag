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
            parameters={
                "query": {"type": "string", "required": True, "description": "Search query"},
                "limit": {"type": "integer", "default": 20, "description": "Maximum results"},
                "exact_phrase": {"type": "boolean", "default": False, "description": "Use exact phrase matching"},
                "prefix_match": {"type": "boolean", "default": False, "description": "Enable prefix matching"}
            }
        )
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute lexical search.
        
        Args:
            params: Search parameters including query, limit, etc.
            
        Returns:
            Search results with lexical scores and metadata
        """
        query = params.get("query", "").strip()
        limit = params.get("limit", 20)
        exact_phrase = params.get("exact_phrase", False)
        prefix_match = params.get("prefix_match", False)
        
        if not query:
            return {
                "response": "No search query provided.",
                "sources": [],
                "metadata": {"error": "empty_query", "results_count": 0}
            }
        
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
            results = self._search_fts5(query, limit, exact_phrase, prefix_match)
            
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
            logger.error(f"Lexical search failed: {e}")
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Prepare FTS5 query
            fts_query = self._prepare_fts5_query(query, exact_phrase, prefix_match)
            
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
            logger.error(f"FTS5 query failed: {e}")
            logger.error(f"Query was: {fts_query}")
            return []
        finally:
            conn.close()
    
    def _prepare_fts5_query(self, query: str, exact_phrase: bool = False, 
                           prefix_match: bool = False) -> str:
        """Prepare query string for FTS5 MATCH syntax.
        
        Args:
            query: Original search query
            exact_phrase: Whether to use exact phrase matching
            prefix_match: Whether to enable prefix matching
            
        Returns:
            FTS5-formatted query string
        """
        # Escape special FTS5 characters
        query = query.replace('"', '""')
        
        if exact_phrase:
            # Exact phrase search
            return f'"{query}"'
        elif prefix_match:
            # Prefix matching - add * to each term
            terms = query.split()
            return " ".join(f"{term}*" for term in terms)
        else:
            # Standard search - let FTS5 handle it
            # For boolean queries (AND, OR, NOT), return as-is
            # For simple queries, FTS5 treats space as implicit AND
            return query
    
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