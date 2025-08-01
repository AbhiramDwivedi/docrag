"""Metadata plugin for DocQuest agent framework.

This plugin provides basic metadata queries about the document collection,
such as file counts, types, and simple statistics.
"""

import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.plugin import Plugin, PluginInfo
from config.config import settings

logger = logging.getLogger(__name__)


class MetadataPlugin(Plugin):
    """Plugin for document collection metadata queries.
    
    This plugin handles queries about file statistics, types, counts,
    and other metadata-based information about the document collection.
    """
    
    def __init__(self):
        """Initialize the metadata plugin."""
        # Don't store connection as instance variable due to thread safety
        pass
    
    def _get_db_connection(self):
        """Get a thread-safe database connection."""
        db_path = Path(settings.db_path)
        if not db_path.exists():
            return None
        # Create a new connection each time for thread safety
        return sqlite3.connect(db_path)
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute metadata query for the given parameters.
        
        Args:
            params: Dictionary containing:
                - question: The question to analyze for metadata intent
                
        Returns:
            Dictionary containing:
                - response: The formatted answer
                - data: Raw data used to generate the response
                - metadata: Additional metadata about the query
        """
        question = params.get("question", "").lower()
        
        if not question.strip():
            return {
                "response": "No question provided.",
                "data": {},
                "metadata": {"error": "empty_question"}
            }
        
        try:
            # Get database connection
            db_connection = self._get_db_connection()
            if db_connection is None:
                return {
                    "response": "No document database found. Please run document ingestion first.",
                    "data": {},
                    "metadata": {"error": "no_database"}
                }
            
            # Analyze question and determine what metadata to return
            query_type = self._classify_metadata_query(question)
            
            try:
                if query_type == "file_count":
                    return self._handle_file_count(question, db_connection)
                elif query_type == "file_types":
                    return self._handle_file_types(question, db_connection)
                elif query_type == "file_list":
                    return self._handle_file_list(question, db_connection)
                elif query_type == "recent_files":
                    return self._handle_recent_files(question, db_connection)
                else:
                    return self._handle_general_stats(question, db_connection)
            finally:
                # Always close the connection
                db_connection.close()
                
        except Exception as e:
            logger.error(f"Error in metadata query: {e}")
            return {
                "response": f"❌ Error querying metadata: {e}",
                "data": {},
                "metadata": {"error": str(e)}
            }
    
    def get_info(self) -> PluginInfo:
        """Return plugin metadata and capabilities."""
        return PluginInfo(
            name="metadata",
            description="Document collection metadata queries and statistics",
            version="1.0.0",
            capabilities=[
                "metadata_query",
                "file_statistics",
                "collection_analysis",
                "file_counts",
                "file_types"
            ],
            parameters={
                "question": "str - The question to analyze for metadata intent"
            }
        )
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters before execution."""
        if "question" not in params:
            return False
        
        question = params.get("question")
        if not isinstance(question, str) or not question.strip():
            return False
        
        return True
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        # No persistent connections to clean up
        logger.info("Metadata plugin cleaned up")
    
    def _classify_metadata_query(self, question: str) -> str:
        """Classify the type of metadata query.
        
        Args:
            question: User question (already lowercased)
            
        Returns:
            Query type identifier
        """
        if any(keyword in question for keyword in ["how many", "count", "number of"]):
            return "file_count"
        elif any(keyword in question for keyword in ["file types", "what types", "types of files"]):
            return "file_types"
        elif any(keyword in question for keyword in ["list", "show me", "what files"]):
            return "file_list"
        elif any(keyword in question for keyword in ["recent", "latest", "newest", "modified"]):
            return "recent_files"
        else:
            return "general_stats"
    
    def _handle_file_count(self, question: str, db_connection) -> Dict[str, Any]:
        """Handle file counting queries."""
        cursor = db_connection.cursor()
        
        # Check if asking for specific file types
        if "pdf" in question:
            cursor.execute("SELECT COUNT(DISTINCT file) FROM chunks WHERE current = 1 AND file LIKE '%.pdf'")
            count = cursor.fetchone()[0]
            return {
                "response": f"There are {count} PDF files in the collection.",
                "data": {"pdf_count": count},
                "metadata": {"query_type": "file_count", "file_type": "pdf"}
            }
        elif "excel" in question or "xlsx" in question:
            cursor.execute("SELECT COUNT(DISTINCT file) FROM chunks WHERE current = 1 AND (file LIKE '%.xlsx' OR file LIKE '%.xls')")
            count = cursor.fetchone()[0]
            return {
                "response": f"There are {count} Excel files in the collection.",
                "data": {"excel_count": count},
                "metadata": {"query_type": "file_count", "file_type": "excel"}
            }
        elif "word" in question or "docx" in question:
            cursor.execute("SELECT COUNT(DISTINCT file) FROM chunks WHERE current = 1 AND file LIKE '%.docx'")
            count = cursor.fetchone()[0]
            return {
                "response": f"There are {count} Word documents in the collection.",
                "data": {"word_count": count},
                "metadata": {"query_type": "file_count", "file_type": "word"}
            }
        else:
            # Total file count
            cursor.execute("SELECT COUNT(DISTINCT file) FROM chunks WHERE current = 1")
            count = cursor.fetchone()[0]
            return {
                "response": f"There are {count} total files in the collection.",
                "data": {"total_count": count},
                "metadata": {"query_type": "file_count", "file_type": "all"}
            }
    
    def _handle_file_types(self, question: str, db_connection) -> Dict[str, Any]:
        """Handle file type enumeration queries."""
        cursor = db_connection.cursor()
        
        # Get file extensions
        cursor.execute("""
            SELECT LOWER(SUBSTR(file, -4)) as ext, COUNT(DISTINCT file) as count 
            FROM chunks 
            WHERE current = 1 
            GROUP BY LOWER(SUBSTR(file, -4))
            ORDER BY count DESC
        """)
        
        results = cursor.fetchall()
        
        if not results:
            return {
                "response": "No files found in the collection.",
                "data": {},
                "metadata": {"query_type": "file_types"}
            }
        
        # Format response
        type_info = []
        data = {}
        for ext, count in results:
            # Clean up extension
            ext = ext.lstrip('.')
            if ext and len(ext) <= 5:  # Reasonable extension length
                type_info.append(f"{count} {ext.upper()} file{'s' if count != 1 else ''}")
                data[ext] = count
        
        if type_info:
            response = "File types in the collection:\n" + "\n".join(f"• {info}" for info in type_info)
        else:
            response = "No recognizable file types found."
        
        return {
            "response": response,
            "data": data,
            "metadata": {"query_type": "file_types"}
        }
    
    def _handle_file_list(self, question: str, db_connection) -> Dict[str, Any]:
        """Handle file listing queries."""
        cursor = db_connection.cursor()
        
        # Limit to prevent overwhelming output
        limit = 20
        cursor.execute("SELECT DISTINCT file FROM chunks WHERE current = 1 ORDER BY file LIMIT ?", (limit,))
        files = [row[0] for row in cursor.fetchall()]
        
        if not files:
            return {
                "response": "No files found in the collection.",
                "data": {"files": []},
                "metadata": {"query_type": "file_list"}
            }
        
        # Get total count for context
        cursor.execute("SELECT COUNT(DISTINCT file) FROM chunks WHERE current = 1")
        total_count = cursor.fetchone()[0]
        
        # Format response
        file_list = "\n".join(f"• {Path(f).name}" for f in files[:limit])
        
        if total_count > limit:
            response = f"Files in the collection (showing {limit} of {total_count}):\n{file_list}\n\n... and {total_count - limit} more files"
        else:
            response = f"Files in the collection ({total_count} total):\n{file_list}"
        
        return {
            "response": response,
            "data": {"files": files, "total_count": total_count, "shown": len(files)},
            "metadata": {"query_type": "file_list"}
        }
    
    def _handle_recent_files(self, question: str, db_connection) -> Dict[str, Any]:
        """Handle recent files queries."""
        cursor = db_connection.cursor()
        
        # Get files ordered by modification time
        cursor.execute("""
            SELECT DISTINCT file, MAX(mtime) as latest_mtime 
            FROM chunks 
            WHERE current = 1 
            GROUP BY file 
            ORDER BY latest_mtime DESC 
            LIMIT 10
        """)
        
        results = cursor.fetchall()
        
        if not results:
            return {
                "response": "No files found in the collection.",
                "data": {"files": []},
                "metadata": {"query_type": "recent_files"}
            }
        
        # Format response
        import datetime
        recent_files = []
        for file, mtime in results:
            try:
                # Convert timestamp to readable date
                date_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                recent_files.append(f"• {Path(file).name} (modified: {date_str})")
            except (ValueError, OSError):
                recent_files.append(f"• {Path(file).name}")
        
        response = f"Most recently modified files:\n" + "\n".join(recent_files)
        
        return {
            "response": response,
            "data": {"files": [{"file": file, "mtime": mtime} for file, mtime in results]},
            "metadata": {"query_type": "recent_files"}
        }
    
    def _handle_general_stats(self, question: str, db_connection) -> Dict[str, Any]:
        """Handle general statistics queries."""
        cursor = db_connection.cursor()
        
        # Get comprehensive stats
        cursor.execute("SELECT COUNT(DISTINCT file) FROM chunks WHERE current = 1")
        file_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE current = 1")
        chunk_count = cursor.fetchone()[0]
        
        if file_count == 0:
            return {
                "response": "No files found in the collection.",
                "data": {},
                "metadata": {"query_type": "general_stats"}
            }
        
        # Get file type breakdown
        cursor.execute("""
            SELECT LOWER(SUBSTR(file, -4)) as ext, COUNT(DISTINCT file) as count 
            FROM chunks 
            WHERE current = 1 
            GROUP BY LOWER(SUBSTR(file, -4))
            ORDER BY count DESC
            LIMIT 5
        """)
        
        type_results = cursor.fetchall()
        type_summary = []
        for ext, count in type_results:
            ext = ext.lstrip('.')
            if ext and len(ext) <= 5:
                type_summary.append(f"{count} {ext.upper()}")
        
        response_lines = [
            f"Document Collection Statistics:",
            f"• Total files: {file_count}",
            f"• Total text chunks: {chunk_count}"
        ]
        
        if type_summary:
            response_lines.append(f"• File types: {', '.join(type_summary)}")
        
        response = "\n".join(response_lines)
        
        return {
            "response": response,
            "data": {
                "file_count": file_count,
                "chunk_count": chunk_count,
                "types": dict(type_results)
            },
            "metadata": {"query_type": "general_stats"}
        }