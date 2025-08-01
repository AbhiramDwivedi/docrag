"""Structured metadata operations for DocQuest agent framework.

This module provides structured, programmatic metadata operations that can be
invoked by the LLM agent. Instead of doing NLP parsing, this plugin exposes
clean interfaces for different metadata operations.
"""

import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.plugin import Plugin, PluginInfo
from config.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """File information structure."""
    name: str
    path: str
    size: int
    modified_time: float
    file_type: str


@dataclass
class MetadataQuery:
    """Structured metadata query parameters."""
    operation: str  # get_latest_files, find_files_by_content, get_file_stats, etc.
    file_type: Optional[str] = None
    count: Optional[int] = None
    time_filter: Optional[str] = None  # "last_week", "last_month", etc.
    keywords: Optional[List[str]] = None
    size_filter: Optional[Dict[str, Any]] = None  # {"operator": "larger", "size_mb": 10}


class MetadataCommandsPlugin(Plugin):
    """Structured metadata plugin with programmatic operations.
    
    This plugin provides clean, structured interfaces for metadata operations
    instead of trying to parse natural language directly.
    """
    
    def __init__(self):
        """Initialize the metadata commands plugin."""
        self.settings = get_settings()
    
    def _get_db_connection(self):
        """Get a thread-safe database connection."""
        db_path = Path(self.settings.db_path)
        if not db_path.exists():
            return None
        return sqlite3.connect(db_path)
    
    def _has_enhanced_schema(self, conn) -> bool:
        """Check if database has Phase 2 enhanced schema."""
        cur = conn.cursor()
        try:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files'")
            return cur.fetchone() is not None
        except:
            return False
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute structured metadata operation.
        
        Args:
            params: Dictionary containing:
                - operation: The specific operation to execute
                - Additional parameters based on operation type
                
        Returns:
            Dictionary containing:
                - response: The formatted answer
                - data: Raw data used to generate the response
                - metadata: Additional metadata about the query
        """
        operation = params.get("operation")
        if not operation:
            return {
                "response": "No operation specified.",
                "data": {},
                "metadata": {"error": "no_operation"}
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
            
            try:
                # Route to specific operation handler
                if operation == "get_latest_files":
                    return self._get_latest_files(params, db_connection)
                elif operation == "find_files_by_content":
                    return self._find_files_by_content(params, db_connection)
                elif operation == "get_file_stats":
                    return self._get_file_stats(params, db_connection)
                elif operation == "get_file_count":
                    return self._get_file_count(params, db_connection)
                elif operation == "get_file_types":
                    return self._get_file_types(params, db_connection)
                else:
                    return {
                        "response": f"Unknown operation: {operation}",
                        "data": {},
                        "metadata": {"error": "unknown_operation", "operation": operation}
                    }
            finally:
                db_connection.close()
                
        except Exception as e:
            logger.error(f"Error in metadata operation {operation}: {e}")
            return {
                "response": f"❌ Error in metadata operation: {e}",
                "data": {},
                "metadata": {"error": str(e), "operation": operation}
            }
    
    def get_info(self) -> PluginInfo:
        """Return plugin metadata and capabilities."""
        return PluginInfo(
            name="metadata_commands",
            description="Structured metadata operations with programmatic interface",
            version="1.0.0",
            capabilities=[
                "get_latest_files",
                "find_files_by_content",
                "get_file_stats",
                "get_file_count",
                "get_file_types"
            ],
            parameters={
                "operation": "str - The operation to execute",
                "file_type": "str - Optional file type filter (PDF, DOCX, MSG, PPTX, etc.)",
                "count": "int - Number of items to return",
                "time_filter": "str - Time filter (last_week, last_month, recent, etc.)",
                "keywords": "list - Keywords to search for",
                "size_filter": "dict - Size filter parameters"
            }
        )
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters before execution."""
        if "operation" not in params:
            return False
        
        operation = params.get("operation")
        if not isinstance(operation, str) or not operation.strip():
            return False
        
        # Additional validation per operation type could be added here
        return True
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        logger.info("Metadata commands plugin cleaned up")
    
    def _parse_time_filter(self, time_filter: str) -> Optional[datetime]:
        """Parse time filter keywords into datetime objects."""
        now = datetime.now()
        
        if time_filter == "today":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_filter == "yesterday":
            return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_filter == "this_week":
            days_since_monday = now.weekday()
            return (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_filter == "last_week":
            days_since_monday = now.weekday()
            return (now - timedelta(days=days_since_monday + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_filter == "last_month":
            return now - timedelta(days=30)
        elif time_filter in ["recent", "latest", "newest"]:
            return now - timedelta(days=7)
        
        return None
    
    def _get_latest_files(self, params: Dict[str, Any], db_connection) -> Dict[str, Any]:
        """Get latest modified files with optional filters."""
        cursor = db_connection.cursor()
        
        file_type = params.get("file_type")
        count = params.get("count", 10)
        time_filter = params.get("time_filter")
        
        conditions = []
        query_params = []
        
        # Check if enhanced schema is available
        has_enhanced = self._has_enhanced_schema(db_connection)
        
        if has_enhanced:
            # Use enhanced schema
            if file_type:
                conditions.append("f.file_type = ?")
                query_params.append(file_type.upper())
            
            if time_filter:
                date_after = self._parse_time_filter(time_filter)
                if date_after:
                    conditions.append("f.modified_time >= ?")
                    query_params.append(date_after.timestamp())
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT f.file_name, f.file_path, f.file_size, f.modified_time, f.file_type
                FROM files f 
                WHERE {where_clause}
                ORDER BY f.modified_time DESC
                LIMIT ?
            """
            query_params.append(count)
            
        else:
            # Use basic schema
            if file_type:
                ext_map = {
                    "PDF": ".pdf", "DOCX": ".docx", "DOC": ".doc", 
                    "XLSX": ".xlsx", "XLS": ".xls", "PPTX": ".pptx", 
                    "PPT": ".ppt", "MSG": ".msg", "TXT": ".txt"
                }
                ext = ext_map.get(file_type.upper())
                if ext:
                    conditions.append("file LIKE ?")
                    query_params.append(f"%{ext}")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            where_clause += " AND current = 1"
            
            query = f"""
                SELECT DISTINCT file, MAX(mtime) as latest_mtime 
                FROM chunks 
                WHERE {where_clause}
                GROUP BY file 
                ORDER BY latest_mtime DESC 
                LIMIT ?
            """
            query_params.append(count)
        
        cursor.execute(query, query_params)
        results = cursor.fetchall()
        
        if not results:
            filters_desc = []
            if file_type:
                filters_desc.append(f"{file_type} files")
            if time_filter:
                filters_desc.append(f"from {time_filter}")
            
            filter_text = " ".join(filters_desc) if filters_desc else "files"
            return {
                "response": f"No {filter_text} found.",
                "data": {"files": []},
                "metadata": {"operation": "get_latest_files", "count": 0}
            }
        
        # Format response
        file_list = []
        files_data = []
        
        for row in results:
            if has_enhanced:
                file_name, file_path, file_size, modified_time, file_type_db = row
                try:
                    date_str = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M")
                    size_mb = (file_size / (1024 * 1024)) if file_size else 0
                    file_list.append(f"• {file_name} ({size_mb:.1f}MB, {date_str})")
                    files_data.append({
                        "name": file_name, "path": file_path, "size": file_size,
                        "modified": modified_time, "type": file_type_db
                    })
                except (ValueError, OSError):
                    file_list.append(f"• {file_name}")
                    files_data.append({"name": file_name, "path": file_path})
            else:
                file_path, modified_time = row
                file_name = Path(file_path).name
                try:
                    date_str = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M")
                    file_list.append(f"• {file_name} ({date_str})")
                except (ValueError, OSError):
                    file_list.append(f"• {file_name}")
                files_data.append({"name": file_name, "path": file_path, "modified": modified_time})
        
        # Build description
        filters_desc = []
        if file_type:
            filters_desc.append(f"{file_type} files")
        else:
            filters_desc.append("files")
        if time_filter:
            filters_desc.append(f"from {time_filter}")
        
        filter_text = " ".join(filters_desc)
        response = f"Latest {filter_text} ({len(results)} found):\n" + "\n".join(file_list)
        
        return {
            "response": response,
            "data": {"files": files_data},
            "metadata": {"operation": "get_latest_files", "count": len(results)}
        }
    
    def _find_files_by_content(self, params: Dict[str, Any], db_connection) -> Dict[str, Any]:
        """Find files containing specific keywords."""
        cursor = db_connection.cursor()
        
        keywords = params.get("keywords", [])
        file_type = params.get("file_type")
        count = params.get("count", 20)
        
        if not keywords:
            return {
                "response": "No keywords specified for content search.",
                "data": {"files": []},
                "metadata": {"operation": "find_files_by_content", "error": "no_keywords"}
            }
        
        conditions = []
        query_params = []
        
        # Search in chunk content
        keyword_conditions = []
        for keyword in keywords:
            keyword_conditions.append("content LIKE ?")
            query_params.append(f"%{keyword}%")
        
        conditions.append(f"({' OR '.join(keyword_conditions)})")
        conditions.append("current = 1")
        
        # Add file type filter if specified
        if file_type:
            ext_map = {
                "PDF": ".pdf", "DOCX": ".docx", "DOC": ".doc", 
                "XLSX": ".xlsx", "XLS": ".xls", "PPTX": ".pptx", 
                "PPT": ".ppt", "MSG": ".msg", "TXT": ".txt"
            }
            ext = ext_map.get(file_type.upper())
            if ext:
                conditions.append("file LIKE ?")
                query_params.append(f"%{ext}")
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT DISTINCT file, MAX(mtime) as latest_mtime
            FROM chunks 
            WHERE {where_clause}
            GROUP BY file 
            ORDER BY latest_mtime DESC 
            LIMIT ?
        """
        query_params.append(count)
        
        cursor.execute(query, query_params)
        results = cursor.fetchall()
        
        if not results:
            keywords_text = ", ".join(keywords)
            type_text = f" {file_type}" if file_type else ""
            return {
                "response": f"No{type_text} files found containing keywords: {keywords_text}",
                "data": {"files": []},
                "metadata": {"operation": "find_files_by_content", "keywords": keywords}
            }
        
        # Format response
        file_list = []
        files_data = []
        
        for file_path, modified_time in results:
            file_name = Path(file_path).name
            try:
                date_str = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d")
                file_list.append(f"• {file_name} ({date_str})")
            except (ValueError, OSError):
                file_list.append(f"• {file_name}")
            files_data.append({"name": file_name, "path": file_path, "modified": modified_time})
        
        keywords_text = ", ".join(keywords)
        type_text = f" {file_type}" if file_type else ""
        response = f"Files{type_text} containing '{keywords_text}' ({len(results)} found):\n" + "\n".join(file_list)
        
        return {
            "response": response,
            "data": {"files": files_data},
            "metadata": {"operation": "find_files_by_content", "count": len(results), "keywords": keywords}
        }
    
    def _get_file_stats(self, params: Dict[str, Any], db_connection) -> Dict[str, Any]:
        """Get file statistics with optional filters."""
        cursor = db_connection.cursor()
        
        # Check if enhanced schema is available
        has_enhanced = self._has_enhanced_schema(db_connection)
        
        if has_enhanced:
            # Enhanced statistics
            cursor.execute("SELECT COUNT(*) FROM files")
            total_files = cursor.fetchone()[0]
            
            cursor.execute("SELECT COALESCE(SUM(file_size), 0) FROM files")
            total_size = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT file_type, COUNT(*) as count, 
                       COALESCE(AVG(file_size), 0) as avg_size
                FROM files
                GROUP BY file_type
                ORDER BY count DESC
                LIMIT 5
            """)
            type_stats = cursor.fetchall()
            
        else:
            # Basic statistics
            cursor.execute("SELECT COUNT(DISTINCT file) FROM chunks WHERE current = 1")
            total_files = cursor.fetchone()[0]
            
            total_size = 0  # Not available in basic schema
            
            cursor.execute("""
                SELECT LOWER(SUBSTR(file, -4)) as ext, COUNT(DISTINCT file) as count 
                FROM chunks 
                WHERE current = 1 
                GROUP BY LOWER(SUBSTR(file, -4))
                ORDER BY count DESC
                LIMIT 5
            """)
            type_results = cursor.fetchall()
            type_stats = [(ext.lstrip('.').upper(), count, 0) for ext, count in type_results if ext and len(ext.lstrip('.')) <= 5]
        
        if total_files == 0:
            return {
                "response": "No files found in the collection.",
                "data": {},
                "metadata": {"operation": "get_file_stats"}
            }
        
        # Format response
        total_mb = total_size / (1024 * 1024) if total_size else 0
        
        response_lines = [
            f"Document Collection Statistics:",
            f"• Total files: {total_files}"
        ]
        
        if total_size > 0:
            response_lines.append(f"• Total size: {total_mb:.1f}MB")
        
        if type_stats:
            response_lines.append("• File types:")
            for file_type, count, avg_size in type_stats:
                if avg_size > 0:
                    avg_mb = avg_size / (1024 * 1024)
                    response_lines.append(f"  - {file_type}: {count} files (avg: {avg_mb:.1f}MB)")
                else:
                    response_lines.append(f"  - {file_type}: {count} files")
        
        response = "\n".join(response_lines)
        
        return {
            "response": response,
            "data": {
                "total_files": total_files,
                "total_size": total_size,
                "type_stats": [{"type": r[0], "count": r[1], "avg_size": r[2]} for r in type_stats]
            },
            "metadata": {"operation": "get_file_stats"}
        }
    
    def _get_file_count(self, params: Dict[str, Any], db_connection) -> Dict[str, Any]:
        """Get count of files with optional filters."""
        cursor = db_connection.cursor()
        
        file_type = params.get("file_type")
        time_filter = params.get("time_filter")
        
        conditions = ["current = 1"]
        query_params = []
        
        if file_type:
            ext_map = {
                "PDF": ".pdf", "DOCX": ".docx", "DOC": ".doc", 
                "XLSX": ".xlsx", "XLS": ".xls", "PPTX": ".pptx", 
                "PPT": ".ppt", "MSG": ".msg", "TXT": ".txt"
            }
            ext = ext_map.get(file_type.upper())
            if ext:
                conditions.append("file LIKE ?")
                query_params.append(f"%{ext}")
        
        where_clause = " AND ".join(conditions)
        
        query = f"SELECT COUNT(DISTINCT file) FROM chunks WHERE {where_clause}"
        cursor.execute(query, query_params)
        count = cursor.fetchone()[0]
        
        # Build description
        if file_type:
            file_desc = f"{file_type} files"
        else:
            file_desc = "files"
        
        return {
            "response": f"There are {count} {file_desc} in the collection.",
            "data": {"count": count, "file_type": file_type},
            "metadata": {"operation": "get_file_count"}
        }
    
    def _get_file_types(self, params: Dict[str, Any], db_connection) -> Dict[str, Any]:
        """Get list of file types in the collection."""
        cursor = db_connection.cursor()
        
        # Check if enhanced schema is available
        has_enhanced = self._has_enhanced_schema(db_connection)
        
        if has_enhanced:
            cursor.execute("""
                SELECT file_type, COUNT(*) as count
                FROM files
                GROUP BY file_type
                ORDER BY count DESC
            """)
            results = cursor.fetchall()
        else:
            cursor.execute("""
                SELECT LOWER(SUBSTR(file, -4)) as ext, COUNT(DISTINCT file) as count 
                FROM chunks 
                WHERE current = 1 
                GROUP BY LOWER(SUBSTR(file, -4))
                ORDER BY count DESC
            """)
            type_results = cursor.fetchall()
            results = [(ext.lstrip('.').upper(), count) for ext, count in type_results if ext and len(ext.lstrip('.')) <= 5]
        
        if not results:
            return {
                "response": "No file types found in the collection.",
                "data": {},
                "metadata": {"operation": "get_file_types"}
            }
        
        # Format response
        type_lines = []
        for file_type, count in results:
            type_lines.append(f"• {file_type}: {count} file{'s' if count != 1 else ''}")
        
        response = "File types in the collection:\n" + "\n".join(type_lines)
        
        return {
            "response": response,
            "data": {"types": [{"type": r[0], "count": r[1]} for r in results]},
            "metadata": {"operation": "get_file_types"}
        }