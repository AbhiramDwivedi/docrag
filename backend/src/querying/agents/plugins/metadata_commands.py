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
from enum import Enum

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from querying.agents.plugin import Plugin, PluginInfo
from shared.config import get_settings

logger = logging.getLogger(__name__)

# Specialized logger for SQL queries
sql_logger = logging.getLogger('sql.query')
plugin_logger = logging.getLogger('plugin.metadata')


class SortBy(Enum):
    """Supported sort criteria."""
    MODIFIED = "modified"
    CREATED = "created" 
    NAME = "name"
    SIZE = "size"
    TYPE = "type"


class SortOrder(Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


class TimeFilter(Enum):
    """Supported time filter keywords."""
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    RECENT = "recent"
    LATEST = "latest"
    NEWEST = "newest"


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
                # Route to unified find_files operation
                if operation == "find_files":
                    return self._find_files(params, db_connection)
                # Keep legacy operations for backward compatibility
                elif operation == "get_latest_files":
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
            name="metadata",
            description="Structured metadata operations with programmatic interface",
            version="2.0.0",
            capabilities=[
                "find_files",  # Primary unified operation
                "metadata_query",  # Main capability for tests
                "file_statistics",  # For performance tests
                "collection_analysis", # For introspection tests
                "file_counts", # For introspection tests
                "file_types", # For introspection tests
                # Legacy operations for backward compatibility
                "get_latest_files",
                "find_files_by_content", 
                "get_file_stats",
                "get_file_count",
                "get_file_types"
            ],
            parameters={
                # Primary find_files operation
                "operation": "str - The operation to execute (find_files recommended)",
                
                # File filtering
                "file_type": "str - File type filter (PDF, DOCX, MSG, PPTX, etc.)",
                "file_types": "list - Multiple file types",
                
                # Size filtering  
                "min_size_mb": "float - Minimum file size in MB",
                "max_size_mb": "float - Maximum file size in MB",
                "min_size_bytes": "int - Minimum file size in bytes",
                "max_size_bytes": "int - Maximum file size in bytes",
                
                # Date filtering
                "created_after": "str - ISO date or relative time (2024-01-01, last_week)",
                "created_before": "str - ISO date or relative time", 
                "modified_after": "str - ISO date or relative time",
                "modified_before": "str - ISO date or relative time",
                
                # Path filtering
                "path_contains": "str - Partial path match",
                "name_contains": "str - Filename contains text",
                
                # Email metadata (enhanced schema only)
                "sender_email": "str - Sender email address",
                "subject_contains": "str - Subject line keywords",
                
                # Output control
                "count": "int - Max results (default: 50, use -1 for all)",
                "sort_by": "str - Sort field: modified|created|name|size|type",
                "sort_order": "str - Sort direction: asc|desc",
                "include_full_path": "bool - Return absolute paths (default: true)",
                
                # Legacy parameters
                "time_filter": "str - Legacy time filter (last_week, last_month, etc.)",
                "keywords": "list - Legacy content keywords"
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
    
    def _parse_date_filter(self, date_str: str) -> Optional[datetime]:
        """Parse date string into datetime object.
        
        Supports both ISO dates (2024-01-01) and relative keywords (last_week).
        """
        if not date_str:
            return None
            
        # Try relative keywords first
        if date_str in [e.value for e in TimeFilter]:
            return self._parse_time_filter(date_str)
        
        # Try ISO date formats
        try:
            # Try various date formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
        except Exception:
            pass
            
        logger.warning(f"Could not parse date string: {date_str}")
        return None
    
    def _find_files(self, params: Dict[str, Any], db_connection) -> Dict[str, Any]:
        """Unified file finder with comprehensive metadata filtering.
        
        This is the primary operation that supports all metadata-based filtering.
        """
        cursor = db_connection.cursor()
        
        # Extract parameters with defaults
        file_type = params.get("file_type")
        file_types = params.get("file_types", [])
        
        # Size filtering
        min_size_mb = params.get("min_size_mb")
        max_size_mb = params.get("max_size_mb")
        min_size_bytes = params.get("min_size_bytes")
        max_size_bytes = params.get("max_size_bytes")
        
        # Date filtering
        created_after = params.get("created_after")
        created_before = params.get("created_before")
        modified_after = params.get("modified_after")
        modified_before = params.get("modified_before")
        
        # Path filtering
        path_contains = params.get("path_contains")
        name_contains = params.get("name_contains")
        
        # Email metadata
        sender_email = params.get("sender_email")
        subject_contains = params.get("subject_contains")
        
        # Time filtering
        time_filter = params.get("time_filter")
        
        # Output control
        count = params.get("count", 50)
        if count == -1:
            count = None  # No limit
        sort_by = params.get("sort_by", SortBy.MODIFIED.value)
        sort_order = params.get("sort_order", SortOrder.DESC.value)
        include_full_path = params.get("include_full_path", True)
        
        # Convert size MB to bytes
        if min_size_mb:
            min_size_bytes = int(min_size_mb * 1024 * 1024)
        if max_size_mb:
            max_size_bytes = int(max_size_mb * 1024 * 1024)
        
        # Check if enhanced schema is available
        has_enhanced = self._has_enhanced_schema(db_connection)
        
        conditions = []
        query_params = []
        
        if has_enhanced:
            # Use enhanced schema with files table
            base_query = "SELECT f.file_name, f.file_path, f.file_size, f.modified_time, f.file_type, f.created_time"
            
            # Add email-specific fields if available
            try:
                cursor.execute("PRAGMA table_info(files)")
                columns = [col[1] for col in cursor.fetchall()]
                if "sender_email" in columns:
                    base_query += ", f.sender_email, f.subject"
                else:
                    base_query += ", NULL as sender_email, NULL as subject"
            except:
                base_query += ", NULL as sender_email, NULL as subject"
                
            base_query += " FROM files f WHERE 1=1"
            
            # File type filtering
            if file_type or file_types:
                types_to_filter = [file_type] if file_type else []
                types_to_filter.extend(file_types)
                if types_to_filter:
                    type_conditions = []
                    for ft in types_to_filter:
                        type_conditions.append("f.file_type = ?")
                        query_params.append(ft.upper())
                    conditions.append(f"({' OR '.join(type_conditions)})")
            
            # Size filtering
            if min_size_bytes:
                conditions.append("f.file_size >= ?")
                query_params.append(min_size_bytes)
            if max_size_bytes:
                conditions.append("f.file_size <= ?")
                query_params.append(max_size_bytes)
            
            # Date filtering
            if created_after:
                date_after = self._parse_date_filter(created_after)
                if date_after:
                    conditions.append("f.created_time >= ?")
                    query_params.append(date_after.timestamp())
            if created_before:
                date_before = self._parse_date_filter(created_before)
                if date_before:
                    conditions.append("f.created_time <= ?")
                    query_params.append(date_before.timestamp())
            if modified_after:
                date_after = self._parse_date_filter(modified_after)
                if date_after:
                    conditions.append("f.modified_time >= ?")
                    query_params.append(date_after.timestamp())
            if modified_before:
                date_before = self._parse_date_filter(modified_before)
                if date_before:
                    conditions.append("f.modified_time <= ?")
                    query_params.append(date_before.timestamp())
            
            # Path filtering
            if path_contains:
                conditions.append("f.file_path LIKE ?")
                query_params.append(f"%{path_contains}%")
            if name_contains:
                conditions.append("f.file_name LIKE ?")
                query_params.append(f"%{name_contains}%")
            
            # Email metadata filtering
            if sender_email:
                conditions.append("f.sender_email = ?")
                query_params.append(sender_email)
            if subject_contains:
                conditions.append("f.subject LIKE ?")
                query_params.append(f"%{subject_contains}%")
                
        else:
            # Use basic schema with chunks table
            base_query = "SELECT DISTINCT file, MAX(mtime) as latest_mtime"
            base_query += " FROM chunks WHERE current = 1"
            
            # File type filtering
            if file_type or file_types:
                types_to_filter = [file_type] if file_type else []
                types_to_filter.extend(file_types)
                if types_to_filter:
                    ext_map = {
                        "PDF": ".pdf", "DOCX": ".docx", "DOC": ".doc",
                        "XLSX": ".xlsx", "XLS": ".xls", "PPTX": ".pptx",
                        "PPT": ".ppt", "MSG": ".msg", "TXT": ".txt"
                    }
                    type_conditions = []
                    for ft in types_to_filter:
                        ext = ext_map.get(ft.upper())
                        if ext:
                            type_conditions.append("file LIKE ?")
                            query_params.append(f"%{ext}")
                    if type_conditions:
                        conditions.append(f"({' OR '.join(type_conditions)})")
            
            # Path filtering
            if path_contains:
                conditions.append("file LIKE ?")
                query_params.append(f"%{path_contains}%")
            if name_contains:
                # Extract filename from path for filtering
                conditions.append("SUBSTR(file, INSTR(file, '/') + 1) LIKE ?")
                query_params.append(f"%{name_contains}%")
            
            # Note: Size and email filtering not available in basic schema
        
        # Build final query with conditions and appropriate GROUP BY
        if conditions:
            query = base_query + " AND " + " AND ".join(conditions)
        else:
            query = base_query
        
        # Add GROUP BY only for basic schema (chunks table needs grouping)
        if not has_enhanced:
            query += " GROUP BY file"
        
        # Add sorting
        sort_column_map = {
            SortBy.MODIFIED.value: "f.modified_time" if has_enhanced else "latest_mtime",
            SortBy.CREATED.value: "f.created_time" if has_enhanced else "latest_mtime", 
            SortBy.NAME.value: "f.file_name" if has_enhanced else "file",
            SortBy.SIZE.value: "f.file_size" if has_enhanced else "latest_mtime",
            SortBy.TYPE.value: "f.file_type" if has_enhanced else "file"
        }
        
        sort_col = sort_column_map.get(sort_by, sort_column_map[SortBy.MODIFIED.value])
        sort_dir = "DESC" if sort_order.upper() == SortOrder.DESC.value.upper() else "ASC"
        query += f" ORDER BY {sort_col} {sort_dir}"
        
        # Add limit
        if count:
            query += " LIMIT ?"
            query_params.append(count)
        
        # Execute query
        plugin_logger.info(f"Operation: {params.get('operation', 'find_files')}")
        plugin_logger.info(f"Parameters: file_type={file_type}, count={count}, time_filter={time_filter}")
        
        # Log the SQL query for debug level
        sql_logger.debug(f"Executing query with {len(query_params)} parameters", extra={
            'sql_query': query,
            'sql_params': query_params
        })
        
        cursor.execute(query, query_params)
        results = cursor.fetchall()
        
        plugin_logger.info(f"Found {len(results)} files matching criteria")
        
        if not results:
            return {
                "response": "No files found matching the specified criteria.",
                "data": {"files": []},
                "metadata": {"operation": "find_files", "count": 0, "total_count": 0}
            }
        
        # Format results
        files_data = []
        file_list = []
        
        for row in results:
            if has_enhanced:
                file_name, file_path, file_size, modified_time, file_type_db = row[:5]
                created_time = row[5] if len(row) > 5 else modified_time
                sender_email_val = row[6] if len(row) > 6 else None
                subject_val = row[7] if len(row) > 7 else None
                
                file_info = {
                    "name": file_name,
                    "path": file_path if include_full_path else file_name,
                    "size": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2) if file_size else 0,
                    "modified": modified_time,
                    "created": created_time,
                    "type": file_type_db
                }
                
                if sender_email_val:
                    file_info["sender"] = sender_email_val
                if subject_val:
                    file_info["subject"] = subject_val
                
                try:
                    date_str = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M")
                    size_mb = file_info["size_mb"]
                    file_list.append(f"• {file_name} ({size_mb}MB, {date_str})")
                except (ValueError, OSError):
                    file_list.append(f"• {file_name}")
                    
            else:
                # Basic schema
                file_path, modified_time = row
                file_name = Path(file_path).name
                
                file_info = {
                    "name": file_name,
                    "path": file_path if include_full_path else file_name,
                    "modified": modified_time
                }
                
                try:
                    date_str = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M")
                    file_list.append(f"• {file_name} ({date_str})")
                except (ValueError, OSError):
                    file_list.append(f"• {file_name}")
            
            files_data.append(file_info)
        
        # Build response
        response_parts = []
        if count and len(results) == count:
            response_parts.append(f"Found {len(results)} files (showing first {count}):")
        else:
            response_parts.append(f"Found {len(results)} files:")
        
        response_parts.extend(file_list[:10])  # Show first 10 in summary
        if len(file_list) > 10:
            response_parts.append(f"... and {len(file_list) - 10} more files")
        
        return {
            "response": "\n".join(response_parts),
            "data": {
                "files": files_data,
                "total_count": len(results),
                "returned_count": len(results)
            },
            "metadata": {
                "operation": "find_files",
                "count": len(results),
                "schema_type": "enhanced" if has_enhanced else "basic",
                "filters_applied": {k: v for k, v in params.items() if v is not None and k != "operation"}
            }
        }
    
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