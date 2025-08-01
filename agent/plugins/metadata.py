"""Enhanced metadata plugin for DocQuest agent framework Phase 2.

This plugin provides comprehensive metadata queries about the document collection,
including advanced filtering by dates, senders, file types, and complex searches.
Supports both basic queries from Phase 1 and enhanced queries from Phase 2.
"""

import logging
import sqlite3
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.plugin import Plugin, PluginInfo
from config.config import get_settings

logger = logging.getLogger(__name__)


class MetadataPlugin(Plugin):
    """Enhanced plugin for document collection metadata queries.
    
    This plugin handles both simple and complex metadata queries:
    - Basic: "how many files", "what file types"
    - Advanced: "emails from John last week", "newest Excel files"
    - Statistical: file size distributions, activity patterns
    """
    
    def __init__(self):
        """Initialize the enhanced metadata plugin."""
        # Don't store connection as instance variable due to thread safety
        self.settings = get_settings()
    
    def _get_db_connection(self):
        """Get a thread-safe database connection."""
        db_path = Path(self.settings.db_path)
        if not db_path.exists():
            return None
        # Create a new connection each time for thread safety
        return sqlite3.connect(db_path)
    
    def _has_enhanced_schema(self, conn) -> bool:
        """Check if database has Phase 2 enhanced schema."""
        cur = conn.cursor()
        try:
            # Check if files table exists
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files'")
            return cur.fetchone() is not None
        except:
            return False
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced metadata query for the given parameters.
        
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
            
            # Check if enhanced schema is available
            has_enhanced = self._has_enhanced_schema(db_connection)
            
            # Analyze question and determine what metadata to return
            query_type, query_params = self._classify_enhanced_metadata_query(question)
            
            try:
                if has_enhanced:
                    return self._handle_enhanced_query(query_type, query_params, question, db_connection)
                else:
                    return self._handle_basic_query(query_type, question, db_connection)
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
            description="Enhanced document collection metadata queries and statistics",
            version="2.0.0",
            capabilities=[
                "metadata_query",
                "file_statistics", 
                "collection_analysis",
                "file_counts",
                "file_types",
                "email_analysis",  # Phase 2
                "date_filtering",   # Phase 2
                "size_filtering",   # Phase 2
                "sender_analysis",  # Phase 2
                "temporal_queries"  # Phase 2
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
        logger.info("Enhanced metadata plugin cleaned up")
    
    def _classify_enhanced_metadata_query(self, question: str) -> tuple:
        """Classify enhanced metadata query and extract parameters.
        
        Args:
            question: User question (already lowercased)
            
        Returns:
            Tuple of (query_type, query_params)
        """
        query_params = {}
        
        # Email-specific queries
        if any(keyword in question for keyword in ["email", "emails", "mail"]):
            if "from" in question or "sender" in question:
                # Extract sender information
                sender_match = re.search(r"from ([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", question)
                if sender_match:
                    query_params["sender_email"] = sender_match.group(1)
                else:
                    # Look for names after "from"
                    name_match = re.search(r"from ([a-zA-Z\s]+?)(?:\s|$|about|regarding)", question)
                    if name_match:
                        query_params["sender_name"] = name_match.group(1).strip()
                
                return "email_by_sender", query_params
            
            elif "about" in question or "regarding" in question or "subject" in question:
                # Extract subject keywords
                subject_patterns = [
                    r"about ([^,\.]+)",
                    r"regarding ([^,\.]+)", 
                    r"subject.*?([^,\.]+)"
                ]
                for pattern in subject_patterns:
                    match = re.search(pattern, question)
                    if match:
                        query_params["subject_contains"] = match.group(1).strip()
                        break
                
                return "email_by_subject", query_params
            
            else:
                return "email_general", query_params
        
        # Date/time-based queries
        date_keywords = ["last week", "last month", "recent", "latest", "newest", "today", "yesterday", "this week"]
        for keyword in date_keywords:
            if keyword in question:
                query_params["time_filter"] = keyword
                break
        
        # File size queries 
        if any(keyword in question for keyword in ["larger than", "bigger than", "smaller than", "size"]):
            size_match = re.search(r"(\d+)\s*(mb|gb|kb|bytes?)", question)
            if size_match:
                size_value = int(size_match.group(1))
                unit = size_match.group(2).lower()
                multiplier = {"kb": 1024, "mb": 1024*1024, "gb": 1024*1024*1024, "bytes": 1, "byte": 1}
                query_params["size_bytes"] = size_value * multiplier.get(unit, 1)
                query_params["size_operator"] = "larger" if "larger" in question or "bigger" in question else "smaller"
        
        # File type specific queries
        file_types = ["pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt", "txt", "email"]
        for file_type in file_types:
            if file_type in question:
                query_params["file_type"] = file_type.upper()
                break
        
        # Basic query classification
        if any(keyword in question for keyword in ["how many", "count", "number of"]):
            return "file_count", query_params
        elif any(keyword in question for keyword in ["file types", "what types", "types of files"]):
            return "file_types", query_params
        elif any(keyword in question for keyword in ["list", "show me", "what files"]):
            return "file_list", query_params
        elif any(keyword in question for keyword in ["recent", "latest", "newest", "modified"]):
            return "recent_files", query_params
        else:
            return "general_stats", query_params
    
    def _handle_enhanced_query(self, query_type: str, query_params: Dict[str, Any], question: str, db_connection) -> Dict[str, Any]:
        """Handle enhanced Phase 2 metadata queries."""
        try:
            if query_type == "email_by_sender":
                return self._handle_emails_by_sender(query_params, db_connection)
            elif query_type == "email_by_subject":
                return self._handle_emails_by_subject(query_params, db_connection)
            elif query_type == "email_general":
                return self._handle_general_email_stats(db_connection)
            elif query_type == "file_count":
                return self._handle_enhanced_file_count(query_params, db_connection)
            elif query_type == "file_list":
                return self._handle_enhanced_file_list(query_params, db_connection)
            elif query_type == "recent_files":
                return self._handle_enhanced_recent_files(query_params, db_connection)
            elif query_type == "file_types":
                return self._handle_enhanced_file_types(query_params, db_connection)
            else:
                return self._handle_enhanced_general_stats(db_connection)
        except Exception as e:
            logger.error(f"Error in enhanced query {query_type}: {e}")
            return {
                "response": f"❌ Error processing enhanced query: {e}",
                "data": {},
                "metadata": {"error": str(e), "query_type": query_type}
            }
    
    def _handle_basic_query(self, query_type: str, question: str, db_connection) -> Dict[str, Any]:
        """Handle basic Phase 1 metadata queries for backward compatibility."""
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
        except Exception as e:
            logger.error(f"Error in basic query {query_type}: {e}")
            return {
                "response": f"❌ Error processing query: {e}",
                "data": {},
                "metadata": {"error": str(e), "query_type": query_type}
            }
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
    
    # Enhanced Phase 2 query handlers
    
    def _parse_time_filter(self, time_filter: str) -> Optional[datetime]:
        """Parse time filter keywords into datetime objects."""
        now = datetime.now()
        
        if time_filter == "today":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_filter == "yesterday":
            return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_filter == "this week":
            # Start of current week (Monday)
            days_since_monday = now.weekday()
            return (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_filter == "last week":
            # Start of last week
            days_since_monday = now.weekday()
            return (now - timedelta(days=days_since_monday + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_filter == "last month":
            # Approximate - 30 days ago
            return now - timedelta(days=30)
        elif time_filter in ["recent", "latest", "newest"]:
            # Last 7 days
            return now - timedelta(days=7)
        
        return None
    
    def _handle_emails_by_sender(self, query_params: Dict[str, Any], db_connection) -> Dict[str, Any]:
        """Handle email queries filtered by sender."""
        cursor = db_connection.cursor()
        
        conditions = []
        params = []
        
        if "sender_email" in query_params:
            conditions.append("e.sender_email = ?")
            params.append(query_params["sender_email"])
        elif "sender_name" in query_params:
            conditions.append("e.sender_name LIKE ?")
            params.append(f"%{query_params['sender_name']}%")
        
        # Add time filter if present
        if "time_filter" in query_params:
            date_after = self._parse_time_filter(query_params["time_filter"])
            if date_after:
                conditions.append("e.email_date >= ?")
                params.append(date_after.timestamp())
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT e.sender_email, e.sender_name, e.subject, e.email_date, f.file_name
            FROM email_metadata e
            JOIN files f ON e.file_path = f.file_path
            WHERE {where_clause}
            ORDER BY e.email_date DESC
            LIMIT 20
        """
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        if not results:
            sender_info = query_params.get("sender_email") or query_params.get("sender_name", "the specified sender")
            time_info = f" {query_params.get('time_filter', '')}" if "time_filter" in query_params else ""
            return {
                "response": f"No emails found from {sender_info}{time_info}.",
                "data": {"emails": []},
                "metadata": {"query_type": "email_by_sender", "sender": sender_info}
            }
        
        # Format response
        email_list = []
        for sender_email, sender_name, subject, email_date, file_name in results:
            try:
                date_str = datetime.fromtimestamp(email_date).strftime("%Y-%m-%d %H:%M")
                display_name = sender_name or sender_email
                email_list.append(f"• {subject} - from {display_name} ({date_str})")
            except (ValueError, OSError):
                display_name = sender_name or sender_email  
                email_list.append(f"• {subject} - from {display_name}")
        
        sender_info = query_params.get("sender_email") or query_params.get("sender_name", "sender")
        time_info = f" {query_params.get('time_filter', '')}" if "time_filter" in query_params else ""
        response = f"Emails from {sender_info}{time_info} ({len(results)} found):\n" + "\n".join(email_list)
        
        return {
            "response": response,
            "data": {"emails": [{"sender_email": r[0], "sender_name": r[1], "subject": r[2], 
                                "email_date": r[3], "file_name": r[4]} for r in results]},
            "metadata": {"query_type": "email_by_sender", "count": len(results)}
        }
    
    def _handle_emails_by_subject(self, query_params: Dict[str, Any], db_connection) -> Dict[str, Any]:
        """Handle email queries filtered by subject content."""
        cursor = db_connection.cursor()
        
        subject_contains = query_params.get("subject_contains", "")
        if not subject_contains:
            return {
                "response": "No subject keywords specified for email search.",
                "data": {"emails": []},
                "metadata": {"query_type": "email_by_subject", "error": "no_subject"}
            }
        
        conditions = ["e.subject LIKE ?"]
        params = [f"%{subject_contains}%"]
        
        # Add time filter if present
        if "time_filter" in query_params:
            date_after = self._parse_time_filter(query_params["time_filter"])
            if date_after:
                conditions.append("e.email_date >= ?")
                params.append(date_after.timestamp())
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT e.sender_email, e.sender_name, e.subject, e.email_date, f.file_name
            FROM email_metadata e
            JOIN files f ON e.file_path = f.file_path
            WHERE {where_clause}
            ORDER BY e.email_date DESC
            LIMIT 20
        """
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        if not results:
            time_info = f" {query_params.get('time_filter', '')}" if "time_filter" in query_params else ""
            return {
                "response": f"No emails found with subject containing '{subject_contains}'{time_info}.",
                "data": {"emails": []},
                "metadata": {"query_type": "email_by_subject", "subject": subject_contains}
            }
        
        # Format response
        email_list = []
        for sender_email, sender_name, subject, email_date, file_name in results:
            try:
                date_str = datetime.fromtimestamp(email_date).strftime("%Y-%m-%d %H:%M")
                display_name = sender_name or sender_email
                email_list.append(f"• {subject} - from {display_name} ({date_str})")
            except (ValueError, OSError):
                display_name = sender_name or sender_email
                email_list.append(f"• {subject} - from {display_name}")
        
        time_info = f" {query_params.get('time_filter', '')}" if "time_filter" in query_params else ""
        response = f"Emails about '{subject_contains}'{time_info} ({len(results)} found):\n" + "\n".join(email_list)
        
        return {
            "response": response,
            "data": {"emails": [{"sender_email": r[0], "sender_name": r[1], "subject": r[2],
                                "email_date": r[3], "file_name": r[4]} for r in results]},
            "metadata": {"query_type": "email_by_subject", "count": len(results)}
        }
    
    def _handle_general_email_stats(self, db_connection) -> Dict[str, Any]:
        """Handle general email statistics queries."""
        cursor = db_connection.cursor()
        
        # Basic email counts
        cursor.execute("SELECT COUNT(*) FROM email_metadata")
        total_emails = cursor.fetchone()[0]
        
        if total_emails == 0:
            return {
                "response": "No emails found in the collection.",
                "data": {},
                "metadata": {"query_type": "email_general"}
            }
        
        # Top senders
        cursor.execute("""
            SELECT sender_email, sender_name, COUNT(*) as count
            FROM email_metadata 
            GROUP BY sender_email, sender_name
            ORDER BY count DESC 
            LIMIT 5
        """)
        top_senders = cursor.fetchall()
        
        # Recent email activity
        cursor.execute("""
            SELECT COUNT(*) 
            FROM email_metadata 
            WHERE email_date >= ?
        """, ((datetime.now() - timedelta(days=7)).timestamp(),))
        recent_emails = cursor.fetchone()[0]
        
        response_lines = [
            f"Email Collection Statistics:",
            f"• Total emails: {total_emails}",
            f"• Recent emails (7 days): {recent_emails}"
        ]
        
        if top_senders:
            response_lines.append("• Top senders:")
            for email, name, count in top_senders:
                display_name = name or email
                response_lines.append(f"  - {display_name}: {count} emails")
        
        response = "\n".join(response_lines)
        
        return {
            "response": response,
            "data": {
                "total_emails": total_emails,
                "recent_emails": recent_emails,
                "top_senders": [{"email": r[0], "name": r[1], "count": r[2]} for r in top_senders]
            },
            "metadata": {"query_type": "email_general"}
        }
    
    def _handle_enhanced_file_count(self, query_params: Dict[str, Any], db_connection) -> Dict[str, Any]:
        """Handle enhanced file counting with filters."""
        cursor = db_connection.cursor()
        
        conditions = []
        params = []
        
        # File type filter
        if "file_type" in query_params:
            conditions.append("f.file_type = ?")
            params.append(query_params["file_type"])
        
        # Time filter
        if "time_filter" in query_params:
            date_after = self._parse_time_filter(query_params["time_filter"])
            if date_after:
                conditions.append("f.modified_time >= ?")
                params.append(date_after.timestamp())
        
        # Size filter
        if "size_bytes" in query_params:
            operator = query_params.get("size_operator", "larger")
            if operator == "larger":
                conditions.append("f.file_size >= ?")
            else:
                conditions.append("f.file_size <= ?")
            params.append(query_params["size_bytes"])
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"SELECT COUNT(*) FROM files f WHERE {where_clause}"
        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        
        # Build description
        description_parts = []
        if "file_type" in query_params:
            description_parts.append(f"{query_params['file_type']} files")
        else:
            description_parts.append("files")
        
        if "time_filter" in query_params:
            description_parts.append(f"from {query_params['time_filter']}")
        
        if "size_bytes" in query_params:
            size_mb = query_params["size_bytes"] / (1024 * 1024)
            operator_text = "larger than" if query_params.get("size_operator") == "larger" else "smaller than"
            description_parts.append(f"{operator_text} {size_mb:.1f}MB")
        
        description = " ".join(description_parts)
        
        return {
            "response": f"There are {count} {description}.",
            "data": {"count": count, "filters": query_params},
            "metadata": {"query_type": "enhanced_file_count"}
        }
    
    def _handle_enhanced_file_list(self, query_params: Dict[str, Any], db_connection) -> Dict[str, Any]:
        """Handle enhanced file listing with filters."""
        cursor = db_connection.cursor()
        
        conditions = []
        params = []
        
        # File type filter
        if "file_type" in query_params:
            conditions.append("f.file_type = ?")
            params.append(query_params["file_type"])
        
        # Time filter
        if "time_filter" in query_params:
            date_after = self._parse_time_filter(query_params["time_filter"])
            if date_after:
                conditions.append("f.modified_time >= ?")
                params.append(date_after.timestamp())
        
        # Size filter
        if "size_bytes" in query_params:
            operator = query_params.get("size_operator", "larger")
            if operator == "larger":
                conditions.append("f.file_size >= ?")
            else:
                conditions.append("f.file_size <= ?")
            params.append(query_params["size_bytes"])
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT f.file_name, f.file_size, f.modified_time, f.file_type
            FROM files f 
            WHERE {where_clause}
            ORDER BY f.modified_time DESC
            LIMIT 20
        """
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        if not results:
            return {
                "response": "No files found matching the specified criteria.",
                "data": {"files": []},
                "metadata": {"query_type": "enhanced_file_list", "filters": query_params}
            }
        
        # Format response
        file_list = []
        for file_name, file_size, modified_time, file_type in results:
            size_mb = (file_size / (1024 * 1024)) if file_size else 0
            try:
                date_str = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d")
                file_list.append(f"• {file_name} ({size_mb:.1f}MB, {date_str})")
            except (ValueError, OSError):
                file_list.append(f"• {file_name} ({size_mb:.1f}MB)")
        
        # Build description
        description_parts = []
        if "file_type" in query_params:
            description_parts.append(f"{query_params['file_type']} files")
        else:
            description_parts.append("files")
        
        if "time_filter" in query_params:
            description_parts.append(f"from {query_params['time_filter']}")
        
        if "size_bytes" in query_params:
            size_mb = query_params["size_bytes"] / (1024 * 1024)
            operator_text = "larger than" if query_params.get("size_operator") == "larger" else "smaller than"
            description_parts.append(f"{operator_text} {size_mb:.1f}MB")
        
        description = " ".join(description_parts)
        response = f"Found {len(results)} {description}:\n" + "\n".join(file_list)
        
        return {
            "response": response,
            "data": {"files": [{"name": r[0], "size": r[1], "modified": r[2], "type": r[3]} for r in results]},
            "metadata": {"query_type": "enhanced_file_list", "count": len(results)}
        }
    
    def _handle_enhanced_recent_files(self, query_params: Dict[str, Any], db_connection) -> Dict[str, Any]:
        """Handle enhanced recent files queries with additional filters."""
        # Use enhanced file list with time filter
        if "time_filter" not in query_params:
            query_params["time_filter"] = "recent"
        
        return self._handle_enhanced_file_list(query_params, db_connection)
    
    def _handle_enhanced_file_types(self, query_params: Dict[str, Any], db_connection) -> Dict[str, Any]:
        """Handle enhanced file type queries with statistics."""
        cursor = db_connection.cursor()
        
        # Get comprehensive file type statistics
        cursor.execute("""
            SELECT f.file_type, COUNT(*) as count, 
                   COALESCE(AVG(f.file_size), 0) as avg_size,
                   COALESCE(SUM(f.file_size), 0) as total_size
            FROM files f
            GROUP BY f.file_type
            ORDER BY count DESC
        """)
        
        results = cursor.fetchall()
        
        if not results:
            return {
                "response": "No file type information available.",
                "data": {},
                "metadata": {"query_type": "enhanced_file_types"}
            }
        
        # Format response with enhanced statistics
        type_lines = []
        total_files = 0
        total_size = 0
        
        for file_type, count, avg_size, type_total_size in results:
            total_files += count
            total_size += type_total_size
            
            avg_mb = avg_size / (1024 * 1024) if avg_size else 0
            total_mb = type_total_size / (1024 * 1024) if type_total_size else 0
            
            type_lines.append(f"• {file_type}: {count} files (avg: {avg_mb:.1f}MB, total: {total_mb:.1f}MB)")
        
        total_mb = total_size / (1024 * 1024) if total_size else 0
        
        response_lines = [
            f"File Types in Collection:",
            f"Total: {total_files} files ({total_mb:.1f}MB)",
            ""
        ] + type_lines
        
        response = "\n".join(response_lines)
        
        return {
            "response": response,
            "data": {
                "types": [{"type": r[0], "count": r[1], "avg_size": r[2], "total_size": r[3]} for r in results],
                "total_files": total_files,
                "total_size": total_size
            },
            "metadata": {"query_type": "enhanced_file_types"}
        }
    
    def _handle_enhanced_general_stats(self, db_connection) -> Dict[str, Any]:
        """Handle enhanced general statistics with rich metadata."""
        cursor = db_connection.cursor()
        
        # Get comprehensive statistics
        cursor.execute("SELECT COUNT(*) FROM files")
        total_files = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM email_metadata")
        total_emails = cursor.fetchone()[0]
        
        cursor.execute("SELECT COALESCE(SUM(file_size), 0) FROM files")
        total_size = cursor.fetchone()[0]
        
        if total_files == 0:
            return {
                "response": "No files found in the collection.",
                "data": {},
                "metadata": {"query_type": "enhanced_general_stats"}
            }
        
        # Recent activity (last 7 days)
        recent_date = (datetime.now() - timedelta(days=7)).timestamp()
        cursor.execute("SELECT COUNT(*) FROM files WHERE modified_time >= ?", (recent_date,))
        recent_files = cursor.fetchone()[0]
        
        # File type breakdown
        cursor.execute("""
            SELECT file_type, COUNT(*) as count
            FROM files 
            GROUP BY file_type 
            ORDER BY count DESC 
            LIMIT 5
        """)
        top_types = cursor.fetchall()
        
        # Build response
        total_mb = total_size / (1024 * 1024) if total_size else 0
        
        response_lines = [
            f"Enhanced Document Collection Statistics:",
            f"• Total files: {total_files} ({total_mb:.1f}MB)",
            f"• Email files: {total_emails}",
            f"• Recent activity (7 days): {recent_files} files"
        ]
        
        if top_types:
            response_lines.append("• Top file types:")
            for file_type, count in top_types:
                response_lines.append(f"  - {file_type}: {count} files")
        
        response = "\n".join(response_lines)
        
        return {
            "response": response,
            "data": {
                "total_files": total_files,
                "total_emails": total_emails,
                "total_size": total_size,
                "recent_files": recent_files,
                "top_types": [{"type": r[0], "count": r[1]} for r in top_types]
            },
            "metadata": {"query_type": "enhanced_general_stats"}
        }