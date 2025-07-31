"""Basic metadata plugin for document collection statistics and queries."""

import logging
import sys
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Set
from collections import Counter

# Add parent directories to path to import from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.plugin import Plugin, PluginInfo
from config.config import settings


logger = logging.getLogger(__name__)


class MetadataPlugin(Plugin):
    """
    Plugin that provides basic metadata queries about the document collection.
    
    Handles queries about file counts, types, and basic statistics.
    """
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute metadata query on the document collection.
        
        Args:
            params: Must contain 'question' key with user's query
            
        Returns:
            Dictionary with:
            - 'result': The answer string
            - 'metadata': Information about the query and database
        """
        question = params['question'].lower()
        
        try:
            # Connect to the database
            conn = sqlite3.connect(settings.db_path)
            cur = conn.cursor()
            
            # Determine query type and execute appropriate query
            # Check for specific file types first before generic file queries
            if 'pdf' in question and ('how many' in question or 'count' in question):
                result = self._count_files_by_type(cur, '.pdf')
            elif 'excel' in question or 'xlsx' in question:
                if 'how many' in question or 'count' in question:
                    result = self._count_files_by_type(cur, '.xlsx')
                else:
                    result = self._list_files_by_type(cur, '.xlsx')
            elif 'word' in question or 'docx' in question:
                if 'how many' in question or 'count' in question:
                    result = self._count_files_by_type(cur, '.docx')
                else:
                    result = self._list_files_by_type(cur, '.docx')
            elif 'how many' in question and ('file' in question or 'document' in question):
                result = self._count_files(cur)
            elif 'file type' in question or 'file types' in question:
                result = self._list_file_types(cur)
            elif 'total' in question and 'file' in question:
                result = self._count_files(cur)
            elif 'list' in question and 'file' in question:
                result = self._list_files(cur, limit=20)  # Limit to avoid overwhelming output
            elif 'recent' in question or 'latest' in question:
                result = self._list_recent_files(cur)
            elif 'statistic' in question or 'stats' in question:
                result = self._get_collection_stats(cur)
            else:
                # Try to extract file type from question
                file_type = self._extract_file_type(question)
                if file_type:
                    if 'how many' in question or 'count' in question:
                        result = self._count_files_by_type(cur, file_type)
                    else:
                        result = self._list_files_by_type(cur, file_type)
                else:
                    result = "I can help with file counts, types, and basic statistics. Try asking 'how many files do we have?' or 'what file types are available?'"
            
            conn.close()
            
            return {
                'result': result,
                'metadata': {
                    'query_type': 'metadata',
                    'database_path': str(settings.db_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in metadata query: {e}")
            return {
                'result': f"Error querying metadata: {e}",
                'metadata': {'error': str(e)}
            }
    
    def _count_files(self, cursor) -> str:
        """Count total number of unique files."""
        cursor.execute("SELECT COUNT(DISTINCT file) FROM chunks WHERE current = 1")
        count = cursor.fetchone()[0]
        return f"There are {count} files in the collection."
    
    def _count_files_by_type(self, cursor, file_extension: str) -> str:
        """Count files by type/extension."""
        cursor.execute("SELECT COUNT(DISTINCT file) FROM chunks WHERE current = 1 AND file LIKE ?", (f'%{file_extension}',))
        count = cursor.fetchone()[0]
        return f"There are {count} {file_extension} files in the collection."
    
    def _list_file_types(self, cursor) -> str:
        """List all file types in the collection."""
        cursor.execute("SELECT DISTINCT file FROM chunks WHERE current = 1")
        files = cursor.fetchall()
        
        extensions = set()
        for (file_path,) in files:
            ext = Path(file_path).suffix.lower()
            if ext:
                extensions.add(ext)
        
        if not extensions:
            return "No file type information available."
        
        ext_list = sorted(extensions)
        return f"Available file types: {', '.join(ext_list)}"
    
    def _list_files(self, cursor, limit: int = 20) -> str:
        """List files in the collection with a limit."""
        cursor.execute("SELECT DISTINCT file FROM chunks WHERE current = 1 LIMIT ?", (limit,))
        files = cursor.fetchall()
        
        if not files:
            return "No files found in the collection."
        
        file_list = [Path(file[0]).name for file in files]
        result = f"Files in collection (showing up to {limit}):\n"
        result += "\n".join(f"- {filename}" for filename in file_list)
        
        if len(files) == limit:
            cursor.execute("SELECT COUNT(DISTINCT file) FROM chunks WHERE current = 1")
            total = cursor.fetchone()[0]
            if total > limit:
                result += f"\n\n... and {total - limit} more files."
        
        return result
    
    def _list_files_by_type(self, cursor, file_extension: str, limit: int = 10) -> str:
        """List files of a specific type."""
        cursor.execute("SELECT DISTINCT file FROM chunks WHERE current = 1 AND file LIKE ? LIMIT ?", 
                      (f'%{file_extension}', limit))
        files = cursor.fetchall()
        
        if not files:
            return f"No {file_extension} files found in the collection."
        
        file_list = [Path(file[0]).name for file in files]
        result = f"{file_extension} files (showing up to {limit}):\n"
        result += "\n".join(f"- {filename}" for filename in file_list)
        
        return result
    
    def _list_recent_files(self, cursor, limit: int = 10) -> str:
        """List recently modified files."""
        cursor.execute("""SELECT DISTINCT file, MAX(mtime) as latest_mtime 
                         FROM chunks WHERE current = 1 
                         GROUP BY file 
                         ORDER BY latest_mtime DESC LIMIT ?""", (limit,))
        files = cursor.fetchall()
        
        if not files:
            return "No files with modification time information found."
        
        result = f"Recent files (showing up to {limit}):\n"
        for file_path, mtime in files:
            filename = Path(file_path).name
            result += f"- {filename}\n"
        
        return result
    
    def _get_collection_stats(self, cursor) -> str:
        """Get comprehensive collection statistics."""
        # Total files
        cursor.execute("SELECT COUNT(DISTINCT file) FROM chunks WHERE current = 1")
        total_files = cursor.fetchone()[0]
        
        # Total chunks
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE current = 1")
        total_chunks = cursor.fetchone()[0]
        
        # File types
        cursor.execute("SELECT DISTINCT file FROM chunks WHERE current = 1")
        files = cursor.fetchall()
        
        extensions = Counter()
        for (file_path,) in files:
            ext = Path(file_path).suffix.lower()
            if ext:
                extensions[ext] += 1
        
        result = f"Document Collection Statistics:\n"
        result += f"- Total files: {total_files}\n"
        result += f"- Total text chunks: {total_chunks}\n"
        
        if extensions:
            result += f"- File types:\n"
            for ext, count in extensions.most_common():
                result += f"  • {ext}: {count} files\n"
        
        return result
    
    def _extract_file_type(self, question: str) -> str:
        """Extract file type/extension from question."""
        # Common file type mappings
        type_mappings = {
            'pdf': '.pdf',
            'excel': '.xlsx',
            'word': '.docx',
            'powerpoint': '.pptx',
            'text': '.txt',
            'email': '.msg'
        }
        
        for type_name, extension in type_mappings.items():
            if type_name in question:
                return extension
        
        return None
    
    def get_info(self) -> PluginInfo:
        """Return plugin metadata and capabilities."""
        return PluginInfo(
            name="metadata",
            description="Provides basic metadata queries about the document collection",
            version="1.0.0",
            capabilities=[
                "Count total files in collection",
                "List available file types",
                "Count files by type (PDF, Excel, Word, etc.)",
                "List recent files",
                "Provide collection statistics",
                "List files with optional filtering"
            ],
            parameters={
                "question": "The metadata question to answer"
            }
        )
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameters before execution.
        
        Args:
            params: Parameters to validate
            
        Returns:
            True if parameters are valid
        """
        return isinstance(params, dict) and 'question' in params and isinstance(params['question'], str) and len(params['question'].strip()) > 0
    
    def can_handle(self, query: str, query_type: str = None) -> bool:
        """
        Determine if this plugin can handle the given query.
        
        Args:
            query: The user's query
            query_type: Optional hint about query type
            
        Returns:
            True if plugin can handle the query
        """
        # Handle metadata-type queries
        if query_type == 'metadata':
            return True
        
        # Check for metadata keywords in query
        query_lower = query.lower()
        metadata_keywords = [
            'how many', 'count', 'total', 'list', 'files', 'file types',
            'pdf', 'excel', 'word', 'recent', 'latest', 'statistics', 'stats'
        ]
        
        return any(keyword in query_lower for keyword in metadata_keywords)