"""Tests for metadata commands plugin."""

import pytest
import sys
import os
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docquest.querying.agents.plugins.metadata_commands import MetadataCommandsPlugin


class TestMetadataCommandsPlugin:
    """Test the MetadataCommandsPlugin functionality."""
    
    def test_plugin_initialization(self):
        """Test that the plugin can be initialized."""
        plugin = MetadataCommandsPlugin()
        assert plugin is not None
        
        info = plugin.get_info()
        assert info.name == "metadata_commands"
        assert "find_files" in info.capabilities
    
    @patch('docquest.querying.agents.plugins.metadata_commands.sqlite3')
    def test_get_latest_files_with_mock_db(self, mock_sqlite):
        """Test get_latest_files with a mocked database."""
        # Setup mock
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ('test.msg', '/path/to/test.msg', 1024, 1672531200, 'MSG')  # Unix timestamp for 2023-01-01
        ]
        
        plugin = MetadataCommandsPlugin()
        
        # Mock the enhanced schema check to return True
        plugin._has_enhanced_schema = Mock(return_value=True)
        
        params = {
            "operation": "get_latest_files",
            "file_type": "MSG",
            "count": 10
        }
        
        result = plugin._get_latest_files(params, mock_conn)
        
        assert "response" in result
        assert "metadata" in result
        assert result["metadata"]["count"] == 1
    
    def test_validate_params_requires_operation(self):
        """Test that validate_params requires operation parameter."""
        plugin = MetadataCommandsPlugin()
        
        # Missing operation should fail
        assert plugin.validate_params({}) is False
        assert plugin.validate_params({"file_type": "PDF"}) is False
        
        # With operation should pass
        assert plugin.validate_params({"operation": "find_files"}) is True
