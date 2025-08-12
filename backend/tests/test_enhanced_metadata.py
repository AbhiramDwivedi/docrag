"""Tests for the enhanced metadata functionality."""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Temporarily skip this import until query_parser module is implemented
# from querying.agents.query_parser import QueryParser, create_enhanced_metadata_params
from querying.agents.plugins.metadata_commands import MetadataCommandsPlugin
from querying.agents.factory import create_enhanced_agent

class TestMetadataCommandsPlugin:
    """Test the metadata commands plugin."""
    
    def test_plugin_info(self):
        """Test plugin info retrieval."""
        plugin = MetadataCommandsPlugin()
        info = plugin.get_info()
        
        assert info.name == "metadata"
        assert "get_latest_files" in info.capabilities
        assert "find_files_by_content" in info.capabilities
    
    def test_validate_params_valid(self):
        """Test parameter validation with valid params."""
        plugin = MetadataCommandsPlugin()
        
        params = {"operation": "get_latest_files", "file_type": "PPTX", "count": 3}
        assert plugin.validate_params(params) is True
    
    def test_validate_params_invalid(self):
        """Test parameter validation with invalid params."""
        plugin = MetadataCommandsPlugin()
        
        # Missing operation
        params = {"file_type": "PPTX"}
        assert plugin.validate_params(params) is False
        
        # Empty operation
        params = {"operation": ""}
        assert plugin.validate_params(params) is False
    
    @patch('querying.agents.plugins.metadata_commands.sqlite3')
    @patch('querying.agents.plugins.metadata_commands.Path')
    def test_execute_no_database(self, mock_path, mock_sqlite):
        """Test execution when database doesn't exist."""
        # Mock Path.exists to return False
        mock_db_path = Mock()
        mock_db_path.exists.return_value = False
        mock_path.return_value = mock_db_path
        
        plugin = MetadataCommandsPlugin()
        result = plugin.execute({"operation": "get_latest_files"})
        
        assert result["metadata"]["error"] == "no_database"
        assert "No document database found" in result["response"]
    
    def test_unknown_operation(self):
        """Test handling of unknown operations."""
        plugin = MetadataCommandsPlugin()
        
        with patch.object(plugin, '_get_db_connection', return_value=Mock()):
            result = plugin.execute({"operation": "unknown_operation"})
            assert result["metadata"]["error"] == "unknown_operation"
            assert "Unknown operation" in result["response"]


class TestEnhancedAgent:
    """Test the enhanced agent with metadata commands."""
    
    def test_create_enhanced_agent(self):
        """Test creating the enhanced agent."""
        agent = create_enhanced_agent()
        
        # Should have metadata plugin
        assert agent.registry.get_plugin("metadata") is not None
        
        # In agentic mode, agent has its own capabilities, not plugin capabilities
        capabilities = agent.get_capabilities()
        # Check for agentic capabilities instead of plugin-specific ones
        assert "Multi-step reasoning and planning" in capabilities
        assert "Document discovery and search" in capabilities
    
    def test_query_processing_integration(self):
        """Test that queries are properly processed with enhanced metadata."""
        agent = create_enhanced_agent()
        
        # This should use the new metadata commands plugin through agentic processing
        # Mock the database to avoid needing real data
        with patch('querying.agents.plugins.metadata_commands.Path') as mock_path:
            mock_db_path = Mock()
            mock_db_path.exists.return_value = False
            mock_path.return_value = mock_db_path
            
            result = agent.process_query("find the three latest presentations")
            
            # In agentic mode, expect agentic response patterns
            assert ("No documents found matching your query" in result or 
                    "No document database found" in result or
                    "No files found matching" in result or
                    "Metadata retrieved" in result)