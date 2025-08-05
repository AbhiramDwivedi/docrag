"""Tests for the enhanced metadata functionality."""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docquest.querying.agents.query_parser import QueryParser, create_enhanced_metadata_params
from docquest.querying.agents.plugins.metadata_commands import MetadataCommandsPlugin
from docquest.querying.agents.factory import create_enhanced_agent


class TestQueryParser:
    """Test the natural language query parser."""
    
    def test_file_type_extraction(self):
        """Test file type extraction from queries."""
        parser = QueryParser()
        
        # Test presentation queries
        assert parser._extract_file_type("show me the latest presentations") == "PPTX"
        assert parser._extract_file_type("find three newest PowerPoint files") == "PPTX"
        assert parser._extract_file_type("latest slides") == "PPTX"
        
        # Test email queries
        assert parser._extract_file_type("find all emails related to google") == "MSG"
        assert parser._extract_file_type("show me recent mail") == "MSG"
        
        # Test document queries
        assert parser._extract_file_type("latest documents") == "DOCX"
        assert parser._extract_file_type("find Word files") == "DOCX"
    
    def test_count_extraction(self):
        """Test count/number extraction from queries."""
        parser = QueryParser()
        
        # Test numeric counts
        assert parser._extract_count("show me 3 latest files") == 3
        assert parser._extract_count("find 10 documents") == 10
        
        # Test word counts
        assert parser._extract_count("show me three latest presentations") == 3
        assert parser._extract_count("find five emails") == 5
        assert parser._extract_count("get the first document") == 1
        
        # Test roman numerals
        assert parser._extract_count("show me iii files") == 3
    
    def test_time_filter_extraction(self):
        """Test time filter extraction from queries."""
        parser = QueryParser()
        
        assert parser._extract_time_filter("latest files from last week") == "last_week"
        assert parser._extract_time_filter("recent documents") == "recent"
        assert parser._extract_time_filter("files modified yesterday") == "yesterday"
        assert parser._extract_time_filter("newest presentations") == "recent"
    
    def test_keyword_extraction(self):
        """Test keyword extraction for content search."""
        parser = QueryParser()
        
        keywords = parser._extract_keywords("find all emails related to google")
        assert "google" in keywords
        
        keywords = parser._extract_keywords("files about marketing strategy")
        assert "marketing strategy" in keywords
        
        keywords = parser._extract_keywords("documents containing budget information")
        assert "budget information" in keywords
    
    def test_operation_determination(self):
        """Test operation type determination."""
        parser = QueryParser()
        
        # Latest files operations
        assert parser._determine_operation("find the three latest presentations") == "get_latest_files"
        assert parser._determine_operation("show me recent documents") == "get_latest_files"
        
        # Content search operations
        assert parser._determine_operation("find all emails related to google") == "find_files_by_content"
        assert parser._determine_operation("documents about budget") == "find_files_by_content"
        
        # Count operations
        assert parser._determine_operation("how many files do we have") == "get_file_count"
        assert parser._determine_operation("count of presentations") == "get_file_count"
    
    def test_full_query_parsing(self):
        """Test complete query parsing."""
        parser = QueryParser()
        
        # Test case 1: "find the three latest modified presentations"
        operation, params = parser.parse_query("find the three latest modified presentations")
        assert operation == "get_latest_files"
        assert params["file_type"] == "PPTX"
        assert params["count"] == 3
        assert params["time_filter"] == "recent"  # "latest" maps to "recent"
        
        # Test case 2: "find all emails related to google"
        operation, params = parser.parse_query("find all emails related to google")
        assert operation == "find_files_by_content"
        assert params["file_type"] == "MSG"
        assert "google" in params["keywords"]


class TestMetadataCommandsPlugin:
    """Test the metadata commands plugin."""
    
    def test_plugin_info(self):
        """Test plugin info retrieval."""
        plugin = MetadataCommandsPlugin()
        info = plugin.get_info()
        
        assert info.name == "metadata_commands"
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
    
    @patch('agent.plugins.metadata_commands.sqlite3')
    @patch('agent.plugins.metadata_commands.Path')
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
        
        # Should have metadata_commands plugin
        assert agent.registry.get_plugin("metadata_commands") is not None
        
        capabilities = agent.get_capabilities()
        assert "get_latest_files" in capabilities
        assert "find_files_by_content" in capabilities
    
    def test_query_processing_integration(self):
        """Test that queries are properly processed with enhanced metadata."""
        agent = create_enhanced_agent()
        
        # This should use the new metadata commands plugin
        # Mock the database to avoid needing real data
        with patch('agent.plugins.metadata_commands.Path') as mock_path:
            mock_db_path = Mock()
            mock_db_path.exists.return_value = False
            mock_path.return_value = mock_db_path
            
            result = agent.process_query("find the three latest presentations")
            
            # Should get response from metadata commands plugin
            assert "No document database found" in result


class TestCreateEnhancedMetadataParams:
    """Test the enhanced metadata parameter creation function."""
    
    def test_presentation_query(self):
        """Test parsing presentation-related queries."""
        params = create_enhanced_metadata_params("find the three latest modified presentations")
        
        assert params["operation"] == "get_latest_files"
        assert params["file_type"] == "PPTX"
        assert params["count"] == 3
        assert params["time_filter"] == "recent"
    
    def test_email_query(self):
        """Test parsing email-related queries."""
        params = create_enhanced_metadata_params("find all emails related to google")
        
        assert params["operation"] == "find_files_by_content"
        assert params["file_type"] == "MSG"
        assert "google" in params["keywords"]
    
    def test_count_query(self):
        """Test parsing count queries."""
        params = create_enhanced_metadata_params("how many PDF files do we have")
        
        assert params["operation"] == "get_file_count"
        assert params["file_type"] == "PDF"