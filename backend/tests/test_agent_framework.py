"""Tests for the DocQuest agent framework."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from querying.agents.agent import Agent
from querying.agents.registry import PluginRegistry
from querying.agents.factory import create_default_agent
from querying.agents.plugin import Plugin, PluginInfo
from querying.agents.plugins.semantic_search import SemanticSearchPlugin
from querying.agents.plugins.metadata_commands import MetadataCommandsPlugin


class TestPluginRegistry:
    """Test the plugin registry functionality."""
    
    def test_register_plugin(self):
        """Test plugin registration."""
        registry = PluginRegistry()
        
        # Create a mock plugin
        mock_plugin = Mock(spec=Plugin)
        mock_plugin.get_info.return_value = PluginInfo(
            name="test_plugin",
            description="Test plugin",
            version="1.0.0",
            capabilities=["test"],
            parameters={}
        )
        
        registry.register(mock_plugin)
        assert registry.get_plugin_count() == 1
        assert registry.get_plugin("test_plugin") == mock_plugin
    
    def test_duplicate_registration_fails(self):
        """Test that duplicate plugin registration fails."""
        registry = PluginRegistry()
        
        # Create mock plugins with same name
        mock_plugin1 = Mock(spec=Plugin)
        mock_plugin2 = Mock(spec=Plugin)
        
        info = PluginInfo(
            name="test_plugin",
            description="Test plugin",
            version="1.0.0",
            capabilities=["test"],
            parameters={}
        )
        
        mock_plugin1.get_info.return_value = info
        mock_plugin2.get_info.return_value = info
        
        registry.register(mock_plugin1)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register(mock_plugin2)
    
    def test_unregister_plugin(self):
        """Test plugin unregistration."""
        registry = PluginRegistry()
        
        mock_plugin = Mock(spec=Plugin)
        mock_plugin.get_info.return_value = PluginInfo(
            name="test_plugin",
            description="Test plugin",
            version="1.0.0",
            capabilities=["test"],
            parameters={}
        )
        
        registry.register(mock_plugin)
        assert registry.get_plugin_count() == 1
        
        success = registry.unregister("test_plugin")
        assert success is True
        assert registry.get_plugin_count() == 0
        assert registry.get_plugin("test_plugin") is None
        
        # Verify cleanup was called
        mock_plugin.cleanup.assert_called_once()
    
    def test_find_plugins_for_capability(self):
        """Test finding plugins by capability."""
        registry = PluginRegistry()
        
        # Create plugins with different capabilities
        plugin1 = Mock(spec=Plugin)
        plugin1.get_info.return_value = PluginInfo(
            name="plugin1",
            description="Plugin 1",
            version="1.0.0",
            capabilities=["search", "query"],
            parameters={}
        )
        
        plugin2 = Mock(spec=Plugin)
        plugin2.get_info.return_value = PluginInfo(
            name="plugin2",
            description="Plugin 2",
            version="1.0.0",
            capabilities=["metadata", "stats"],
            parameters={}
        )
        
        registry.register(plugin1)
        registry.register(plugin2)
        
        search_plugins = registry.find_plugins_for_capability("search")
        assert search_plugins == ["plugin1"]
        
        metadata_plugins = registry.find_plugins_for_capability("metadata")
        assert metadata_plugins == ["plugin2"]
        
        missing_plugins = registry.find_plugins_for_capability("nonexistent")
        assert missing_plugins == []


class TestAgent:
    """Test the core agent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        registry = PluginRegistry()
        agent = Agent(registry)
        
        assert agent.registry == registry
        assert agent.get_capabilities() == []
    
    def test_agent_with_plugins(self):
        """Test agent with registered plugins."""
        registry = PluginRegistry()
        
        # Mock plugin
        mock_plugin = Mock(spec=Plugin)
        mock_plugin.get_info.return_value = PluginInfo(
            name="test_plugin",
            description="Test plugin",
            version="1.0.0",
            capabilities=["test_capability"],
            parameters={}
        )
        
        registry.register(mock_plugin)
        agent = Agent(registry)
        
        capabilities = agent.get_capabilities()
        assert "test_capability" in capabilities
    
    @patch('backend.src.querying.agents.plugins.semantic_search.settings')
    def test_query_processing_no_api_key(self, mock_settings):
        """Test query processing without API key."""
        mock_settings.openai_api_key = "your-openai-api-key-here"
        
        agent = create_default_agent()
        result = agent.process_query("test question")
        
        assert "OpenAI API key not configured" in result
    
    def test_empty_query(self):
        """Test handling of empty queries."""
        agent = create_default_agent()
        result = agent.process_query("")
        
        assert "No relevant information found" in result
    
    def test_explain_reasoning_no_query(self):
        """Test explain_reasoning with no prior query."""
        agent = create_default_agent()
        explanation = agent.explain_reasoning()
        
        assert explanation is None
    
    def test_explain_reasoning_with_query(self):
        """Test explain_reasoning after processing a query."""
        agent = create_default_agent()
        agent.process_query("test question")
        
        explanation = agent.explain_reasoning()
        assert explanation is not None
        assert "Query: test question" in explanation
        assert "Execution time:" in explanation


class TestSemanticSearchPlugin:
    """Test the semantic search plugin."""
    
    def test_plugin_info(self):
        """Test plugin info retrieval."""
        plugin = SemanticSearchPlugin()
        info = plugin.get_info()
        
        assert info.name == "semantic_search"
        assert "semantic_search" in info.capabilities
        assert "document_query" in info.capabilities
    
    def test_validate_params_valid(self):
        """Test parameter validation with valid params."""
        plugin = SemanticSearchPlugin()
        
        params = {"question": "test question", "k": 5}
        assert plugin.validate_params(params) is True
    
    def test_validate_params_invalid(self):
        """Test parameter validation with invalid params."""
        plugin = SemanticSearchPlugin()
        
        # Missing question
        params = {"k": 5}
        assert plugin.validate_params(params) is False
        
        # Empty question
        params = {"question": "", "k": 5}
        assert plugin.validate_params(params) is False
        
        # Invalid k
        params = {"question": "test", "k": 0}
        assert plugin.validate_params(params) is False
    
    @patch('backend.src.querying.agents.plugins.semantic_search.settings')
    def test_execute_no_api_key(self, mock_settings):
        """Test execution without API key."""
        mock_settings.openai_api_key = "your-openai-api-key-here"
        
        plugin = SemanticSearchPlugin()
        result = plugin.execute({"question": "test question"})
        
        assert result["metadata"]["error"] == "missing_api_key"
        assert "OpenAI API key not configured" in result["response"]


class TestMetadataCommandsPlugin:
    """Test the metadata plugin."""
    
    def test_plugin_info(self):
        """Test plugin info retrieval."""
        plugin = MetadataCommandsPlugin()
        info = plugin.get_info()
        
        assert info.name == "metadata"
        assert "find_files" in info.capabilities
        assert "get_file_stats" in info.capabilities
    
    def test_validate_params_valid(self):
        """Test parameter validation with valid params."""
        plugin = MetadataCommandsPlugin()
        
        params = {"operation": "find_files"}
        assert plugin.validate_params(params) is True
    
    def test_validate_params_invalid(self):
        """Test parameter validation with invalid params."""
        plugin = MetadataCommandsPlugin()
        
        # Missing question
        params = {}
        assert plugin.validate_params(params) is False
        
        # Empty question
        params = {"question": ""}
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
        result = plugin.execute({"operation": "find_files"})
        
        assert result["metadata"]["error"] == "no_database"
        assert "No document database found" in result["response"]


class TestAgentIntegration:
    """Test agent integration scenarios."""
    
    def test_create_default_agent(self):
        """Test creating the default agent."""
        agent = create_default_agent()
        
        assert agent.registry.get_plugin_count() == 2  # semantic_search + metadata
        assert agent.registry.get_plugin("semantic_search") is not None
        assert agent.registry.get_plugin("metadata") is not None
        
        capabilities = agent.get_capabilities()
        assert "semantic_search" in capabilities
        assert "find_files" in capabilities
    
    def test_query_classification_metadata(self):
        """Test that metadata queries are properly classified."""
        agent = create_default_agent()
        
        # This should trigger metadata plugin (no API key needed)
        result = agent.process_query("how many files do we have?")
        
        # Should get a response from metadata plugin, not an API key error
        assert "OpenAI API key" not in result
        assert ("files in the collection" in result or 
                "No document database found" in result or
                "No files found matching" in result)
    
    @patch('backend.src.querying.agents.plugins.semantic_search.settings')
    def test_query_classification_semantic(self, mock_settings):
        """Test that content queries are properly classified."""
        mock_settings.openai_api_key = "your-openai-api-key-here"
        
        agent = create_default_agent()
        
        # This should trigger semantic search plugin
        result = agent.process_query("what is the compliance policy?")
        
        # Should get API key error since we don't have a real key
        assert "OpenAI API key not configured" in result
    
    def test_explain_reasoning_integration(self):
        """Test reasoning explanation in integrated scenario."""
        agent = create_default_agent()
        
        # Process a metadata query
        agent.process_query("how many files do we have?")
        
        explanation = agent.explain_reasoning()
        assert "Query: how many files do we have?" in explanation
        assert "metadata" in explanation.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])