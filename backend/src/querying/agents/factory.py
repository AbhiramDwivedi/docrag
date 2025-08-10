"""Agent factory for creating and configuring DocQuest agents.

This module provides a convenient way to create and configure agents
with the standard set of plugins for DocQuest functionality, including
Phase III advanced intelligence capabilities.
"""

import logging
from typing import Optional, List
from .agent import Agent
from .registry import PluginRegistry
from .plugins.semantic_search import SemanticSearchPlugin
from .plugins.metadata_commands import MetadataCommandsPlugin
from .plugins.document_relationships import DocumentRelationshipPlugin
from .plugins.comprehensive_reporting import ComprehensiveReportingPlugin
from .plugins.knowledge_graph import KnowledgeGraphPlugin

logger = logging.getLogger(__name__)


def create_full_agent() -> Agent:
    """Create an agent with all available capabilities.
    
    This agent includes all plugins: semantic search, metadata commands,
    document relationships, comprehensive reporting, and knowledge graph.
    
    Returns:
        Configured Agent instance with full capabilities
    """
    registry = PluginRegistry()
    
    # Register all Phase III plugins
    plugins = [
        ("semantic_search", SemanticSearchPlugin),
        ("metadata_commands", MetadataCommandsPlugin),
        ("document_relationships", DocumentRelationshipPlugin),
        ("comprehensive_reporting", ComprehensiveReportingPlugin),
        ("knowledge_graph", KnowledgeGraphPlugin)
    ]
    
    for plugin_name, plugin_class in plugins:
        try:
            plugin_instance = plugin_class()
            registry.register(plugin_instance)
            logger.info(f"Registered plugin: {plugin_name}")
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_name}: {e}")
    
    agent = Agent(registry)
    
    logger.info(f"Created full-featured agent with {registry.get_plugin_count()} plugins")
    logger.info(f"Available capabilities: {agent.get_capabilities()}")
    
    return agent


def create_enhanced_agent() -> Agent:
    """Create an agent with enhanced metadata functionality.
    
    This agent uses structured metadata commands for advanced document
    discovery and analysis.
    
    Returns:
        Configured Agent instance with semantic search and enhanced metadata plugins
    """
    registry = PluginRegistry()
    
    # Register core plugins
    try:
        semantic_plugin = SemanticSearchPlugin()
        registry.register(semantic_plugin)
        logger.info("Registered semantic search plugin")
    except Exception as e:
        logger.error(f"Failed to register semantic search plugin: {e}")
    
    try:
        metadata_plugin = MetadataCommandsPlugin()
        registry.register(metadata_plugin)
        logger.info("Registered metadata commands plugin")
    except Exception as e:
        logger.error(f"Failed to register metadata commands plugin: {e}")
    
    agent = Agent(registry)
    
    logger.info(f"Created enhanced agent with {registry.get_plugin_count()} plugins")
    logger.info(f"Available capabilities: {agent.get_capabilities()}")
    
    return agent


def create_knowledge_graph_agent() -> Agent:
    """Create an agent with knowledge graph capabilities.
    
    This agent includes semantic search, metadata commands, and knowledge graph plugins
    for enhanced entity exploration and relationship analysis.
    
    Returns:
        Configured Agent instance with knowledge graph capabilities
    """
    registry = PluginRegistry()
    
    # Register core plugins
    plugins = [
        ("semantic_search", SemanticSearchPlugin),
        ("metadata_commands", MetadataCommandsPlugin),
        ("knowledge_graph", KnowledgeGraphPlugin)
    ]
    
    for plugin_name, plugin_class in plugins:
        try:
            plugin_instance = plugin_class()
            registry.register(plugin_instance)
            logger.info(f"Registered plugin: {plugin_name}")
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_name}: {e}")
    
    agent = Agent(registry)
    
    logger.info(f"Created knowledge graph agent with {registry.get_plugin_count()} plugins")
    logger.info(f"Available capabilities: {agent.get_capabilities()}")
    
    return agent


def create_default_agent() -> Agent:
    """Create an agent with the default set of plugins.
    
    Returns:
        Configured Agent instance with semantic search and metadata plugins
    """
    return create_enhanced_agent()


def create_agent_with_plugins(plugin_names: Optional[list] = None) -> Agent:
    """Create an agent with specific plugins.
    
    Args:
        plugin_names: List of plugin names to include. If None, includes all available plugins.
        
    Returns:
        Configured Agent instance
    """
    registry = PluginRegistry()
    
    # Available plugins
    available_plugins = {
        "semantic_search": SemanticSearchPlugin,
        "metadata_commands": MetadataCommandsPlugin,
        "document_relationships": DocumentRelationshipPlugin,
        "comprehensive_reporting": ComprehensiveReportingPlugin,
        "knowledge_graph": KnowledgeGraphPlugin
    }
    
    # Use all plugins if none specified
    if plugin_names is None:
        plugin_names = list(available_plugins.keys())
    
    # Register requested plugins
    for plugin_name in plugin_names:
        if plugin_name in available_plugins:
            try:
                plugin_class = available_plugins[plugin_name]
                plugin_instance = plugin_class()
                registry.register(plugin_instance)
                logger.info(f"Registered plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Failed to register plugin {plugin_name}: {e}")
        else:
            logger.warning(f"Unknown plugin requested: {plugin_name}")
    
    agent = Agent(registry)
    logger.info(f"Created agent with {registry.get_plugin_count()} plugins")
    
    return agent


def create_minimal_agent() -> Agent:
    """Create a minimal agent with only semantic search capability.
    
    Returns:
        Minimal agent instance with only semantic search
    """
    registry = PluginRegistry()
    
    try:
        semantic_plugin = SemanticSearchPlugin()
        registry.register(semantic_plugin)
        logger.info("Registered semantic search plugin")
    except Exception as e:
        logger.error(f"Failed to register semantic search plugin: {e}")
    
    agent = Agent(registry)
    logger.info(f"Created minimal agent with {registry.get_plugin_count()} plugins")
    
    return agent