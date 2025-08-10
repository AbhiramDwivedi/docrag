"""DocQuest Agentic Architecture

This module implements the core agent framework for DocQuest, enabling intelligent
document analysis through a plugin-based architecture with advanced capabilities.
"""

from .agent import Agent
from .plugin import Plugin, PluginInfo
from .registry import PluginRegistry
from .factory import create_default_agent, create_agent_with_plugins, create_enhanced_agent, create_full_agent, create_minimal_agent

__all__ = [
    "Agent", 
    "Plugin", 
    "PluginInfo", 
    "PluginRegistry", 
    "create_default_agent", 
    "create_agent_with_plugins", 
    "create_enhanced_agent",
    "create_full_agent",
    "create_minimal_agent"
]