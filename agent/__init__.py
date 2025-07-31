"""
DocQuest Agent Framework

Provides plugin-based architecture for intelligent document analysis and querying.
"""

from .agent import Agent
from .plugin import Plugin, PluginInfo
from .registry import PluginRegistry

__all__ = ['Agent', 'Plugin', 'PluginInfo', 'PluginRegistry']