"""
DocQuest Plugins

Built-in plugins for the DocQuest agent framework.
"""

from .semantic_search import SemanticSearchPlugin
from .metadata import MetadataPlugin

__all__ = ['SemanticSearchPlugin', 'MetadataPlugin']