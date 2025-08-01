"""DocQuest agent plugins.

This package contains the core plugins for the DocQuest agent framework.
"""

from .semantic_search import SemanticSearchPlugin
from .metadata_commands import MetadataCommandsPlugin
# Legacy import for backward compatibility
from .metadata_legacy import MetadataPlugin as MetadataLegacyPlugin

__all__ = ["SemanticSearchPlugin", "MetadataCommandsPlugin", "MetadataLegacyPlugin"]