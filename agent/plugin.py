"""Plugin interface and base classes for the DocQuest agent framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class PluginInfo:
    """Metadata about a plugin's capabilities and configuration."""
    
    name: str
    description: str
    version: str
    capabilities: List[str]
    parameters: Dict[str, Any]


class Plugin(ABC):
    """Base class for all DocQuest plugins.
    
    Plugins implement specific functionality that can be invoked by the agent
    to handle different types of queries and operations.
    """
    
    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plugin operation with given parameters.
        
        Args:
            params: Dictionary of parameters required for plugin execution
            
        Returns:
            Dictionary containing execution results and metadata
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If execution fails
        """
        pass
    
    @abstractmethod
    def get_info(self) -> PluginInfo:
        """Return plugin metadata and capabilities.
        
        Returns:
            PluginInfo object describing the plugin's capabilities
        """
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters before execution.
        
        Args:
            params: Dictionary of parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        pass
    
    def cleanup(self) -> None:
        """Cleanup plugin resources.
        
        Called when plugin is being unloaded or system is shutting down.
        Override this method if your plugin needs to perform cleanup.
        """
        pass