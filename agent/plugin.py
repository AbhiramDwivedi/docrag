"""Plugin base classes and interfaces for the DocQuest agent framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class PluginInfo:
    """Metadata about a plugin's capabilities."""
    name: str
    description: str
    version: str
    capabilities: List[str]
    parameters: Dict[str, str]  # parameter_name -> description


class Plugin(ABC):
    """Base class for all DocQuest plugins."""
    
    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute plugin operation with given parameters.
        
        Args:
            params: Plugin-specific parameters
            
        Returns:
            Dictionary containing:
            - 'result': The main result (str for answers, dict for structured data)
            - 'metadata': Optional metadata about the operation
            - 'confidence': Optional confidence score (0-1)
            - 'sources': Optional list of sources used
        """
        pass
    
    @abstractmethod
    def get_info(self) -> PluginInfo:
        """Return plugin metadata and capabilities."""
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameters before execution.
        
        Args:
            params: Parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        pass
    
    def can_handle(self, query: str, query_type: Optional[str] = None) -> bool:
        """
        Determine if this plugin can handle the given query.
        
        Args:
            query: The user's query
            query_type: Optional hint about query type
            
        Returns:
            True if plugin can handle the query
        """
        return True  # Default implementation - override for smarter routing