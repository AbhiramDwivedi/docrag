"""Plugin registry for plugin discovery and management."""

import logging
from typing import Dict, List, Type
from .plugin import Plugin, PluginInfo


logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing DocQuest plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}
    
    def register(self, plugin: Plugin) -> None:
        """
        Register a plugin with the agent framework.
        
        Args:
            plugin: Plugin instance to register
        """
        info = plugin.get_info()
        if info.name in self._plugins:
            logger.warning(f"Plugin '{info.name}' is already registered. Overwriting.")
        
        self._plugins[info.name] = plugin
        logger.info(f"Registered plugin: {info.name} v{info.version}")
    
    def register_class(self, plugin_class: Type[Plugin]) -> None:
        """
        Register a plugin class for lazy instantiation.
        
        Args:
            plugin_class: Plugin class to register
        """
        # Create temporary instance to get info
        temp_instance = plugin_class()
        info = temp_instance.get_info()
        
        self._plugin_classes[info.name] = plugin_class
        logger.info(f"Registered plugin class: {info.name}")
    
    def get_plugin(self, name: str) -> Plugin:
        """
        Get a plugin by name, instantiating if necessary.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance
            
        Raises:
            KeyError: If plugin is not found
        """
        if name in self._plugins:
            return self._plugins[name]
        
        if name in self._plugin_classes:
            plugin = self._plugin_classes[name]()
            self._plugins[name] = plugin
            return plugin
        
        raise KeyError(f"Plugin '{name}' not found")
    
    def discover_plugins(self) -> List[Plugin]:
        """
        Discover and return all available plugins.
        
        Returns:
            List of all registered plugin instances
        """
        plugins = []
        
        # Get already instantiated plugins
        plugins.extend(self._plugins.values())
        
        # Instantiate plugin classes that aren't already instantiated
        for name, plugin_class in self._plugin_classes.items():
            if name not in self._plugins:
                plugin = plugin_class()
                self._plugins[name] = plugin
                plugins.append(plugin)
        
        return plugins
    
    def get_plugins_for_query(self, query: str, query_type: str = None) -> List[Plugin]:
        """
        Get plugins that can handle the given query.
        
        Args:
            query: User's query
            query_type: Optional query type hint
            
        Returns:
            List of plugins that can handle the query
        """
        capable_plugins = []
        
        for plugin in self.discover_plugins():
            if plugin.can_handle(query, query_type):
                capable_plugins.append(plugin)
        
        return capable_plugins
    
    def list_capabilities(self) -> Dict[str, PluginInfo]:
        """
        Get information about all registered plugins.
        
        Returns:
            Dictionary mapping plugin names to their info
        """
        capabilities = {}
        
        for plugin in self.discover_plugins():
            info = plugin.get_info()
            capabilities[info.name] = info
        
        return capabilities
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name to unregister
            
        Returns:
            True if plugin was found and removed, False otherwise
        """
        removed = False
        
        if name in self._plugins:
            del self._plugins[name]
            removed = True
        
        if name in self._plugin_classes:
            del self._plugin_classes[name]
            removed = True
        
        if removed:
            logger.info(f"Unregistered plugin: {name}")
        
        return removed