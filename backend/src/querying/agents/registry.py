"""Plugin registry for managing and discovering available plugins."""

import logging
from typing import Dict, List, Optional
from .plugin import Plugin, PluginInfo

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing plugin lifecycle and discovery.
    
    The registry maintains a catalog of available plugins and provides
    methods for registration, discovery, and plugin metadata access.
    """
    
    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_info: Dict[str, PluginInfo] = {}
    
    def register(self, plugin: Plugin) -> None:
        """Register a plugin with the agent framework.
        
        Args:
            plugin: Plugin instance to register
            
        Raises:
            ValueError: If plugin name conflicts with existing plugin
        """
        info = plugin.get_info()
        
        if info.name in self._plugins:
            raise ValueError(f"Plugin '{info.name}' is already registered")
        
        self._plugins[info.name] = plugin
        self._plugin_info[info.name] = info
        
        logger.info(f"Registered plugin: {info.name} v{info.version}")
    
    def register_plugin(self, name: str, plugin: Plugin) -> None:
        """Register a plugin with the agent framework (backward compatibility).
        
        Args:
            name: Plugin name (for backward compatibility, actual name comes from plugin.get_info())
            plugin: Plugin instance to register
            
        Raises:
            ValueError: If plugin name conflicts with existing plugin
        """
        # Use the new register method
        self.register(plugin)
    
    def unregister(self, plugin_name: str) -> bool:
        """Unregister a plugin from the framework.
        
        Args:
            plugin_name: Name of the plugin to unregister
            
        Returns:
            True if plugin was unregistered, False if not found
        """
        if plugin_name not in self._plugins:
            return False
        
        # Cleanup plugin resources
        try:
            self._plugins[plugin_name].cleanup()
        except Exception as e:
            logger.warning(f"Error during plugin cleanup for {plugin_name}: {e}")
        
        del self._plugins[plugin_name]
        del self._plugin_info[plugin_name]
        
        logger.info(f"Unregistered plugin: {plugin_name}")
        return True
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Get a registered plugin by name.
        
        Args:
            plugin_name: Name of the plugin to retrieve
            
        Returns:
            Plugin instance if found, None otherwise
        """
        return self._plugins.get(plugin_name)
    
    def discover_plugins(self) -> List[Plugin]:
        """Discover and return all available plugins.
        
        Returns:
            List of all registered plugin instances
        """
        return list(self._plugins.values())
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get metadata for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            PluginInfo if plugin exists, None otherwise
        """
        return self._plugin_info.get(plugin_name)
    
    def list_capabilities(self) -> Dict[str, List[str]]:
        """List all capabilities provided by registered plugins.
        
        Returns:
            Dictionary mapping plugin names to their capabilities
        """
        return {
            name: info.capabilities 
            for name, info in self._plugin_info.items()
        }
    
    def find_plugins_for_capability(self, capability: str) -> List[str]:
        """Find all plugins that provide a specific capability.
        
        Args:
            capability: The capability to search for
            
        Returns:
            List of plugin names that provide the capability
        """
        return [
            name for name, info in self._plugin_info.items()
            if capability in info.capabilities
        ]
    
    def get_plugin_count(self) -> int:
        """Get the number of registered plugins.
        
        Returns:
            Number of registered plugins
        """
        return len(self._plugins)