"""
Configuration management module for SentinelDocs.

This module provides functions for loading and accessing application configuration.
"""

import os
import yaml
from typing import Dict, Any, Optional

# Default configuration file path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                   "config", "default.yaml")

class Config:
    """Configuration manager for SentinelDocs."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a custom configuration file
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict containing configuration values
        """
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading configuration from {self.config_path}: {str(e)}")
            # Return empty config as fallback
            return {}
            
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-separated path.
        
        Args:
            path: Dot-separated path to configuration value (e.g., "app.title")
            default: Default value to return if path not found
            
        Returns:
            Configuration value at path or default if not found
        """
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value

# Singleton instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the configuration instance.
    
    Args:
        config_path: Optional path to a custom configuration file
        
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance 