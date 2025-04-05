"""
Basic tests for SentinelDocs.

This module contains basic tests for the SentinelDocs package to verify functionality.
"""

import unittest
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentineldocs.utils.config import get_config, Config
from sentineldocs.utils.logging import setup_logging, get_logger


class TestConfig(unittest.TestCase):
    """Test the configuration module."""

    def test_config_loading(self):
        """Test that the configuration loader works correctly."""
        config = get_config()
        self.assertIsInstance(config, Config)
        
        # Test retrieving values with dot notation
        app_title = config.get("app.title")
        self.assertIsNotNone(app_title)
        
        # Test default values
        non_existent = config.get("non.existent", "default")
        self.assertEqual(non_existent, "default")


class TestLogging(unittest.TestCase):
    """Test the logging module."""

    def test_logging_setup(self):
        """Test that logging setup works."""
        logger = setup_logging(level="INFO", log_to_console=True)
        self.assertIsNotNone(logger)
        
        # Test getting a module logger
        module_logger = get_logger("test_module")
        self.assertIsNotNone(module_logger)


if __name__ == "__main__":
    unittest.main() 