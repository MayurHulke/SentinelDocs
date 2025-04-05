#!/usr/bin/env python
"""
SentinelDocs Test Runner

This script discovers and runs all tests for the SentinelDocs package.
"""

import unittest
import sys
import os
from pathlib import Path


def run_tests():
    """Discover and run all tests."""
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Discover tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test_*.py")
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return exit code based on test result
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests()) 