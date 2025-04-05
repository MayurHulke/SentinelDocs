#!/usr/bin/env python
"""
SentinelDocs - Main entry point

This is the main entry point for running the SentinelDocs application.
Run this file to start the application.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the main app function
from sentineldocs.app import main

if __name__ == "__main__":
    # Run the application
    main()