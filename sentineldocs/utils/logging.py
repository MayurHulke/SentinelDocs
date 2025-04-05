"""
Logging utilities for SentinelDocs.

This module provides functions for configuring and using logging throughout the application.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Default log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up application logging.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("sentineldocs")
    logger.setLevel(level)
    logger.handlers = []  # Remove any existing handlers
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    # If no specific log file but we have a log directory, create one there
    elif os.path.exists(LOG_DIR) or os.makedirs(LOG_DIR, exist_ok=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOG_DIR, f"sentineldocs_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized at level {logging.getLevelName(level)}")
    return logger

# Get a logger for a specific module
def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"sentineldocs.{module_name}") 