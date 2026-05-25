# src/utils/logger.py
"""
Logging configuration for Hybrid VDB system.

Provides consistent logging across all modules.

Author: Saberzerker
Date: 2025-11-16
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

from src.config import LOG_LEVEL, LOG_FILE


def setup_logger(name: str = None) -> logging.Logger:
    """
    Configure and return logger instance.
    
    Args:
        name: Logger name (typically __name__ of calling module)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name or "hybrid_vdb")
    
    # Only configure if not already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(LOG_LEVEL)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(levelname)s - %(message)s'
    )
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (all levels)
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance for a module.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)