"""
Logging Configuration for EEW System

Provides structured logging for debugging and monitoring.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Color codes for terminal output
COLORS = {
    'DEBUG': '\033[36m',     # Cyan
    'INFO': '\033[32m',      # Green
    'WARNING': '\033[33m',   # Yellow
    'ERROR': '\033[31m',     # Red
    'CRITICAL': '\033[35m',  # Magenta
    'RESET': '\033[0m'
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output"""
    
    def format(self, record):
        color = COLORS.get(record.levelname, COLORS['RESET'])
        reset = COLORS['RESET']
        
        # Add color to level name
        record.levelname = f"{color}{record.levelname}{reset}"
        
        return super().format(record)


def setup_logger(
    name: str = "eew",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with optional file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Log format
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter(log_format, date_format))
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "eew") -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name (use dot notation for hierarchy, e.g., "eew.consensus")
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Component-specific loggers
def get_model_logger():
    return get_logger("eew.model")

def get_consensus_logger():
    return get_logger("eew.consensus")

def get_device_logger():
    return get_logger("eew.device")

def get_network_logger():
    return get_logger("eew.network")

def get_simulation_logger():
    return get_logger("eew.simulation")

def get_alert_logger():
    return get_logger("eew.alert")
