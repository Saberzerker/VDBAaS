# src/utils/time_utils.py
"""
Time utility functions for the hybrid VDB system.

Author: Anonymous
Date: 2025-11-16
"""

import time
from datetime import datetime, timezone


def current_timestamp_ms() -> float:
    """
    Get current timestamp in milliseconds.
    
    Returns:
        Timestamp in milliseconds since epoch
    """
    return time.time() * 1000


def current_timestamp_s() -> float:
    """
    Get current timestamp in seconds.
    
    Returns:
        Timestamp in seconds since epoch
    """
    return time.time()


def timestamp_to_datetime(timestamp: float) -> datetime:
    """
    Convert Unix timestamp to datetime object.
    
    Args:
        timestamp: Unix timestamp (seconds)
    
    Returns:
        datetime object in UTC
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> float:
    """
    Convert datetime to Unix timestamp.
    
    Args:
        dt: datetime object
    
    Returns:
        Unix timestamp (seconds)
    """
    return dt.timestamp()


def format_timestamp(timestamp: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format Unix timestamp as string.
    
    Args:
        timestamp: Unix timestamp (seconds)
        fmt: Format string (default: YYYY-MM-DD HH:MM:SS)
    
    Returns:
        Formatted timestamp string
    """
    dt = timestamp_to_datetime(timestamp)
    return dt.strftime(fmt)


def elapsed_time_str(start_time: float) -> str:
    """
    Get human-readable elapsed time string.
    
    Args:
        start_time: Start timestamp (seconds)
    
    Returns:
        Elapsed time string (e.g., "2m 35s")
    """
    elapsed = time.time() - start_time
    
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    elif elapsed < 3600:
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        return f"{hours}h {minutes}m"