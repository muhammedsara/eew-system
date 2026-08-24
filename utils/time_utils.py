"""
Time Utilities for EEW System

Handles temporal operations including adaptive threshold calculation
based on time-of-day (Paper Equation 2).
"""

from datetime import datetime, timezone
from typing import Tuple
import time


def get_adaptive_threshold_hour(hour: int) -> float:
    """
    Get adaptive detection threshold based on hour of day.
    
    From Paper Equation 2:
        θ(h) = 0.75 if 2 ≤ h < 6   (night - low ambient noise)
        θ(h) = 0.90 if 8 ≤ h < 18  (day - high traffic)  
        θ(h) = 0.85 otherwise
    
    Args:
        hour: Hour of day (0-23)
        
    Returns:
        Threshold value [0, 1]
    """
    if 2 <= hour < 6:
        return 0.75  # Night: lower threshold (more sensitive)
    elif 8 <= hour < 18:
        return 0.90  # Day: higher threshold (filter urban noise)
    else:
        return 0.85  # Default


def get_current_threshold() -> Tuple[float, int]:
    """
    Get current adaptive threshold based on local time.
    
    Returns:
        Tuple of (threshold, current_hour)
    """
    current_hour = datetime.now().hour
    threshold = get_adaptive_threshold_hour(current_hour)
    return threshold, current_hour


def unix_to_datetime(timestamp: float) -> datetime:
    """
    Convert Unix timestamp to datetime object.
    
    Args:
        timestamp: Unix epoch timestamp
        
    Returns:
        datetime object (UTC)
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def datetime_to_unix(dt: datetime) -> float:
    """
    Convert datetime object to Unix timestamp.
    
    Args:
        dt: datetime object
        
    Returns:
        Unix epoch timestamp
    """
    return dt.timestamp()


def get_current_unix() -> float:
    """Get current Unix timestamp"""
    return time.time()


def is_within_temporal_window(
    timestamp1: float,
    timestamp2: float,
    window_seconds: float = 2.0
) -> bool:
    """
    Check if two timestamps are within the temporal window.
    
    Used for temporal clustering of earthquake triggers.
    Paper uses 2-second sliding window.
    
    Args:
        timestamp1: First Unix timestamp
        timestamp2: Second Unix timestamp
        window_seconds: Temporal window size (default: 2s)
        
    Returns:
        True if within window
    """
    return abs(timestamp1 - timestamp2) <= window_seconds


def format_alert_time(timestamp: float) -> str:
    """
    Format timestamp for alert messages.
    
    Args:
        timestamp: Unix epoch timestamp
        
    Returns:
        Human-readable time string (local time)
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def calculate_p_s_delay(distance_km: float) -> float:
    """
    Calculate expected P-S wave arrival time difference.
    
    P-wave velocity: ~6 km/s
    S-wave velocity: ~3.5 km/s
    
    Args:
        distance_km: Distance from epicenter in kilometers
        
    Returns:
        Time delay in seconds between P and S wave arrival
    """
    p_velocity = 6.0  # km/s
    s_velocity = 3.5  # km/s
    
    p_time = distance_km / p_velocity
    s_time = distance_km / s_velocity
    
    return s_time - p_time


def calculate_warning_time(distance_km: float, detection_latency: float = 3.0) -> float:
    """
    Calculate available warning time before S-wave arrival.
    
    Args:
        distance_km: Distance from epicenter
        detection_latency: System detection + alert latency (default: 3s)
        
    Returns:
        Available warning time in seconds (can be negative if too close)
    """
    s_velocity = 3.5  # km/s
    s_wave_time = distance_km / s_velocity
    
    return s_wave_time - detection_latency


class TimeWindow:
    """
    Sliding time window for temporal clustering.
    
    Maintains triggers within a configurable window duration.
    """
    
    def __init__(self, window_seconds: float = 2.0):
        self.window_seconds = window_seconds
        self.triggers = []  # List of (timestamp, trigger_data)
    
    def add_trigger(self, timestamp: float, data: dict):
        """Add a new trigger to the window"""
        self.triggers.append((timestamp, data))
        self._cleanup(timestamp)
    
    def _cleanup(self, current_time: float):
        """Remove triggers outside the window"""
        cutoff = current_time - self.window_seconds
        self.triggers = [
            (ts, data) for ts, data in self.triggers
            if ts >= cutoff
        ]
    
    def get_active_triggers(self, current_time: float = None) -> list:
        """Get all triggers within the current window"""
        if current_time is None:
            current_time = get_current_unix()
        self._cleanup(current_time)
        return self.triggers
    
    def clear(self):
        """Clear all triggers"""
        self.triggers = []
