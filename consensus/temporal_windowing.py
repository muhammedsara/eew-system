"""
Temporal Windowing for Consensus Protocol

Stage 2 of the consensus protocol (Paper Section III.B):
"For each spatial cluster, we apply a 2-second sliding window to capture 
near-simultaneous triggers. This duration accommodates P-S wave arrival 
time differences (typically 1–3s for urban EEW scenarios)."

Author: Muhammed Şara
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


@dataclass
class TemporalWindow:
    """Represents a time window of triggers"""
    window_id: int
    start_time: float
    end_time: float
    triggers: List[Dict] = field(default_factory=list)
    mobile_count: int = 0
    iot_count: int = 0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def total_count(self) -> int:
        return self.mobile_count + self.iot_count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'window_id': self.window_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'mobile_count': self.mobile_count,
            'iot_count': self.iot_count,
            'total_count': self.total_count,
            'triggers': self.triggers
        }


class TemporalWindower:
    """
    Temporal windowing for earthquake trigger grouping.
    
    Groups triggers that occur within a sliding time window.
    Default window size: 2 seconds.
    
    Attributes:
        window_seconds: Window duration in seconds (default: 2s)
    """
    
    def __init__(self, window_seconds: float = None):
        """
        Initialize temporal windower.
        
        Args:
            window_seconds: Window duration (default from config: 2s)
        """
        self.window_seconds = window_seconds or config.consensus.temporal_window_seconds
        
        # For streaming mode
        self._current_window = None
        self._window_counter = 0
        self._pending_triggers = []
    
    def group_triggers(self, triggers: List[Dict]) -> List[TemporalWindow]:
        """
        Group triggers into temporal windows.
        
        Uses sliding window approach: each window starts at the first
        trigger and spans window_seconds duration.
        
        Args:
            triggers: List of trigger dictionaries with 'time' or 'timestamp' key
            
        Returns:
            List of TemporalWindow objects
        """
        if not triggers:
            return []
        
        # Sort by timestamp
        sorted_triggers = sorted(
            triggers,
            key=lambda t: t.get('time', t.get('timestamp', 0))
        )
        
        windows = []
        window_id = 0
        i = 0
        
        while i < len(sorted_triggers):
            # Start new window
            window_start = sorted_triggers[i].get(
                'time', sorted_triggers[i].get('timestamp', 0)
            )
            window_end = window_start + self.window_seconds
            
            # Collect triggers in this window
            window_triggers = []
            mobile_count = 0
            iot_count = 0
            
            j = i
            while j < len(sorted_triggers):
                t_time = sorted_triggers[j].get(
                    'time', sorted_triggers[j].get('timestamp', 0)
                )
                
                if t_time <= window_end:
                    window_triggers.append(sorted_triggers[j])
                    if sorted_triggers[j].get('device_type') == 'mobile':
                        mobile_count += 1
                    else:
                        iot_count += 1
                    j += 1
                else:
                    break
            
            # Create window
            windows.append(TemporalWindow(
                window_id=window_id,
                start_time=window_start,
                end_time=window_end,
                triggers=window_triggers,
                mobile_count=mobile_count,
                iot_count=iot_count
            ))
            
            window_id += 1
            i = j  # Move to next unprocessed trigger
        
        return windows
    
    def group_with_overlap(
        self,
        triggers: List[Dict],
        overlap_percent: float = 0.5
    ) -> List[TemporalWindow]:
        """
        Group triggers using overlapping windows.
        
        Overlapping windows ensure no triggers are missed at window boundaries.
        
        Args:
            triggers: List of trigger dictionaries
            overlap_percent: Window overlap (0.5 = 50% overlap)
            
        Returns:
            List of potentially overlapping TemporalWindow objects
        """
        if not triggers:
            return []
        
        # Sort by timestamp
        sorted_triggers = sorted(
            triggers,
            key=lambda t: t.get('time', t.get('timestamp', 0))
        )
        
        # Get time range
        t_start = sorted_triggers[0].get('time', sorted_triggers[0].get('timestamp', 0))
        t_end = sorted_triggers[-1].get('time', sorted_triggers[-1].get('timestamp', 0))
        
        # Calculate step size
        step = self.window_seconds * (1 - overlap_percent)
        
        windows = []
        window_id = 0
        current_start = t_start
        
        while current_start <= t_end:
            window_end = current_start + self.window_seconds
            
            # Find triggers in this window
            window_triggers = [
                t for t in sorted_triggers
                if current_start <= t.get('time', t.get('timestamp', 0)) <= window_end
            ]
            
            if window_triggers:
                mobile_count = sum(
                    1 for t in window_triggers 
                    if t.get('device_type') == 'mobile'
                )
                iot_count = sum(
                    1 for t in window_triggers 
                    if t.get('device_type') == 'iot_anchor'
                )
                
                windows.append(TemporalWindow(
                    window_id=window_id,
                    start_time=current_start,
                    end_time=window_end,
                    triggers=window_triggers,
                    mobile_count=mobile_count,
                    iot_count=iot_count
                ))
                window_id += 1
            
            current_start += step
        
        return windows
    
    def process_streaming(
        self,
        trigger: Dict,
        current_time: float = None
    ) -> Optional[TemporalWindow]:
        """
        Process trigger in streaming mode.
        
        For real-time processing where triggers arrive one at a time.
        Returns a completed window when window duration is exceeded.
        
        Args:
            trigger: New trigger dictionary
            current_time: Current timestamp (for window expiry check)
            
        Returns:
            Completed TemporalWindow if window expired, None otherwise
        """
        current_time = current_time or time.time()
        trigger_time = trigger.get('time', trigger.get('timestamp', current_time))
        
        # Check if current window expired
        completed_window = None
        
        if self._current_window is not None:
            if trigger_time > self._current_window.end_time:
                # Window expired, return it
                completed_window = self._current_window
                self._current_window = None
        
        # Start new window if needed
        if self._current_window is None:
            self._window_counter += 1
            self._current_window = TemporalWindow(
                window_id=self._window_counter,
                start_time=trigger_time,
                end_time=trigger_time + self.window_seconds,
                triggers=[],
                mobile_count=0,
                iot_count=0
            )
        
        # Add trigger to current window
        self._current_window.triggers.append(trigger)
        if trigger.get('device_type') == 'mobile':
            self._current_window.mobile_count += 1
        else:
            self._current_window.iot_count += 1
        
        return completed_window
    
    def flush_streaming(self) -> Optional[TemporalWindow]:
        """
        Flush current streaming window.
        
        Returns:
            Current window if exists, None otherwise
        """
        window = self._current_window
        self._current_window = None
        return window
    
    def reset_streaming(self):
        """Reset streaming state"""
        self._current_window = None
        self._window_counter = 0
        self._pending_triggers = []
    
    def find_peak_window(self, windows: List[TemporalWindow]) -> Optional[TemporalWindow]:
        """
        Find the window with the most triggers (peak activity).
        
        Args:
            windows: List of temporal windows
            
        Returns:
            Window with highest trigger count
        """
        if not windows:
            return None
        
        return max(windows, key=lambda w: w.total_count)
    
    def find_windows_with_iot(
        self,
        windows: List[TemporalWindow],
        min_iot: int = 1
    ) -> List[TemporalWindow]:
        """
        Filter windows that have at least min_iot IoT anchor triggers.
        
        This is important for validation (Paper Equation 3):
        "The IoT requirement (n_IoT ≥ 1) ensures at least one high-reliability 
        sensor confirms the event."
        
        Args:
            windows: List of temporal windows
            min_iot: Minimum IoT anchor count required
            
        Returns:
            Filtered list of windows
        """
        return [w for w in windows if w.iot_count >= min_iot]
