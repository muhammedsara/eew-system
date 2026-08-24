"""
STA/LTA (Short-Term Average / Long-Term Average) Trigger Algorithm

Battery-efficient pre-trigger for earthquake detection.
Only activates the expensive AI model when PGA > 0.05g threshold is exceeded.

From Paper Section III.E:
"Two-stage triggering (STA/LTA + AI) minimizes false activations and 
battery consumption, reducing passive monitoring from 25 mAh to 3-5 mAh"

Author: Muhammed Şara
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


@dataclass
class TriggerEvent:
    """Represents a trigger event from STA/LTA algorithm"""
    timestamp: float
    pga: float
    sta_value: float
    lta_value: float
    ratio: float
    triggered: bool


def calculate_pga(accelerometer_data: np.ndarray) -> float:
    """
    Calculate Peak Ground Acceleration (PGA) from 3-axis accelerometer data.
    
    PGA is the maximum vector magnitude of acceleration during a time window.
    
    Args:
        accelerometer_data: Array of shape (n, 3) with [ax, ay, az] in m/s²
        
    Returns:
        PGA in g units (1g ≈ 9.81 m/s²)
    """
    if accelerometer_data.ndim == 1:
        accelerometer_data = accelerometer_data.reshape(-1, 3)
    
    # Calculate vector magnitude for each sample
    magnitudes = np.sqrt(np.sum(accelerometer_data ** 2, axis=1))
    
    # Remove gravity component (approximately 1g when stationary)
    # This gives us the acceleration relative to rest
    magnitudes_detrended = np.abs(magnitudes - 9.81)
    
    # Peak value in g units
    pga = np.max(magnitudes_detrended) / 9.81
    
    return pga


def calculate_vector_magnitude(data: np.ndarray) -> np.ndarray:
    """
    Calculate vector magnitude from 3-axis data.
    
    Args:
        data: Array of shape (n, 3) with [x, y, z]
        
    Returns:
        Array of magnitude values
    """
    return np.sqrt(np.sum(data ** 2, axis=1))


class STALTA:
    """
    STA/LTA (Short-Term Average / Long-Term Average) Trigger
    
    A classic seismological algorithm for detecting sudden changes in
    signal energy, used here as a battery-efficient pre-trigger.
    
    Algorithm:
    1. Continuously compute STA (short window average)
    2. Continuously compute LTA (long window average)
    3. When STA/LTA ratio exceeds threshold → trigger
    4. Optionally, check PGA threshold for additional filtering
    
    This is much cheaper than running AI inference continuously.
    
    Attributes:
        sta_window: Short-term averaging window in seconds
        lta_window: Long-term averaging window in seconds
        trigger_ratio: STA/LTA ratio to trigger detection
        pga_threshold: Minimum PGA (in g) to consider triggering
        sampling_rate: Sampling rate in Hz
    """
    
    def __init__(
        self,
        sta_window: float = None,
        lta_window: float = None,
        trigger_ratio: float = None,
        pga_threshold: float = None,
        sampling_rate: int = None
    ):
        """
        Initialize STA/LTA trigger.
        
        Args:
            sta_window: STA window in seconds (default: 0.5s)
            lta_window: LTA window in seconds (default: 10s)
            trigger_ratio: Trigger threshold (default: 3.0)
            pga_threshold: PGA threshold in g (default: 0.05g)
            sampling_rate: Sampling rate in Hz (default: 50)
        """
        cfg = config.sta_lta
        
        self.sta_window = sta_window or cfg.sta_window
        self.lta_window = lta_window or cfg.lta_window
        self.trigger_ratio = trigger_ratio or cfg.trigger_ratio
        self.pga_threshold = pga_threshold or cfg.pga_threshold
        self.detrigger_ratio = cfg.detrigger_ratio
        self.sampling_rate = sampling_rate or config.model.sampling_rate
        
        # Convert windows to samples
        self.sta_samples = int(self.sta_window * self.sampling_rate)
        self.lta_samples = int(self.lta_window * self.sampling_rate)
        
        # State for streaming processing
        self._buffer = []
        self._triggered = False
        self._lta_history = []
    
    def reset(self):
        """Reset internal state"""
        self._buffer = []
        self._triggered = False
        self._lta_history = []
    
    def _calculate_sta_lta_classic(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate STA/LTA using classic recursive algorithm.
        
        Args:
            data: 1D array of signal values (e.g., magnitude)
            
        Returns:
            Tuple of (sta_values, lta_values, ratio_values)
        """
        n = len(data)
        sta = np.zeros(n)
        lta = np.zeros(n)
        ratio = np.zeros(n)
        
        # Use squared signal (energy)
        data_squared = data ** 2
        
        # Initialize with cumulative sums
        for i in range(n):
            # STA: average of recent samples
            sta_start = max(0, i - self.sta_samples + 1)
            sta[i] = np.mean(data_squared[sta_start:i+1])
            
            # LTA: average of long-term samples
            lta_start = max(0, i - self.lta_samples + 1)
            lta[i] = np.mean(data_squared[lta_start:i+1])
            
            # Ratio (avoid division by zero)
            if lta[i] > 1e-10:
                ratio[i] = sta[i] / lta[i]
            else:
                ratio[i] = 0
        
        return sta, lta, ratio
    
    def process_window(
        self,
        accelerometer_data: np.ndarray,
        timestamp: float = 0.0
    ) -> TriggerEvent:
        """
        Process a window of accelerometer data for trigger detection.
        
        Args:
            accelerometer_data: Array of shape (n, 3) with [ax, ay, az] in m/s²
            timestamp: Current timestamp
            
        Returns:
            TriggerEvent with trigger status and diagnostics
        """
        # Calculate magnitude
        magnitude = calculate_vector_magnitude(accelerometer_data)
        
        # Remove gravity bias
        magnitude_detrended = np.abs(magnitude - 9.81)
        
        # Calculate STA/LTA
        sta, lta, ratio = self._calculate_sta_lta_classic(magnitude_detrended)
        
        # Get latest values
        current_sta = sta[-1]
        current_lta = lta[-1]
        current_ratio = ratio[-1]
        
        # Calculate PGA
        pga = calculate_pga(accelerometer_data)
        
        # Trigger logic
        triggered = False
        if current_ratio >= self.trigger_ratio and pga >= self.pga_threshold:
            triggered = True
        
        return TriggerEvent(
            timestamp=timestamp,
            pga=pga,
            sta_value=current_sta,
            lta_value=current_lta,
            ratio=current_ratio,
            triggered=triggered
        )
    
    def process_stream(
        self,
        sample: np.ndarray,
        timestamp: float = 0.0
    ) -> Optional[TriggerEvent]:
        """
        Process a single sample in streaming mode.
        
        This is used for real-time processing where samples arrive one at a time.
        
        Args:
            sample: Single sample [ax, ay, az] in m/s²
            timestamp: Sample timestamp
            
        Returns:
            TriggerEvent if enough samples accumulated, None otherwise
        """
        # Add to buffer
        self._buffer.append(sample)
        
        # Need at least LTA window of samples
        if len(self._buffer) < self.lta_samples:
            return None
        
        # Keep buffer at maximum size
        if len(self._buffer) > self.lta_samples:
            self._buffer = self._buffer[-self.lta_samples:]
        
        # Process current window
        data = np.array(self._buffer)
        return self.process_window(data, timestamp)
    
    def detect_triggers(
        self,
        accelerometer_data: np.ndarray,
        timestamps: np.ndarray = None
    ) -> List[TriggerEvent]:
        """
        Detect all trigger events in a dataset.
        
        Args:
            accelerometer_data: Array of shape (n, 3)
            timestamps: Optional array of timestamps
            
        Returns:
            List of TriggerEvent objects for triggered windows
        """
        if timestamps is None:
            timestamps = np.arange(len(accelerometer_data)) / self.sampling_rate
        
        # Calculate magnitude
        magnitude = calculate_vector_magnitude(accelerometer_data)
        magnitude_detrended = np.abs(magnitude - 9.81)
        
        # Calculate STA/LTA for entire dataset
        sta, lta, ratio = self._calculate_sta_lta_classic(magnitude_detrended)
        
        # Find trigger points
        triggers = []
        triggered_state = False
        
        for i in range(self.lta_samples, len(ratio)):
            # Calculate PGA for recent window
            window_start = max(0, i - self.sta_samples)
            pga = np.max(magnitude_detrended[window_start:i+1]) / 9.81
            
            # State machine: trigger/detrigger logic
            if not triggered_state:
                if ratio[i] >= self.trigger_ratio and pga >= self.pga_threshold:
                    triggered_state = True
                    triggers.append(TriggerEvent(
                        timestamp=timestamps[i],
                        pga=pga,
                        sta_value=sta[i],
                        lta_value=lta[i],
                        ratio=ratio[i],
                        triggered=True
                    ))
            else:
                if ratio[i] < self.detrigger_ratio:
                    triggered_state = False
        
        return triggers


class CombinedTrigger:
    """
    Combined STA/LTA + AI trigger for efficient earthquake detection.
    
    Two-stage detection:
    1. STA/LTA provides cheap, continuous monitoring
    2. When triggered, activate expensive AI model for confirmation
    
    This saves significant battery compared to running AI continuously.
    """
    
    def __init__(self, model=None):
        """
        Initialize combined trigger.
        
        Args:
            model: Optional pre-loaded AI model (EarthquakeDetectorCNN or TFLiteInference)
        """
        self.sta_lta = STALTA()
        self.model = model
        
        # Statistics
        self.sta_lta_triggers = 0
        self.ai_confirmations = 0
    
    def set_model(self, model):
        """Set the AI model for confirmation"""
        self.model = model
    
    def process(
        self,
        accelerometer_data: np.ndarray,
        gyroscope_data: np.ndarray = None,
        timestamp: float = 0.0
    ) -> Tuple[bool, float, dict]:
        """
        Process sensor data through two-stage detection.
        
        Args:
            accelerometer_data: Array of shape (n, 3)
            gyroscope_data: Optional array of shape (n, 3), zeros if not available
            timestamp: Current timestamp
            
        Returns:
            Tuple of (is_earthquake, confidence, diagnostics)
        """
        # Stage 1: STA/LTA trigger
        trigger_event = self.sta_lta.process_window(accelerometer_data, timestamp)
        
        diagnostics = {
            'sta_lta_triggered': trigger_event.triggered,
            'pga': trigger_event.pga,
            'sta_lta_ratio': trigger_event.ratio,
            'ai_used': False,
            'ai_confidence': 0.0
        }
        
        if not trigger_event.triggered:
            return False, 0.0, diagnostics
        
        self.sta_lta_triggers += 1
        
        # Stage 2: AI confirmation
        if self.model is None:
            # No model available, return STA/LTA result only
            return True, 0.7, diagnostics
        
        # Prepare input for AI model
        if gyroscope_data is None:
            gyroscope_data = np.zeros_like(accelerometer_data)
        
        # Combine accel + gyro into 6-channel input
        combined = np.hstack([accelerometer_data, gyroscope_data])
        
        # Ensure correct shape (150 timesteps)
        if len(combined) > 150:
            combined = combined[-150:]
        elif len(combined) < 150:
            # Pad with zeros
            padding = np.zeros((150 - len(combined), 6))
            combined = np.vstack([padding, combined])
        
        # Run AI inference
        diagnostics['ai_used'] = True
        
        # Handle both full model and TFLite
        if hasattr(self.model, 'predict_earthquake'):
            is_earthquake, confidence = self.model.predict_earthquake(combined)
        else:
            is_earthquake, confidence = self.model.predict_single(combined)
        
        diagnostics['ai_confidence'] = confidence
        
        if is_earthquake:
            self.ai_confirmations += 1
        
        return is_earthquake, confidence, diagnostics
    
    def get_stats(self) -> dict:
        """Get trigger statistics"""
        return {
            'sta_lta_triggers': self.sta_lta_triggers,
            'ai_confirmations': self.ai_confirmations,
            'confirmation_rate': (
                self.ai_confirmations / self.sta_lta_triggers 
                if self.sta_lta_triggers > 0 else 0
            )
        }
