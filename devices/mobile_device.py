"""
Mobile Device Simulator

Simulates smartphone-based earthquake detection with:
- MEMS accelerometer (50 Hz, 3s window)
- PGA trigger (>0.05g)
- TFLite inference (<200ms)
- 91.4% accuracy (noisy due to human activity)
- Weight: 0.3 in consensus

From Paper Section 3.1 (Tier 1: Edge Detection):
"Mobile devices (~190) perform local inference using TFLite models (188 KiB INT8 1D CNN).
Mobile devices activate upon PGA trigger (>0.05g) to conserve battery."

Author: Muhammed Şara
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from enum import Enum
import hashlib
import time
import uuid
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config, DeviceType


class DeviceState(Enum):
    """Device operational state"""
    IDLE = "idle"
    TRIGGERED = "triggered"
    TRANSMITTING = "transmitting"
    OFFLINE = "offline"


@dataclass
class TriggerMessage:
    """
    Privacy-preserving trigger message (87 bytes payload).
    
    From Paper Section III.D:
    "Unlike centralized systems that transmit raw accelerometer streams,
    our architecture transmits only: Device ID (hashed), Device type,
    GPS coordinates (coarse-grained, 100m precision), Timestamp,
    Binary decision + confidence score"
    """
    device_id: str           # Hashed for anonymity
    device_type: str         # 'mobile' or 'iot_anchor'
    latitude: float          # Coarse-grained (100m precision)
    longitude: float         # Coarse-grained (100m precision)
    timestamp: float         # Unix epoch
    detected: bool           # Binary decision
    confidence: float        # AI model confidence [0, 1]
    pga: float              # Peak Ground Acceleration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'lat': round(self.latitude, 4),  # ~11m precision
            'lon': round(self.longitude, 4),
            'time': self.timestamp,
            'detected': self.detected,
            'confidence': round(self.confidence, 3),
            'pga': round(self.pga, 4)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TriggerMessage':
        """Create from dictionary"""
        return cls(
            device_id=data['device_id'],
            device_type=data['device_type'],
            latitude=data['lat'],
            longitude=data['lon'],
            timestamp=data['time'],
            detected=data['detected'],
            confidence=data['confidence'],
            pga=data.get('pga', 0.0)
        )


@dataclass
class MobileDevice:
    """
    Mobile Device Simulator
    
    Simulates a smartphone with MEMS accelerometer for earthquake detection.
    
    Attributes:
        device_id: Unique device identifier (hashed for privacy)
        latitude: Device latitude
        longitude: Device longitude
        accuracy: Detection accuracy (default: 0.88)
        sampling_rate: IMU sampling rate in Hz (default: 50)
        state: Current operational state
    """
    
    # Location (will be set during simulation)
    latitude: float = 0.0
    longitude: float = 0.0
    
    # Device properties
    device_id: str = field(default_factory=lambda: hashlib.sha256(
        uuid.uuid4().bytes
    ).hexdigest()[:16])
    
    accuracy: float = field(default_factory=lambda: config.device.mobile_accuracy)
    sampling_rate: int = field(default_factory=lambda: config.device.mobile_sampling_rate)
    inference_time_ms: float = field(default_factory=lambda: config.device.mobile_inference_time_ms)
    
    # State
    state: DeviceState = DeviceState.IDLE
    last_trigger_time: float = 0.0
    
    # Sensor noise characteristics
    noise_std: float = 0.02  # Standard deviation of noise (g units)
    
    # Model reference (set externally)
    model: Any = None
    
    def __post_init__(self):
        """Initialize device with unique ID if not provided"""
        if not self.device_id:
            self.device_id = hashlib.sha256(
                uuid.uuid4().bytes
            ).hexdigest()[:16]
    
    def set_location(self, lat: float, lon: float):
        """Set device GPS location"""
        # Apply coarse-graining for privacy (100m precision)
        precision = config.device.gps_precision_meters / 111000  # degrees
        self.latitude = round(lat / precision) * precision
        self.longitude = round(lon / precision) * precision
    
    def add_sensor_noise(self, data: np.ndarray) -> np.ndarray:
        """
        Add realistic sensor noise to accelerometer data.
        
        Simulates MEMS sensor characteristics including:
        - Gaussian noise
        - Device orientation uncertainty
        - Motion artifacts (if device is moving)
        """
        noise = np.random.normal(0, self.noise_std * 9.81, data.shape)
        return data + noise
    
    def process_earthquake(
        self,
        earthquake_data: np.ndarray,
        is_earthquake: bool = True,
        timestamp: float = None
    ) -> Optional[TriggerMessage]:
        """
        Process incoming seismic data and generate trigger if detected.
        
        Args:
            earthquake_data: Accelerometer data (n, 3) in m/s²
            is_earthquake: Ground truth label (for simulation)
            timestamp: Event timestamp
            
        Returns:
            TriggerMessage if earthquake detected, None otherwise
        """
        timestamp = timestamp or time.time()
        
        # Add sensor noise
        noisy_data = self.add_sensor_noise(earthquake_data)
        
        # Calculate PGA
        magnitudes = np.sqrt(np.sum(noisy_data ** 2, axis=1))
        pga = np.max(np.abs(magnitudes - 9.81)) / 9.81
        
        # Check PGA trigger threshold
        if pga < config.sta_lta.pga_threshold:
            return None
        
        # Simulate model inference
        self.state = DeviceState.TRIGGERED
        
        # Simulate accuracy-based detection
        # For simulation: we use accuracy to probabilistically determine correct detection
        random_val = np.random.random()
        
        if is_earthquake:
            # True earthquake: detect with `accuracy` probability
            detected = random_val < self.accuracy
            confidence = 0.85 + 0.14 * np.random.random() if detected else 0.3 + 0.3 * np.random.random()
        else:
            # False event: false positive with (1 - accuracy) probability
            detected = random_val > self.accuracy  # False positive
            confidence = 0.50 + 0.20 * np.random.random() if detected else 0.15 + 0.20 * np.random.random()
        
        if not detected:
            self.state = DeviceState.IDLE
            return None
        
        # Generate trigger message
        self.state = DeviceState.TRANSMITTING
        self.last_trigger_time = timestamp
        
        message = TriggerMessage(
            device_id=self.device_id,
            device_type=DeviceType.MOBILE.value,
            latitude=self.latitude,
            longitude=self.longitude,
            timestamp=timestamp,
            detected=True,
            confidence=float(confidence),
            pga=float(pga)
        )
        
        self.state = DeviceState.IDLE
        return message
    
    def process_with_model(
        self,
        accelerometer_data: np.ndarray,
        gyroscope_data: np.ndarray = None,
        timestamp: float = None
    ) -> Optional[TriggerMessage]:
        """
        Process data using actual AI model (for hardware deployment).
        
        Args:
            accelerometer_data: 3-axis accelerometer (n, 3) in m/s²
            gyroscope_data: 3-axis gyroscope (n, 3) in rad/s
            timestamp: Current timestamp
            
        Returns:
            TriggerMessage if earthquake detected
        """
        if self.model is None:
            raise ValueError("Model not set. Use set_model() first.")
        
        timestamp = timestamp or time.time()
        
        # Add noise
        noisy_data = self.add_sensor_noise(accelerometer_data)
        
        # Calculate PGA for trigger check
        magnitudes = np.sqrt(np.sum(noisy_data ** 2, axis=1))
        pga = np.max(np.abs(magnitudes - 9.81)) / 9.81
        
        if pga < config.sta_lta.pga_threshold:
            return None
        
        # Prepare model input
        if gyroscope_data is None:
            gyroscope_data = np.zeros_like(noisy_data)
        
        combined = np.hstack([noisy_data, gyroscope_data])
        
        # Ensure correct shape
        if len(combined) > 150:
            combined = combined[-150:]
        elif len(combined) < 150:
            padding = np.zeros((150 - len(combined), 6))
            combined = np.vstack([padding, combined])
        
        # Run inference
        if hasattr(self.model, 'predict_earthquake'):
            detected, confidence = self.model.predict_earthquake(combined)
        else:
            detected, confidence = self.model.predict_single(combined)
        
        if not detected:
            return None
        
        return TriggerMessage(
            device_id=self.device_id,
            device_type=DeviceType.MOBILE.value,
            latitude=self.latitude,
            longitude=self.longitude,
            timestamp=timestamp,
            detected=True,
            confidence=float(confidence),
            pga=float(pga)
        )
    
    def set_model(self, model):
        """Set AI model for inference"""
        self.model = model
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize device state"""
        return {
            'device_id': self.device_id,
            'device_type': 'mobile',
            'latitude': self.latitude,
            'longitude': self.longitude,
            'accuracy': self.accuracy,
            'state': self.state.value
        }


def create_mobile_device(lat: float, lon: float, **kwargs) -> MobileDevice:
    """Factory function to create a mobile device at specific location"""
    device = MobileDevice(**kwargs)
    device.set_location(lat, lon)
    return device


def create_mobile_fleet(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    num_devices: int,
    seed: int = None
) -> list:
    """
    Create a fleet of mobile devices distributed around a center point.
    
    Args:
        center_lat, center_lon: Center coordinates
        radius_km: Distribution radius in km
        num_devices: Number of devices to create
        seed: Random seed for reproducibility
        
    Returns:
        List of MobileDevice instances
    """
    if seed is not None:
        np.random.seed(seed)
    
    devices = []
    
    for _ in range(num_devices):
        # Random angle and distance
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, radius_km)
        
        # Convert to lat/lon offset
        lat_offset = distance * np.cos(angle) / 111  # km to degrees
        lon_offset = distance * np.sin(angle) / (111 * np.cos(np.radians(center_lat)))
        
        device = MobileDevice()
        device.set_location(
            center_lat + lat_offset,
            center_lon + lon_offset
        )
        devices.append(device)
    
    return devices
