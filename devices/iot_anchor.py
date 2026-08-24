"""
IoT Anchor Simulator

Simulates calibrated IoT anchor stations (Raspberry Pi 4 + MPU-6050) with:
- High accuracy (97.1%)
- Fixed location (no GPS drift)
- Always-on operation with UPS backup
- Weight: 0.7 in consensus (higher reliability)

From Paper Section III.A (Layer 1: Edge Detection):
"IoT anchors (Raspberry Pi 4 + calibrated MPU-6050) run continuously with UPS backup."

Author: Muhammed Şara
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import hashlib
import time
import uuid
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config, DeviceType
from devices.mobile_device import TriggerMessage, DeviceState


@dataclass
class IoTAnchor:
    """
    IoT Anchor Station Simulator
    
    Simulates a dedicated earthquake detection station with:
    - Raspberry Pi 4 + calibrated MPU-6050
    - Fixed mounting (no motion artifacts)
    - Higher sampling rate (100 Hz)
    - 97.1% accuracy (calibrated sensors, paper Section 4.3)
    - Always-on with UPS backup
    
    Attributes:
        device_id: Unique station identifier
        latitude: Fixed station latitude
        longitude: Fixed station longitude
        accuracy: Detection accuracy (default: 0.95)
        sampling_rate: IMU sampling rate in Hz (default: 100)
    """
    
    # Location (fixed for IoT anchors)
    latitude: float = 0.0
    longitude: float = 0.0
    
    # Device properties
    device_id: str = field(default_factory=lambda: f"IOT_{hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:12]}")
    station_name: str = ""
    
    accuracy: float = field(default_factory=lambda: config.device.iot_accuracy)
    sampling_rate: int = field(default_factory=lambda: config.device.iot_sampling_rate)
    inference_time_ms: float = field(default_factory=lambda: config.device.iot_inference_time_ms)
    
    # State
    state: DeviceState = DeviceState.IDLE
    is_online: bool = True
    has_ups_power: bool = True
    last_heartbeat: float = 0.0
    last_trigger_time: float = 0.0
    
    # Calibrated sensor characteristics (much lower noise than mobile)
    noise_std: float = 0.005  # Very low noise due to calibration
    
    # Model reference
    model: Any = None
    
    def __post_init__(self):
        """Initialize with unique ID if not provided"""
        if not self.station_name:
            self.station_name = f"Station_{self.device_id[-6:]}"
    
    def set_location(self, lat: float, lon: float, name: str = None):
        """
        Set fixed station location.
        
        Unlike mobile devices, IoT anchors have precise, fixed locations.
        No coarse-graining applied.
        """
        self.latitude = lat
        self.longitude = lon
        if name:
            self.station_name = name
    
    def add_sensor_noise(self, data: np.ndarray) -> np.ndarray:
        """
        Add minimal sensor noise (calibrated sensor).
        
        Much lower noise compared to mobile MEMS sensors.
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
        
        IoT anchors have higher accuracy and reliability than mobile devices.
        
        Args:
            earthquake_data: Accelerometer data (n, 3) in m/s²
            is_earthquake: Ground truth label (for simulation)
            timestamp: Event timestamp
            
        Returns:
            TriggerMessage if earthquake detected, None otherwise
        """
        if not self.is_online:
            return None
        
        timestamp = timestamp or time.time()
        
        # Add minimal sensor noise
        noisy_data = self.add_sensor_noise(earthquake_data)
        
        # Calculate PGA
        magnitudes = np.sqrt(np.sum(noisy_data ** 2, axis=1))
        pga = np.max(np.abs(magnitudes - 9.81)) / 9.81
        
        # Lower PGA threshold for calibrated sensors
        iot_pga_threshold = config.sta_lta.pga_threshold * 0.8  # More sensitive
        
        if pga < iot_pga_threshold:
            return None
        
        # Simulate model inference with higher accuracy
        self.state = DeviceState.TRIGGERED
        
        random_val = np.random.random()
        
        if is_earthquake:
            # True earthquake: detect with higher accuracy
            detected = random_val < self.accuracy
            confidence = 0.90 + 0.09 * np.random.random() if detected else 0.4 + 0.2 * np.random.random()
        else:
            # False event: lower false positive rate
            detected = random_val > self.accuracy  # Very low FP rate
            confidence = 0.45 + 0.20 * np.random.random() if detected else 0.10 + 0.15 * np.random.random()
        
        if not detected:
            self.state = DeviceState.IDLE
            return None
        
        # Generate trigger message
        self.state = DeviceState.TRANSMITTING
        self.last_trigger_time = timestamp
        
        message = TriggerMessage(
            device_id=self.device_id,
            device_type=DeviceType.IOT_ANCHOR.value,
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
        Process data using actual AI model.
        
        Args:
            accelerometer_data: 3-axis accelerometer (n, 3) in m/s²
            gyroscope_data: 3-axis gyroscope (n, 3) in rad/s
            timestamp: Current timestamp
            
        Returns:
            TriggerMessage if earthquake detected
        """
        if self.model is None:
            raise ValueError("Model not set. Use set_model() first.")
        
        if not self.is_online:
            return None
        
        timestamp = timestamp or time.time()
        
        # Minimal noise for calibrated sensor
        noisy_data = self.add_sensor_noise(accelerometer_data)
        
        # Calculate PGA
        magnitudes = np.sqrt(np.sum(noisy_data ** 2, axis=1))
        pga = np.max(np.abs(magnitudes - 9.81)) / 9.81
        
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
            device_type=DeviceType.IOT_ANCHOR.value,
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
    
    def heartbeat(self) -> Dict[str, Any]:
        """
        Send heartbeat for health monitoring.
        
        Returns:
            Status dictionary
        """
        self.last_heartbeat = time.time()
        return {
            'device_id': self.device_id,
            'station_name': self.station_name,
            'is_online': self.is_online,
            'has_ups_power': self.has_ups_power,
            'state': self.state.value,
            'timestamp': self.last_heartbeat
        }
    
    def simulate_power_outage(self):
        """Simulate power outage (switch to UPS)"""
        self.has_ups_power = True
        # UPS provides limited runtime
    
    def simulate_network_issue(self, offline: bool = True):
        """Simulate network connectivity issues"""
        self.is_online = not offline
        if offline:
            self.state = DeviceState.OFFLINE
        else:
            self.state = DeviceState.IDLE
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize device state"""
        return {
            'device_id': self.device_id,
            'device_type': 'iot_anchor',
            'station_name': self.station_name,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'accuracy': self.accuracy,
            'is_online': self.is_online,
            'has_ups_power': self.has_ups_power,
            'state': self.state.value
        }


def create_iot_anchor(lat: float, lon: float, name: str = None, **kwargs) -> IoTAnchor:
    """Factory function to create an IoT anchor at specific location"""
    anchor = IoTAnchor(**kwargs)
    anchor.set_location(lat, lon, name)
    return anchor


def create_iot_grid(
    center_lat: float,
    center_lon: float,
    grid_spacing_km: float = 10.0,
    grid_size: int = 3
) -> List[IoTAnchor]:
    """
    Create a grid of IoT anchors around a center point.
    
    From Paper Section IV.A:
    "IoT anchors: 10 devices in grid pattern (10 km spacing)"
    
    Args:
        center_lat, center_lon: Center coordinates
        grid_spacing_km: Spacing between anchors in km
        grid_size: Grid dimension (3 = 3x3 = 9 anchors, plus center = 10)
        
    Returns:
        List of IoTAnchor instances
    """
    anchors = []
    half_size = grid_size // 2
    
    # Convert km to degrees
    lat_step = grid_spacing_km / 111
    lon_step = grid_spacing_km / (111 * np.cos(np.radians(center_lat)))
    
    station_num = 1
    
    for i in range(-half_size, half_size + 1):
        for j in range(-half_size, half_size + 1):
            lat = center_lat + i * lat_step
            lon = center_lon + j * lon_step
            
            anchor = IoTAnchor()
            anchor.set_location(lat, lon, f"Station_{station_num:02d}")
            anchors.append(anchor)
            station_num += 1
    
    # If we need exactly 10 anchors, add center anchor
    if len(anchors) == 9:
        center_anchor = IoTAnchor()
        center_anchor.set_location(center_lat, center_lon, "Station_Center")
        anchors.append(center_anchor)
    
    return anchors
