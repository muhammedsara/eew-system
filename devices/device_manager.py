"""
Device Manager

Manages the fleet of mobile devices and IoT anchors.
Handles device registration, message routing, and network simulation.

Author: Muhammed Şara
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from devices.mobile_device import MobileDevice, TriggerMessage, create_mobile_fleet
from devices.iot_anchor import IoTAnchor, create_iot_grid
from utils.haversine import haversine_distance, calculate_affected_radius


@dataclass
class DeviceManager:
    """
    Manages hybrid mobile-IoT device fleet.
    
    Responsibilities:
    - Device registration and tracking
    - Spatial distribution
    - Message collection and routing
    - Network simulation (packet loss)
    
    Attributes:
        mobile_devices: List of mobile devices
        iot_anchors: List of IoT anchor stations
    """
    
    mobile_devices: List[MobileDevice] = field(default_factory=list)
    iot_anchors: List[IoTAnchor] = field(default_factory=list)
    
    # Network simulation
    packet_loss_rate: float = 0.0
    network_latency_ms: float = 50.0
    
    # Statistics
    total_triggers_sent: int = 0
    total_triggers_lost: int = 0
    
    def __post_init__(self):
        """Initialize device indices"""
        self._device_index = {}
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild device lookup index"""
        self._device_index = {}
        for device in self.mobile_devices:
            self._device_index[device.device_id] = device
        for anchor in self.iot_anchors:
            self._device_index[anchor.device_id] = anchor
    
    def add_mobile_device(self, device: MobileDevice):
        """Add a mobile device to the fleet"""
        self.mobile_devices.append(device)
        self._device_index[device.device_id] = device
    
    def add_iot_anchor(self, anchor: IoTAnchor):
        """Add an IoT anchor to the fleet"""
        self.iot_anchors.append(anchor)
        self._device_index[anchor.device_id] = anchor
    
    def get_device(self, device_id: str) -> Optional[Any]:
        """Get device by ID"""
        return self._device_index.get(device_id)
    
    def setup_scenario(
        self,
        center_lat: float,
        center_lon: float,
        num_mobile: int = 190,
        num_iot: int = 10,
        distribution_radius_km: float = 50.0,
        iot_grid_spacing_km: float = 10.0,
        seed: int = None
    ):
        """
        Set up a simulation scenario with devices distributed around a center.
        
        Args:
            center_lat, center_lon: Epicenter/center coordinates
            num_mobile: Number of mobile devices
            num_iot: Number of IoT anchors
            distribution_radius_km: Radius for mobile device distribution
            iot_grid_spacing_km: Spacing between IoT anchors
            seed: Random seed
        """
        # Create mobile fleet
        self.mobile_devices = create_mobile_fleet(
            center_lat, center_lon,
            distribution_radius_km,
            num_mobile,
            seed
        )
        
        # Create IoT grid
        grid_size = int(np.sqrt(num_iot - 1))  # -1 for center anchor
        self.iot_anchors = create_iot_grid(
            center_lat, center_lon,
            iot_grid_spacing_km,
            grid_size
        )
        
        self._rebuild_index()
    
    def get_devices_within_radius(
        self,
        epicenter_lat: float,
        epicenter_lon: float,
        radius_km: float
    ) -> Tuple[List[MobileDevice], List[IoTAnchor]]:
        """
        Get all devices within a radius of the epicenter.
        
        Args:
            epicenter_lat, epicenter_lon: Epicenter coordinates
            radius_km: Affected radius
            
        Returns:
            Tuple of (affected_mobile, affected_iot)
        """
        affected_mobile = []
        affected_iot = []
        
        for device in self.mobile_devices:
            dist = haversine_distance(
                epicenter_lat, epicenter_lon,
                device.latitude, device.longitude
            )
            if dist <= radius_km:
                affected_mobile.append(device)
        
        for anchor in self.iot_anchors:
            dist = haversine_distance(
                epicenter_lat, epicenter_lon,
                anchor.latitude, anchor.longitude
            )
            if dist <= radius_km:
                affected_iot.append(anchor)
        
        return affected_mobile, affected_iot
    
    def simulate_earthquake_triggers(
        self,
        earthquake_data: np.ndarray,
        epicenter_lat: float,
        epicenter_lon: float,
        magnitude: float,
        timestamp: float = None
    ) -> List[TriggerMessage]:
        """
        Simulate earthquake detection across all affected devices.
        
        Args:
            earthquake_data: Seismic waveform data
            epicenter_lat, epicenter_lon: Epicenter
            magnitude: Earthquake magnitude
            timestamp: Event timestamp
            
        Returns:
            List of trigger messages from devices that detected the earthquake
        """
        timestamp = timestamp or time.time()
        
        # Calculate affected radius
        radius_km = calculate_affected_radius(magnitude)
        
        # Get affected devices
        affected_mobile, affected_iot = self.get_devices_within_radius(
            epicenter_lat, epicenter_lon, radius_km
        )
        
        triggers = []
        
        # Process mobile devices
        for device in affected_mobile:
            message = device.process_earthquake(
                earthquake_data,
                is_earthquake=True,
                timestamp=timestamp
            )
            if message:
                # Apply packet loss
                if not self._simulate_packet_loss():
                    triggers.append(message)
                    self.total_triggers_sent += 1
                else:
                    self.total_triggers_lost += 1
        
        # Process IoT anchors
        for anchor in affected_iot:
            message = anchor.process_earthquake(
                earthquake_data,
                is_earthquake=True,
                timestamp=timestamp
            )
            if message:
                if not self._simulate_packet_loss():
                    triggers.append(message)
                    self.total_triggers_sent += 1
                else:
                    self.total_triggers_lost += 1
        
        return triggers
    
    def simulate_false_positive_triggers(
        self,
        noise_data: np.ndarray,
        source_lat: float,
        source_lon: float,
        affected_radius_km: float = 1.0,
        timestamp: float = None
    ) -> List[TriggerMessage]:
        """
        Simulate false positive triggers from noise sources.
        
        Args:
            noise_data: Noise waveform data (e.g., from WISDM/UCI HAR)
            source_lat, source_lon: Noise source location
            affected_radius_km: Radius of noise effect
            timestamp: Event timestamp
            
        Returns:
            List of false positive trigger messages
        """
        timestamp = timestamp or time.time()
        
        # Get affected devices (typically small radius for local noise)
        affected_mobile, affected_iot = self.get_devices_within_radius(
            source_lat, source_lon, affected_radius_km
        )
        
        triggers = []
        
        # Process mobile devices (more susceptible to false positives)
        for device in affected_mobile:
            message = device.process_earthquake(
                noise_data,
                is_earthquake=False,  # Ground truth: NOT earthquake
                timestamp=timestamp
            )
            if message:
                if not self._simulate_packet_loss():
                    triggers.append(message)
                    self.total_triggers_sent += 1
        
        # IoT anchors rarely produce false positives due to calibration
        for anchor in affected_iot:
            message = anchor.process_earthquake(
                noise_data,
                is_earthquake=False,
                timestamp=timestamp
            )
            if message:
                if not self._simulate_packet_loss():
                    triggers.append(message)
                    self.total_triggers_sent += 1
        
        return triggers
    
    def _simulate_packet_loss(self) -> bool:
        """
        Simulate network packet loss.
        
        Returns:
            True if packet is lost, False otherwise
        """
        if self.packet_loss_rate <= 0:
            return False
        return np.random.random() < self.packet_loss_rate
    
    def set_packet_loss_rate(self, rate: float):
        """Set network packet loss rate (0.0 to 1.0)"""
        self.packet_loss_rate = max(0.0, min(1.0, rate))
    
    def get_all_device_locations(self) -> Dict[str, List[Dict]]:
        """
        Get locations of all devices for visualization.
        
        Returns:
            Dictionary with 'mobile' and 'iot' lists of location dicts
        """
        return {
            'mobile': [
                {
                    'device_id': d.device_id,
                    'lat': d.latitude,
                    'lon': d.longitude,
                    'type': 'mobile'
                }
                for d in self.mobile_devices
            ],
            'iot': [
                {
                    'device_id': a.device_id,
                    'station_name': a.station_name,
                    'lat': a.latitude,
                    'lon': a.longitude,
                    'type': 'iot_anchor'
                }
                for a in self.iot_anchors
            ]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get device manager statistics"""
        return {
            'num_mobile_devices': len(self.mobile_devices),
            'num_iot_anchors': len(self.iot_anchors),
            'total_devices': len(self.mobile_devices) + len(self.iot_anchors),
            'packet_loss_rate': self.packet_loss_rate,
            'total_triggers_sent': self.total_triggers_sent,
            'total_triggers_lost': self.total_triggers_lost,
            'effective_delivery_rate': (
                self.total_triggers_sent / (self.total_triggers_sent + self.total_triggers_lost)
                if (self.total_triggers_sent + self.total_triggers_lost) > 0 else 1.0
            )
        }
    
    def reset_statistics(self):
        """Reset trigger statistics"""
        self.total_triggers_sent = 0
        self.total_triggers_lost = 0
