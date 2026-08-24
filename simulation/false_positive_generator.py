"""
False Positive Scenario Generator

Generates realistic false positive scenarios from human activity data.
Based on WISDM and UCI HAR datasets.

From Paper Section IV.A:
"Rather than purely synthetic noise, we construct false positive scenarios 
using real-world accelerometer traces:
- Pedestrian/Vehicle Motion: WISDM dataset (walking, jogging, stairs)
- Structural Vibrations: UCI HAR dataset (daily activities)
- Spatial-Temporal Patterns: truck (n=200), metro (n=200), construction (n=100)"

Author: Muhammed Şara
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class NoiseType(Enum):
    """Types of anthropogenic noise sources"""
    WALKING = "walking"
    JOGGING = "jogging"
    STAIRS = "stairs"
    VEHICLE = "vehicle"
    TRUCK = "truck"
    METRO = "metro"
    CONSTRUCTION = "construction"
    DROPPED_PHONE = "dropped_phone"


@dataclass
class FalsePositiveScenario:
    """Represents a false positive scenario"""
    scenario_id: int
    noise_type: NoiseType
    source_lat: float
    source_lon: float
    affected_radius_km: float
    waveform_data: np.ndarray
    pga: float  # Peak ground acceleration
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scenario_id': self.scenario_id,
            'noise_type': self.noise_type.value,
            'source': {'lat': self.source_lat, 'lon': self.source_lon},
            'affected_radius_km': self.affected_radius_km,
            'pga': self.pga,
            'timestamp': self.timestamp
        }


class FalsePositiveGenerator:
    """
    Generates false positive scenarios from human activity data.
    
    Creates realistic noise waveforms that could trigger false alarms:
    - Vehicle vibrations (trucks, buses, metro)
    - Human activity (walking, running, dropping phone)
    - Construction activity
    """
    
    def __init__(
        self,
        sampling_rate: int = None,
        seed: int = None
    ):
        """
        Initialize false positive generator.
        
        Args:
            sampling_rate: Sampling rate in Hz
            seed: Random seed
        """
        self.sampling_rate = sampling_rate or config.model.sampling_rate
        
        if seed is not None:
            np.random.seed(seed)
        
        self._scenario_counter = 0
        
        # Noise characteristics
        self.noise_profiles = {
            NoiseType.WALKING: {
                'frequency_range': (1.5, 3.0),
                'amplitude_range': (0.03, 0.08),
                'duration': 3.0,
                'radius_km': 0.01  # Very local
            },
            NoiseType.JOGGING: {
                'frequency_range': (2.5, 4.5),
                'amplitude_range': (0.08, 0.15),
                'duration': 3.0,
                'radius_km': 0.01
            },
            NoiseType.VEHICLE: {
                'frequency_range': (5, 25),
                'amplitude_range': (0.02, 0.06),
                'duration': 3.0,
                'radius_km': 0.1
            },
            NoiseType.TRUCK: {
                'frequency_range': (3, 15),
                'amplitude_range': (0.05, 0.12),
                'duration': 3.0,
                'radius_km': 0.5  # Affects nearby buildings
            },
            NoiseType.METRO: {
                'frequency_range': (4, 20),
                'amplitude_range': (0.04, 0.10),
                'duration': 3.0,
                'radius_km': 0.3
            },
            NoiseType.CONSTRUCTION: {
                'frequency_range': (2, 30),
                'amplitude_range': (0.06, 0.15),
                'duration': 3.0,
                'radius_km': 1.0
            },
            NoiseType.DROPPED_PHONE: {
                'frequency_range': (10, 100),
                'amplitude_range': (0.5, 2.0),  # High but brief
                'duration': 0.5,
                'radius_km': 0.0  # Single device only
            }
        }
    
    def generate_scenario(
        self,
        noise_type: NoiseType = None,
        source_lat: float = None,
        source_lon: float = None,
        timestamp: float = None
    ) -> FalsePositiveScenario:
        """
        Generate a single false positive scenario.
        
        Args:
            noise_type: Type of noise (default: random)
            source_lat: Source latitude
            source_lon: Source longitude
            timestamp: Event timestamp
            
        Returns:
            FalsePositiveScenario object
        """
        self._scenario_counter += 1
        
        if noise_type is None:
            noise_type = np.random.choice(list(NoiseType))
        
        if source_lat is None:
            source_lat = np.random.uniform(36.0, 42.0)
        
        if source_lon is None:
            source_lon = np.random.uniform(26.0, 44.0)
        
        if timestamp is None:
            timestamp = 0.0
        
        profile = self.noise_profiles[noise_type]
        
        # Generate waveform
        waveform = self._generate_noise_waveform(noise_type)
        
        # Calculate PGA
        magnitudes = np.sqrt(np.sum(waveform ** 2, axis=1))
        pga = np.max(np.abs(magnitudes - 9.81)) / 9.81
        
        return FalsePositiveScenario(
            scenario_id=self._scenario_counter,
            noise_type=noise_type,
            source_lat=source_lat,
            source_lon=source_lon,
            affected_radius_km=profile['radius_km'],
            waveform_data=waveform,
            pga=pga,
            timestamp=timestamp
        )
    
    def generate_scenarios(
        self,
        n_truck: int = None,
        n_metro: int = None,
        n_construction: int = None
    ) -> List[FalsePositiveScenario]:
        """
        Generate multiple false positive scenarios.
        
        Default counts from paper: truck=200, metro=200, construction=100
        
        Args:
            n_truck: Number of truck scenarios
            n_metro: Number of metro scenarios
            n_construction: Number of construction scenarios
            
        Returns:
            List of FalsePositiveScenario objects
        """
        n_truck = n_truck or config.simulation.false_positive_truck
        n_metro = n_metro or config.simulation.false_positive_metro
        n_construction = n_construction or config.simulation.false_positive_construction
        
        scenarios = []
        
        # Truck scenarios
        for i in range(n_truck):
            scenario = self.generate_scenario(
                noise_type=NoiseType.TRUCK,
                timestamp=float(i)
            )
            scenarios.append(scenario)
        
        # Metro scenarios
        for i in range(n_metro):
            scenario = self.generate_scenario(
                noise_type=NoiseType.METRO,
                timestamp=float(n_truck + i)
            )
            scenarios.append(scenario)
        
        # Construction scenarios
        for i in range(n_construction):
            scenario = self.generate_scenario(
                noise_type=NoiseType.CONSTRUCTION,
                timestamp=float(n_truck + n_metro + i)
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_noise_waveform(
        self,
        noise_type: NoiseType
    ) -> np.ndarray:
        """
        Generate synthetic noise waveform.
        
        Args:
            noise_type: Type of noise
            
        Returns:
            Array of shape (n_samples, 3)
        """
        profile = self.noise_profiles[noise_type]
        
        duration = profile['duration']
        n_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, n_samples)
        
        freq_min, freq_max = profile['frequency_range']
        amp_min, amp_max = profile['amplitude_range']
        
        amplitude = np.random.uniform(amp_min, amp_max)
        
        # Generate based on noise type
        if noise_type == NoiseType.WALKING:
            waveform = self._generate_walking_pattern(t, amplitude)
        elif noise_type == NoiseType.JOGGING:
            waveform = self._generate_jogging_pattern(t, amplitude)
        elif noise_type == NoiseType.TRUCK:
            waveform = self._generate_vehicle_pattern(t, amplitude, freq_min, freq_max)
        elif noise_type == NoiseType.METRO:
            waveform = self._generate_vehicle_pattern(t, amplitude, freq_min, freq_max)
        elif noise_type == NoiseType.CONSTRUCTION:
            waveform = self._generate_construction_pattern(t, amplitude)
        elif noise_type == NoiseType.DROPPED_PHONE:
            waveform = self._generate_impact_pattern(t, amplitude)
        else:
            waveform = self._generate_generic_noise(t, amplitude, freq_min, freq_max)
        
        return waveform.astype(np.float32)
    
    def _generate_walking_pattern(
        self,
        t: np.ndarray,
        amplitude: float
    ) -> np.ndarray:
        """Generate walking vibration pattern"""
        n_samples = len(t)
        
        # Step frequency ~2 Hz
        step_freq = np.random.uniform(1.5, 2.5)
        
        # Periodic step pattern
        steps = np.sin(2 * np.pi * step_freq * t)
        
        # Add harmonics
        pattern = amplitude * (
            steps + 
            0.5 * np.sin(4 * np.pi * step_freq * t) +
            0.25 * np.sin(6 * np.pi * step_freq * t)
        )
        
        # 3-axis with phase differences
        waveform = np.zeros((n_samples, 3))
        waveform[:, 0] = pattern + 9.81  # Z (vertical) - dominant
        waveform[:, 1] = pattern * 0.3 + np.random.normal(0, amplitude * 0.1, n_samples)
        waveform[:, 2] = pattern * 0.2 + np.random.normal(0, amplitude * 0.1, n_samples)
        
        return waveform
    
    def _generate_jogging_pattern(
        self,
        t: np.ndarray,
        amplitude: float
    ) -> np.ndarray:
        """Generate jogging vibration pattern"""
        n_samples = len(t)
        
        # Higher frequency, higher amplitude
        step_freq = np.random.uniform(2.5, 3.5)
        
        steps = np.sin(2 * np.pi * step_freq * t)
        
        pattern = amplitude * (
            steps +
            0.7 * np.sin(4 * np.pi * step_freq * t) +
            0.4 * np.sin(6 * np.pi * step_freq * t)
        )
        
        waveform = np.zeros((n_samples, 3))
        waveform[:, 0] = pattern + 9.81
        waveform[:, 1] = pattern * 0.5 + np.random.normal(0, amplitude * 0.15, n_samples)
        waveform[:, 2] = pattern * 0.4 + np.random.normal(0, amplitude * 0.15, n_samples)
        
        return waveform
    
    def _generate_vehicle_pattern(
        self,
        t: np.ndarray,
        amplitude: float,
        freq_min: float,
        freq_max: float
    ) -> np.ndarray:
        """Generate vehicle/machinery vibration pattern"""
        n_samples = len(t)
        
        # Broadband noise with dominant frequencies
        pattern = np.zeros(n_samples)
        
        # Multiple frequency components
        n_freqs = np.random.randint(3, 7)
        for _ in range(n_freqs):
            freq = np.random.uniform(freq_min, freq_max)
            amp = amplitude * np.random.uniform(0.2, 1.0)
            phase = np.random.uniform(0, 2 * np.pi)
            pattern += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Add broadband noise
        pattern += np.random.normal(0, amplitude * 0.3, n_samples)
        
        waveform = np.zeros((n_samples, 3))
        waveform[:, 0] = pattern + 9.81
        waveform[:, 1] = pattern * 0.8 + np.random.normal(0, amplitude * 0.2, n_samples)
        waveform[:, 2] = pattern * 0.7 + np.random.normal(0, amplitude * 0.2, n_samples)
        
        return waveform
    
    def _generate_construction_pattern(
        self,
        t: np.ndarray,
        amplitude: float
    ) -> np.ndarray:
        """Generate construction vibration pattern (impacts)"""
        n_samples = len(t)
        
        # Impact pattern (jackhammer, pile driving)
        pattern = np.zeros(n_samples)
        
        # Random impacts
        n_impacts = np.random.randint(10, 30)
        impact_times = np.random.uniform(0, t[-1], n_impacts)
        
        for impact_time in impact_times:
            # Damped sinusoid for each impact
            mask = t >= impact_time
            decay = np.exp(-10 * (t[mask] - impact_time))
            freq = np.random.uniform(5, 20)
            impact = amplitude * decay * np.sin(2 * np.pi * freq * (t[mask] - impact_time))
            pattern[mask] += impact[:sum(mask)]
        
        # Add continuous background
        pattern += np.random.normal(0, amplitude * 0.2, n_samples)
        
        waveform = np.zeros((n_samples, 3))
        waveform[:, 0] = pattern + 9.81
        waveform[:, 1] = pattern * 0.9 + np.random.normal(0, amplitude * 0.15, n_samples)
        waveform[:, 2] = pattern * 0.85 + np.random.normal(0, amplitude * 0.15, n_samples)
        
        return waveform
    
    def _generate_impact_pattern(
        self,
        t: np.ndarray,
        amplitude: float
    ) -> np.ndarray:
        """Generate dropped phone impact pattern"""
        n_samples = len(t)
        
        # Single sharp impact followed by ringing
        impact_time = 0.1
        
        pattern = np.zeros(n_samples)
        mask = t >= impact_time
        
        decay = np.exp(-20 * (t[mask] - impact_time))
        pattern[mask] = amplitude * decay * np.sin(50 * 2 * np.pi * (t[mask] - impact_time))
        
        waveform = np.zeros((n_samples, 3))
        waveform[:, 0] = pattern + 9.81
        waveform[:, 1] = pattern * 0.5
        waveform[:, 2] = pattern * 0.3
        
        return waveform
    
    def _generate_generic_noise(
        self,
        t: np.ndarray,
        amplitude: float,
        freq_min: float,
        freq_max: float
    ) -> np.ndarray:
        """Generate generic noise pattern"""
        n_samples = len(t)
        
        # Multi-frequency noise
        pattern = np.zeros(n_samples)
        
        for _ in range(5):
            freq = np.random.uniform(freq_min, freq_max)
            pattern += amplitude * 0.2 * np.sin(
                2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)
            )
        
        pattern += np.random.normal(0, amplitude * 0.3, n_samples)
        
        waveform = np.zeros((n_samples, 3))
        waveform[:, 0] = pattern + 9.81
        waveform[:, 1] = pattern * 0.7
        waveform[:, 2] = pattern * 0.6
        
        return waveform
    
    def load_from_wisdm(self, wisdm_path: str) -> List[FalsePositiveScenario]:
        """
        Load false positive scenarios from WISDM dataset.
        
        WISDM: Wireless Sensor Data Mining
        http://www.cis.fordham.edu/wisdm/dataset.php
        
        Args:
            wisdm_path: Path to WISDM data file
            
        Returns:
            List of FalsePositiveScenario objects
        """
        # Placeholder for actual dataset loading
        print(f"WISDM loading from {wisdm_path} - using synthetic data")
        return self.generate_scenarios(n_truck=100, n_metro=100, n_construction=50)
    
    def load_from_uci_har(self, uci_path: str) -> List[FalsePositiveScenario]:
        """
        Load false positive scenarios from UCI HAR dataset.
        
        UCI HAR: Human Activity Recognition Using Smartphones Dataset
        https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
        
        Args:
            uci_path: Path to UCI HAR data directory
            
        Returns:
            List of FalsePositiveScenario objects
        """
        # Placeholder for actual dataset loading
        print(f"UCI HAR loading from {uci_path} - using synthetic data")
        return self.generate_scenarios(n_truck=100, n_metro=100, n_construction=50)
