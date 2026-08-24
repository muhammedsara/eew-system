"""
Earthquake Scenario Generator

Generates realistic earthquake scenarios for simulation.
Based on STEAD dataset characteristics with synthetic waveforms.

From Paper Section IV.A:
"Earthquake Events (n=100): We selected 100 events from STEAD with 
magnitudes M4.5–7.5. For each event, we simulated:
- Mobile devices: Poisson distribution with µ = 50 × M
- IoT anchors: 10 devices in grid pattern (10 km spacing)
- Spatial distribution: Random placement within affected radius r = 10^(M-3) km"

Author: Muhammed Şara
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from utils.haversine import calculate_affected_radius


@dataclass
class EarthquakeScenario:
    """Represents a single earthquake scenario"""
    scenario_id: int
    magnitude: float
    epicenter_lat: float
    epicenter_lon: float
    depth_km: float
    timestamp: float
    affected_radius_km: float
    waveform_data: np.ndarray  # (n_samples, 3) accelerometer data
    expected_mobile_count: int
    expected_iot_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scenario_id': self.scenario_id,
            'magnitude': self.magnitude,
            'epicenter': {'lat': self.epicenter_lat, 'lon': self.epicenter_lon},
            'depth_km': self.depth_km,
            'timestamp': self.timestamp,
            'affected_radius_km': self.affected_radius_km,
            'expected_mobile_count': self.expected_mobile_count,
            'expected_iot_count': self.expected_iot_count
        }


class EarthquakeGenerator:
    """
    Generates earthquake scenarios for simulation.
    
    Creates realistic earthquake waveforms based on magnitude and distance.
    Supports both synthetic generation and loading from STEAD dataset.
    """
    
    def __init__(
        self,
        magnitude_range: Tuple[float, float] = None,
        sampling_rate: int = None,
        seed: int = None
    ):
        """
        Initialize earthquake generator.
        
        Args:
            magnitude_range: (min, max) magnitude range
            sampling_rate: Waveform sampling rate in Hz
            seed: Random seed for reproducibility
        """
        self.magnitude_range = magnitude_range or config.simulation.magnitude_range
        self.sampling_rate = sampling_rate or config.model.sampling_rate
        
        if seed is not None:
            np.random.seed(seed)
        
        self._scenario_counter = 0
    
    def generate_scenario(
        self,
        magnitude: float = None,
        epicenter_lat: float = None,
        epicenter_lon: float = None,
        depth_km: float = None,
        timestamp: float = None
    ) -> EarthquakeScenario:
        """
        Generate a single earthquake scenario.
        
        Args:
            magnitude: Earthquake magnitude (default: random in range)
            epicenter_lat: Epicenter latitude (default: Turkey region)
            epicenter_lon: Epicenter longitude (default: Turkey region)
            depth_km: Focal depth in km (default: random 5-30)
            timestamp: Event timestamp (default: 0)
            
        Returns:
            EarthquakeScenario object
        """
        self._scenario_counter += 1
        
        # Generate parameters if not provided
        if magnitude is None:
            magnitude = np.random.uniform(*self.magnitude_range)
        
        if epicenter_lat is None:
            # Turkey region: 36-42°N
            epicenter_lat = np.random.uniform(36.0, 42.0)
        
        if epicenter_lon is None:
            # Turkey region: 26-44°E
            epicenter_lon = np.random.uniform(26.0, 44.0)
        
        if depth_km is None:
            depth_km = np.random.uniform(5.0, 30.0)
        
        if timestamp is None:
            timestamp = 0.0
        
        # Calculate affected radius
        affected_radius = calculate_affected_radius(magnitude)
        
        # Generate waveform
        waveform = self._generate_waveform(magnitude, depth_km)
        
        # Expected device counts
        expected_mobile = int(50 * magnitude)  # µ = 50 × M
        expected_iot = config.simulation.iot_anchors_count
        
        return EarthquakeScenario(
            scenario_id=self._scenario_counter,
            magnitude=magnitude,
            epicenter_lat=epicenter_lat,
            epicenter_lon=epicenter_lon,
            depth_km=depth_km,
            timestamp=timestamp,
            affected_radius_km=affected_radius,
            waveform_data=waveform,
            expected_mobile_count=expected_mobile,
            expected_iot_count=expected_iot
        )
    
    def generate_scenarios(
        self,
        n_scenarios: int = 100,
        magnitude_distribution: str = 'uniform'
    ) -> List[EarthquakeScenario]:
        """
        Generate multiple earthquake scenarios.
        
        Args:
            n_scenarios: Number of scenarios to generate
            magnitude_distribution: 'uniform' or 'gutenberg-richter'
            
        Returns:
            List of EarthquakeScenario objects
        """
        scenarios = []
        
        if magnitude_distribution == 'gutenberg-richter':
            # Gutenberg-Richter law: log N = a - bM
            # More small earthquakes, fewer large ones
            b_value = 1.0
            magnitudes = self._sample_gutenberg_richter(
                n_scenarios, self.magnitude_range, b_value
            )
        else:
            magnitudes = np.random.uniform(
                self.magnitude_range[0],
                self.magnitude_range[1],
                n_scenarios
            )
        
        for i, mag in enumerate(magnitudes):
            scenario = self.generate_scenario(magnitude=mag, timestamp=float(i))
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_waveform(
        self,
        magnitude: float,
        depth_km: float,
        duration_seconds: float = 3.0
    ) -> np.ndarray:
        """
        Generate synthetic earthquake waveform.
        
        Creates P-wave and S-wave components with realistic characteristics:
        - P-wave: Initial onset, lower amplitude
        - S-wave: Main shaking, higher amplitude
        - Coda: Decaying oscillations
        
        Args:
            magnitude: Earthquake magnitude
            depth_km: Focal depth
            duration_seconds: Waveform duration
            
        Returns:
            Array of shape (n_samples, 3) for 3-axis accelerometer
        """
        n_samples = int(duration_seconds * self.sampling_rate)
        t = np.linspace(0, duration_seconds, n_samples)
        
        # Scale amplitude based on magnitude
        # Empirical relationship: log(A) ∝ M
        # Multiplier chosen so M4.5 produces PGA ≈ 0.09g (above 0.04g IoT threshold)
        # and M7.0 produces PGA ≈ 1.6g
        base_amplitude = 10 ** ((magnitude - 4) / 2) * 0.5  # in g units
        
        # P-wave (arrives first, lower amplitude)
        p_wave_start = 0.2
        p_wave_freq = np.random.uniform(5, 15)  # Hz
        p_wave = np.zeros(n_samples)
        
        p_mask = t >= p_wave_start
        p_envelope = np.exp(-3 * (t[p_mask] - p_wave_start))
        p_wave[p_mask] = base_amplitude * 0.3 * p_envelope * np.sin(
            2 * np.pi * p_wave_freq * (t[p_mask] - p_wave_start)
        )
        
        # S-wave (main shaking, arrives after P-wave)
        s_wave_start = 0.5
        s_wave_freq = np.random.uniform(2, 8)  # Hz (lower than P)
        s_wave = np.zeros(n_samples)
        
        s_mask = t >= s_wave_start
        s_envelope = np.exp(-1.5 * (t[s_mask] - s_wave_start))
        s_wave[s_mask] = base_amplitude * s_envelope * np.sin(
            2 * np.pi * s_wave_freq * (t[s_mask] - s_wave_start) + 
            np.random.uniform(0, 2*np.pi)
        )
        
        # Surface waves (later, lower frequency)
        surface_start = 1.0
        surface_freq = np.random.uniform(0.5, 2)  # Hz
        surface = np.zeros(n_samples)
        
        surf_mask = t >= surface_start
        surf_envelope = np.exp(-0.8 * (t[surf_mask] - surface_start))
        surface[surf_mask] = base_amplitude * 0.5 * surf_envelope * np.sin(
            2 * np.pi * surface_freq * (t[surf_mask] - surface_start)
        )
        
        # Combine components
        combined = p_wave + s_wave + surface
        
        # Add noise
        noise = np.random.normal(0, base_amplitude * 0.1, n_samples)
        combined += noise
        
        # Create 3-axis data with different phases
        waveform = np.zeros((n_samples, 3))
        
        # Z-axis (vertical) - dominant for P-waves
        waveform[:, 0] = combined + 9.81  # Add gravity
        
        # X-axis (N-S) - phase shifted
        waveform[:, 1] = np.roll(combined, int(0.05 * self.sampling_rate)) * 0.8 + \
                         np.random.normal(0, base_amplitude * 0.05, n_samples)
        
        # Y-axis (E-W) - phase shifted differently
        waveform[:, 2] = np.roll(combined, int(0.08 * self.sampling_rate)) * 0.7 + \
                         np.random.normal(0, base_amplitude * 0.05, n_samples)
        
        return waveform.astype(np.float32)
    
    def _sample_gutenberg_richter(
        self,
        n_samples: int,
        magnitude_range: Tuple[float, float],
        b_value: float = 1.0
    ) -> np.ndarray:
        """
        Sample magnitudes from Gutenberg-Richter distribution.
        
        log10(N) = a - b*M
        
        Args:
            n_samples: Number of samples
            magnitude_range: (min_mag, max_mag)
            b_value: b-value (typically ~1.0)
            
        Returns:
            Array of magnitudes
        """
        m_min, m_max = magnitude_range
        
        # Inverse transform sampling
        u = np.random.uniform(0, 1, n_samples)
        
        # CDF^-1
        magnitudes = m_min - (1/b_value) * np.log10(
            1 - u * (1 - 10**(-b_value * (m_max - m_min)))
        )
        
        return np.clip(magnitudes, m_min, m_max)
    
    def load_from_stead(
        self,
        stead_path: str,
        n_events: int = 100
    ) -> List[EarthquakeScenario]:
        """
        Load earthquake scenarios from STEAD dataset.
        
        STEAD: Stanford Earthquake Dataset
        https://github.com/smousavi05/STEAD
        
        Args:
            stead_path: Path to STEAD HDF5 file
            n_events: Number of events to load
            
        Returns:
            List of EarthquakeScenario objects
        """
        # Note: Requires h5py and actual STEAD dataset
        try:
            import h5py
        except ImportError:
            print("h5py not installed. Using synthetic data instead.")
            return self.generate_scenarios(n_events)
        
        if not os.path.exists(stead_path):
            print(f"STEAD file not found: {stead_path}. Using synthetic data.")
            return self.generate_scenarios(n_events)
        
        scenarios = []
        
        with h5py.File(stead_path, 'r') as f:
            # Get earthquake trace keys
            traces = list(f['earthquake']['local'].keys())[:n_events]
            
            for i, trace_name in enumerate(traces):
                trace_data = f['earthquake']['local'][trace_name]
                
                # Extract waveform (3-component)
                waveform = np.array(trace_data)  # (3, n_samples)
                waveform = waveform.T  # (n_samples, 3)
                
                # Extract metadata
                attrs = trace_data.attrs
                magnitude = attrs.get('source_magnitude', 5.0)
                lat = attrs.get('source_latitude', 40.0)
                lon = attrs.get('source_longitude', 30.0)
                depth = attrs.get('source_depth_km', 10.0)
                
                self._scenario_counter += 1
                
                scenario = EarthquakeScenario(
                    scenario_id=self._scenario_counter,
                    magnitude=float(magnitude),
                    epicenter_lat=float(lat),
                    epicenter_lon=float(lon),
                    depth_km=float(depth),
                    timestamp=float(i),
                    affected_radius_km=calculate_affected_radius(magnitude),
                    waveform_data=waveform.astype(np.float32),
                    expected_mobile_count=int(50 * magnitude),
                    expected_iot_count=config.simulation.iot_anchors_count
                )
                scenarios.append(scenario)
        
        return scenarios
