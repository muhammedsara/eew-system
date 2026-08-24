"""
Configuration Management for Hybrid Mobile-IoT EEW System

Central configuration for all system parameters as defined in the research paper:
"Reducing False Alarms in Crowdsensed Earthquake Early Warning via 
Weighted Spatiotemporal Consensus of Hybrid Mobile-IoT Edge Intelligence"

Author: Muhammed Şara
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import os


class DeviceType(Enum):
    """Device type enumeration"""
    MOBILE = "mobile"
    IOT_ANCHOR = "iot_anchor"


@dataclass
class ModelConfig:
    """Edge AI Model Configuration (1D CNN)
    
    Target: 188 KB TFLite model (quantized)
    Input: 150 timesteps × 3 channels (accel_xyz)
    """
    input_timesteps: int = 150
    input_channels: int = 3  # accel_x, accel_y, accel_z
    sampling_rate: int = 50  # Hz
    window_duration: float = 3.0  # seconds
    
    # CNN Architecture
    conv_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [5, 5, 3])  # Per-block kernel sizes (paper Section 5.1)
    pool_size: int = 2
    dropout_rate: float = 0.4  # Paper Section 5.1
    
    # Output
    num_classes: int = 2  # earthquake, non-earthquake
    
    # Training
    batch_size: int = 128  # Paper Section 5.2
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    
    # Paths - extensible for future datasets (MyShake, etc.)
    model_save_path: str = "models/earthquake_detector.h5"
    tflite_path: str = "models/earthquake_detector.tflite"
    
    # Dataset configuration (extensible)
    supported_datasets: List[str] = field(default_factory=lambda: [
        "stead",      # Stanford Earthquake Dataset
        "myshake",    # UC Berkeley MyShake (future)
        "instance",   # INSTANCE dataset (future)
        "custom"      # Custom dataset support
    ])


@dataclass
class PreprocessingConfig:
    """Data Preprocessing Configuration (Model v1.0 Pipeline)"""
    sampling_rate: int = 50
    window_duration: float = 3.0
    
    # Normalization
    normalize: bool = True
    norm_type: str = 'zscore'  # 'zscore', 'minmax', 'none'
    
    # Filtering
    apply_filter: bool = True
    filter_type: str = 'bandpass'
    lowcut: float = 0.5
    highcut: float = 20.0
    filter_order: int = 4


@dataclass
class STALTAConfig:
    """STA/LTA Trigger Configuration
    
    Short-Term Average / Long-Term Average for battery-efficient detection
    """
    sta_window: float = 0.5  # seconds
    lta_window: float = 10.0  # seconds
    trigger_ratio: float = 3.0  # STA/LTA ratio threshold
    pga_threshold: float = 0.05  # g (gravity units)
    detrigger_ratio: float = 1.5


@dataclass
class ConsensusConfig:
    """Weighted Spatiotemporal Consensus Configuration
    
    Core innovation: reliability-weighted voting for hybrid sensor fusion
    """
    # Reliability Weights (Paper Section III.C)
    mobile_weight: float = 0.3  # Lower weight due to noisy environment
    iot_weight: float = 0.7    # Higher weight due to calibrated sensors
    
    # Spatial Clustering (DBSCAN)
    spatial_eps_km: float = 5.0  # epsilon radius in kilometers
    min_samples: int = 3         # minimum devices for cluster
    
    # Temporal Windowing
    temporal_window_seconds: float = 2.0  # sliding window duration
    
    # Adaptive Thresholding (Paper Equation 2)
    threshold_night: float = 0.75   # 2:00-6:00 (low ambient noise)
    threshold_day: float = 0.90     # 8:00-18:00 (high traffic)
    threshold_default: float = 0.85  # Other hours
    
    # Validation Rules (Paper Equation 3)
    min_iot_triggers: int = 1    # At least 1 IoT anchor required
    min_total_triggers: int = 3  # At least 3 total devices
    
    # Weight optimization grid (for ablation studies)
    weight_grid: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5)
    ])


@dataclass
class DeviceConfig:
    """Device Simulation Configuration"""
    
    # Mobile Device Parameters
    mobile_accuracy: float = 0.914  # 91.4±2.1% from shake-table tests (paper Section 4.3)
    mobile_inference_time_ms: float = 200  # <200ms on Pixel-class
    mobile_sampling_rate: int = 50  # Hz
    
    # IoT Anchor Parameters
    iot_accuracy: float = 0.971  # 97.1±1.2% from shake-table tests (paper Section 4.3)
    iot_inference_time_ms: float = 150  # Raspberry Pi 4
    iot_sampling_rate: int = 100  # Hz (higher for precision)
    
    # GPS Configuration
    gps_precision_meters: float = 100  # Coarse-grained for privacy
    
    # Network
    mqtt_qos: int = 1
    message_size_bytes: int = 87  # Privacy-preserving payload


@dataclass
class SimulationConfig:
    """Trace-Driven Simulation Configuration"""
    
    # Earthquake Scenarios
    num_earthquake_scenarios: int = 100
    magnitude_range: Tuple[float, float] = (4.5, 7.5)
    
    # Device Distribution
    devices_per_magnitude_factor: int = 50  # µ = 50 × M
    iot_anchors_count: int = 10
    iot_grid_spacing_km: float = 10.0
    
    # False Positive Scenarios
    false_positive_truck: int = 200
    false_positive_metro: int = 200
    false_positive_construction: int = 100
    
    # Network Resilience
    packet_loss_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])
    
    # Random Seed
    random_seed: int = 42


@dataclass
class NetworkConfig:
    """Network Configuration"""
    
    # MQTT Settings
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic_triggers: str = "eew/triggers"
    mqtt_topic_alerts: str = "eew/alerts"
    mqtt_topic_status: str = "eew/status"
    
    # Latency Targets (Paper Section IV.G)
    target_edge_ai_ms: float = 200
    target_network_ms: float = 310
    target_consensus_ms: float = 520
    target_alert_ms: float = 1840
    target_total_ms: float = 3000


@dataclass  
class AlertConfig:
    """Alert Dissemination Configuration"""
    
    # Alert Channels
    enable_sms: bool = True
    enable_push: bool = True
    enable_dashboard: bool = True
    
    # SMS Gateway (mock for simulation)
    sms_gateway_url: str = ""
    
    # Alert Message Template
    alert_template: str = (
        "🚨 DEPREM UYARISI!\n"
        "Büyüklük: M{magnitude:.1f}\n"
        "Konum: {lat:.2f}°N, {lon:.2f}°E\n"
        "Zaman: {time}\n"
        "Güvenli bir alana geçin!"
    )


@dataclass
class DashboardConfig:
    """Web Dashboard Configuration"""
    
    host: str = "0.0.0.0"
    port: int = 8080  # Changed from 5000 to avoid macOS AirPlay Receiver conflict
    debug: bool = True
    
    # Map Settings
    default_center_lat: float = 40.5  # Turkey
    default_center_lon: float = 30.2
    default_zoom: int = 8
    
    # Update Intervals (ms)
    device_update_interval: int = 1000
    metrics_update_interval: int = 2000


@dataclass
class PathConfig:
    """File Path Configuration"""
    
    # Base directories
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    @property
    def data_dir(self) -> str:
        return os.path.join(self.base_dir, "data")
    
    @property
    def models_dir(self) -> str:
        return os.path.join(self.base_dir, "models")
    
    @property
    def results_dir(self) -> str:
        return os.path.join(self.base_dir, "results")
    
    @property
    def logs_dir(self) -> str:
        return os.path.join(self.base_dir, "logs")


@dataclass
class Config:
    """Master Configuration"""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    sta_lta: STALTAConfig = field(default_factory=STALTAConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    def validate(self) -> bool:
        """Validate configuration consistency"""
        # Check weight sum
        assert abs(self.consensus.mobile_weight + self.consensus.iot_weight - 1.0) < 0.01, \
            "Weights must sum to 1.0"
        
        # Check thresholds
        assert 0 < self.consensus.threshold_night <= 1.0
        assert 0 < self.consensus.threshold_day <= 1.0
        assert 0 < self.consensus.threshold_default <= 1.0
        
        # Check model dimensions
        expected_timesteps = int(self.model.window_duration * self.model.sampling_rate)
        assert self.model.input_timesteps == expected_timesteps, \
            f"Input timesteps ({self.model.input_timesteps}) should match window_duration * sampling_rate ({expected_timesteps})"
        
        return True


# Global configuration instance
config = Config()

# Ensure configuration is valid
config.validate()
