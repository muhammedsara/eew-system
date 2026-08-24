"""
Trace-Driven Simulation Framework

Main simulation orchestrator that runs earthquake and false positive
scenarios through the complete system pipeline.

Implements the experimental evaluation from Paper Section IV.

Author: Muhammed Şara
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import time
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from simulation.earthquake_generator import EarthquakeGenerator, EarthquakeScenario
from simulation.false_positive_generator import FalsePositiveGenerator, FalsePositiveScenario
from devices.device_manager import DeviceManager
from consensus.engine import ConsensusEngine, ConsensusDecision


@dataclass
class SimulationResult:
    """Result of a single simulation run"""
    scenario_id: int
    is_earthquake: bool  # Ground truth
    detected: bool       # System decision
    score: float
    threshold: float
    mobile_count: int
    iot_count: int
    processing_time_ms: float
    timestamp: float
    
    @property
    def is_true_positive(self) -> bool:
        return self.is_earthquake and self.detected
    
    @property
    def is_false_positive(self) -> bool:
        return not self.is_earthquake and self.detected
    
    @property
    def is_true_negative(self) -> bool:
        return not self.is_earthquake and not self.detected
    
    @property
    def is_false_negative(self) -> bool:
        return self.is_earthquake and not self.detected


@dataclass
class SimulationSummary:
    """Summary of simulation results"""
    total_scenarios: int
    earthquake_scenarios: int
    false_positive_scenarios: int
    
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Metrics
    precision: float
    recall: float  # TPR
    f1_score: float
    false_positive_rate: float
    accuracy: float
    
    # Timing
    avg_processing_time_ms: float
    total_simulation_time_s: float
    
    # Detailed results
    results: List[SimulationResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_scenarios': self.total_scenarios,
            'earthquake_scenarios': self.earthquake_scenarios,
            'false_positive_scenarios': self.false_positive_scenarios,
            'confusion_matrix': {
                'tp': self.true_positives,
                'fp': self.false_positives,
                'tn': self.true_negatives,
                'fn': self.false_negatives
            },
            'metrics': {
                'precision': round(self.precision, 4),
                'recall_tpr': round(self.recall, 4),
                'f1_score': round(self.f1_score, 4),
                'fpr': round(self.false_positive_rate, 4),
                'accuracy': round(self.accuracy, 4)
            },
            'timing': {
                'avg_processing_time_ms': round(self.avg_processing_time_ms, 2),
                'total_simulation_time_s': round(self.total_simulation_time_s, 2)
            }
        }


class TraceSimulator:
    """
    Trace-driven simulation framework.
    
    Runs earthquake and false positive scenarios through the
    complete detection pipeline and evaluates performance.
    """
    
    def __init__(
        self,
        mobile_weight: float = None,
        iot_weight: float = None,
        packet_loss_rate: float = 0.0,
        seed: int = None
    ):
        """
        Initialize simulator.
        
        Args:
            mobile_weight: Weight for mobile votes
            iot_weight: Weight for IoT votes
            packet_loss_rate: Network packet loss rate
            seed: Random seed
        """
        self.seed = seed or config.simulation.random_seed
        np.random.seed(self.seed)
        
        # Generators
        self.earthquake_generator = EarthquakeGenerator(seed=self.seed)
        self.fp_generator = FalsePositiveGenerator(seed=self.seed)
        
        # Device manager
        self.device_manager = DeviceManager()
        self.device_manager.set_packet_loss_rate(packet_loss_rate)
        
        # Consensus engine
        self.consensus_engine = ConsensusEngine(
            mobile_weight=mobile_weight,
            iot_weight=iot_weight
        )
        
        # Results storage
        self.results: List[SimulationResult] = []
    
    def run_simulation(
        self,
        n_earthquakes: int = 100,
        n_false_positives: int = 500,
        show_progress: bool = True
    ) -> SimulationSummary:
        """
        Run complete simulation.
        
        Args:
            n_earthquakes: Number of earthquake scenarios
            n_false_positives: Number of false positive scenarios
            show_progress: Show progress bar
            
        Returns:
            SimulationSummary with results
        """
        start_time = time.time()
        self.results = []
        
        # Generate scenarios
        earthquake_scenarios = self.earthquake_generator.generate_scenarios(n_earthquakes)
        fp_scenarios = self.fp_generator.generate_scenarios(
            n_truck=n_false_positives // 3,
            n_metro=n_false_positives // 3,
            n_construction=n_false_positives - 2 * (n_false_positives // 3)
        )
        
        # Run earthquake scenarios
        scenarios = earthquake_scenarios
        if show_progress:
            scenarios = tqdm(earthquake_scenarios, desc="Earthquake scenarios")
        
        for scenario in scenarios:
            result = self._run_earthquake_scenario(scenario)
            self.results.append(result)
        
        # Run false positive scenarios
        fp_list = fp_scenarios
        if show_progress:
            fp_list = tqdm(fp_scenarios, desc="False positive scenarios")
        
        for scenario in fp_list:
            result = self._run_fp_scenario(scenario)
            self.results.append(result)
        
        # Calculate summary
        total_time = time.time() - start_time
        summary = self._calculate_summary(
            n_earthquakes, len(fp_scenarios), total_time
        )
        
        return summary
    
    def _run_earthquake_scenario(
        self,
        scenario: EarthquakeScenario
    ) -> SimulationResult:
        """Run a single earthquake scenario"""
        # Cap distribution radius to urban area (~50km) for realistic clustering
        # Even large earthquakes only affect devices in a localized urban area
        distribution_radius = min(scenario.affected_radius_km, 50.0)
        
        # Setup device distribution
        self.device_manager.setup_scenario(
            center_lat=scenario.epicenter_lat,
            center_lon=scenario.epicenter_lon,
            num_mobile=scenario.expected_mobile_count,
            num_iot=scenario.expected_iot_count,
            distribution_radius_km=distribution_radius,
            seed=self.seed + scenario.scenario_id
        )
        
        # Simulate triggers
        triggers = self.device_manager.simulate_earthquake_triggers(
            earthquake_data=scenario.waveform_data,
            epicenter_lat=scenario.epicenter_lat,
            epicenter_lon=scenario.epicenter_lon,
            magnitude=scenario.magnitude,
            timestamp=scenario.timestamp
        )
        
        # Convert to trigger dicts
        trigger_dicts = [t.to_dict() for t in triggers]
        
        # Run consensus
        decision = self.consensus_engine.process(
            trigger_dicts,
            timestamp=scenario.timestamp,
            hour=np.random.randint(0, 24)  # Random hour
        )
        
        return SimulationResult(
            scenario_id=scenario.scenario_id,
            is_earthquake=True,
            detected=decision.is_earthquake,
            score=decision.score,
            threshold=decision.threshold,
            mobile_count=decision.mobile_count,
            iot_count=decision.iot_count,
            processing_time_ms=decision.processing_time_ms,
            timestamp=scenario.timestamp
        )
    
    def _run_fp_scenario(
        self,
        scenario: FalsePositiveScenario
    ) -> SimulationResult:
        """Run a single false positive scenario"""
        # Use smaller device distribution for false positives
        self.device_manager.setup_scenario(
            center_lat=scenario.source_lat,
            center_lon=scenario.source_lon,
            num_mobile=20,  # Fewer devices for local noise
            num_iot=3,
            distribution_radius_km=5.0,
            seed=self.seed + scenario.scenario_id + 10000
        )
        
        # Simulate false triggers
        triggers = self.device_manager.simulate_false_positive_triggers(
            noise_data=scenario.waveform_data,
            source_lat=scenario.source_lat,
            source_lon=scenario.source_lon,
            affected_radius_km=scenario.affected_radius_km,
            timestamp=scenario.timestamp
        )
        
        # Convert to trigger dicts
        trigger_dicts = [t.to_dict() for t in triggers]
        
        # Run consensus
        decision = self.consensus_engine.process(
            trigger_dicts,
            timestamp=scenario.timestamp,
            hour=np.random.randint(8, 18)  # Daytime (more strict)
        )
        
        return SimulationResult(
            scenario_id=scenario.scenario_id,
            is_earthquake=False,
            detected=decision.is_earthquake,
            score=decision.score,
            threshold=decision.threshold,
            mobile_count=decision.mobile_count,
            iot_count=decision.iot_count,
            processing_time_ms=decision.processing_time_ms,
            timestamp=scenario.timestamp
        )
    
    def _calculate_summary(
        self,
        n_earthquakes: int,
        n_fps: int,
        total_time: float
    ) -> SimulationSummary:
        """Calculate summary statistics"""
        tp = sum(1 for r in self.results if r.is_true_positive)
        fp = sum(1 for r in self.results if r.is_false_positive)
        tn = sum(1 for r in self.results if r.is_true_negative)
        fn = sum(1 for r in self.results if r.is_false_negative)
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        accuracy = (tp + tn) / len(self.results) if self.results else 0
        
        avg_time = np.mean([r.processing_time_ms for r in self.results]) if self.results else 0
        
        return SimulationSummary(
            total_scenarios=len(self.results),
            earthquake_scenarios=n_earthquakes,
            false_positive_scenarios=n_fps,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=fpr,
            accuracy=accuracy,
            avg_processing_time_ms=avg_time,
            total_simulation_time_s=total_time,
            results=self.results
        )
    
    def run_ablation_study(
        self,
        weight_pairs: List[Tuple[float, float]] = None,
        n_earthquakes: int = 100,
        n_false_positives: int = 500
    ) -> Dict[str, SimulationSummary]:
        """
        Run ablation study for weight optimization.
        
        Args:
            weight_pairs: List of (mobile_weight, iot_weight) pairs
            n_earthquakes: Earthquakes per configuration
            n_false_positives: False positives per configuration
            
        Returns:
            Dictionary mapping weight_pair to SimulationSummary
        """
        if weight_pairs is None:
            weight_pairs = config.consensus.weight_grid
        
        results = {}
        
        for mobile_weight, iot_weight in weight_pairs:
            print(f"\nTesting weights: mobile={mobile_weight}, IoT={iot_weight}")
            
            # Update weights
            self.consensus_engine.update_weights(mobile_weight, iot_weight)
            
            # Run simulation
            summary = self.run_simulation(
                n_earthquakes=n_earthquakes,
                n_false_positives=n_false_positives,
                show_progress=True
            )
            
            key = f"({mobile_weight}, {iot_weight})"
            results[key] = summary
            
            print(f"  TPR: {summary.recall:.2%}, FPR: {summary.false_positive_rate:.2%}, F1: {summary.f1_score:.3f}")
        
        return results
    
    def run_resilience_test(
        self,
        packet_loss_rates: List[float] = None,
        n_earthquakes: int = 100,
        n_false_positives: int = 500
    ) -> Dict[float, SimulationSummary]:
        """
        Test system resilience under network packet loss.
        
        Args:
            packet_loss_rates: List of packet loss rates to test
            n_earthquakes: Earthquakes per configuration
            n_false_positives: False positives per configuration
            
        Returns:
            Dictionary mapping packet_loss_rate to SimulationSummary
        """
        if packet_loss_rates is None:
            packet_loss_rates = config.simulation.packet_loss_rates
        
        results = {}
        
        for loss_rate in packet_loss_rates:
            print(f"\nTesting packet loss: {loss_rate:.0%}")
            
            # Update packet loss
            self.device_manager.set_packet_loss_rate(loss_rate)
            
            # Run simulation
            summary = self.run_simulation(
                n_earthquakes=n_earthquakes,
                n_false_positives=n_false_positives,
                show_progress=True
            )
            
            results[loss_rate] = summary
            
            print(f"  TPR: {summary.recall:.2%}, FPR: {summary.false_positive_rate:.2%}")
        
        return results
