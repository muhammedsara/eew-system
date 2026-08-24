"""
Baseline Detection Methods

Implements baseline methods for comparison (Paper Section IV.B):
- B1: MyShake-like (mobile-only, ≥5 devices within 10km)
- B2: IoT-only (≥2 anchors triggered)
- B3: Unweighted hybrid (equal weights)

Author: Muhammed Şara
"""

from typing import List, Dict, Tuple, Any
from enum import Enum
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from consensus.spatial_clustering import SpatialClusterer
from consensus.temporal_windowing import TemporalWindower


class BaselineType(Enum):
    """Baseline detection methods from paper"""
    MYSHAKE_LIKE = "myshake_like"  # B1: Mobile-only
    IOT_ONLY = "iot_only"          # B2: IoT-only
    UNWEIGHTED = "unweighted"       # B3: Equal weights


class BaselineDetector:
    """
    Baseline detection methods for comparison.
    
    From Paper Table II:
    | Method        | Precision | Recall | FPR   | F1    |
    |---------------|-----------|--------|-------|-------|
    | MyShake-like  | 75.4%     | 92.0%  | 15.2% | 0.830 |
    | IoT-only      | 95.1%     | 78.0%  | 1.8%  | 0.857 |
    | Unweighted    | 88.2%     | 88.0%  | 7.6%  | 0.881 |
    | Proposed      | 96.0%     | 94.0%  | 2.0%  | 0.950 |
    """
    
    def __init__(self, baseline_type: BaselineType):
        """
        Initialize baseline detector.
        
        Args:
            baseline_type: Type of baseline method
        """
        self.baseline_type = baseline_type
        self.spatial_clusterer = SpatialClusterer(eps_km=10.0)  # 10km for MyShake
        self.temporal_windower = TemporalWindower()
    
    def detect(self, triggers: List[Dict]) -> Tuple[bool, float]:
        """
        Run baseline detection.
        
        Args:
            triggers: List of trigger dictionaries
            
        Returns:
            Tuple of (is_earthquake, confidence)
        """
        if self.baseline_type == BaselineType.MYSHAKE_LIKE:
            return self._detect_myshake(triggers)
        elif self.baseline_type == BaselineType.IOT_ONLY:
            return self._detect_iot_only(triggers)
        elif self.baseline_type == BaselineType.UNWEIGHTED:
            return self._detect_unweighted(triggers)
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline_type}")
    
    def _detect_myshake(self, triggers: List[Dict]) -> Tuple[bool, float]:
        """
        MyShake-like detection (B1).
        
        Algorithm:
        - Cluster mobile triggers within 10km
        - Require ≥5 mobile devices
        - Simple majority voting
        """
        # Filter mobile triggers only
        mobile_triggers = [
            t for t in triggers
            if t.get('device_type') == 'mobile'
        ]
        
        if len(mobile_triggers) < 5:
            return False, 0.0
        
        # Spatial clustering with larger radius
        clusters = self.spatial_clusterer.cluster(mobile_triggers)
        
        if not clusters:
            return False, 0.0
        
        # Check if any cluster has ≥5 devices
        for cluster in clusters:
            if cluster.mobile_count >= 5:
                # Simple average confidence
                confidences = [
                    t.get('confidence', 0.5)
                    for t in cluster.triggers
                ]
                confidence = np.mean(confidences)
                return True, confidence
        
        return False, 0.0
    
    def _detect_iot_only(self, triggers: List[Dict]) -> Tuple[bool, float]:
        """
        IoT-only detection (B2).
        
        Algorithm:
        - Use only IoT anchor triggers
        - Require ≥2 anchors triggered
        - Traditional seismic network approach
        """
        # Filter IoT triggers only
        iot_triggers = [
            t for t in triggers
            if t.get('device_type') == 'iot_anchor'
        ]
        
        if len(iot_triggers) < 2:
            return False, 0.0
        
        # Temporal grouping
        windows = self.temporal_windower.group_triggers(iot_triggers)
        
        for window in windows:
            if window.iot_count >= 2:
                confidences = [
                    t.get('confidence', 0.5)
                    for t in window.triggers
                ]
                confidence = np.mean(confidences)
                return True, confidence
        
        return False, 0.0
    
    def _detect_unweighted(self, triggers: List[Dict]) -> Tuple[bool, float]:
        """
        Unweighted hybrid detection (B3).
        
        Algorithm:
        - Use both mobile and IoT triggers
        - Equal weights (0.5, 0.5)
        - Same thresholding as proposed method
        """
        if len(triggers) < 3:
            return False, 0.0
        
        # Spatial clustering
        clusters = self.spatial_clusterer.cluster(triggers)
        
        if not clusters:
            return False, 0.0
        
        for cluster in clusters:
            windows = self.temporal_windower.group_triggers(cluster.triggers)
            
            for window in windows:
                if window.total_count >= 3:
                    # Equal weight voting
                    confidences = [
                        t.get('confidence', 0.5)
                        for t in window.triggers
                    ]
                    score = np.mean(confidences)
                    
                    # Same threshold logic (simplified)
                    threshold = 0.85
                    
                    if score >= threshold:
                        return True, score
        
        return False, 0.0
    
    @staticmethod
    def get_expected_performance(baseline_type: BaselineType) -> Dict[str, float]:
        """
        Get expected performance from paper.
        
        Returns:
            Dictionary with expected metrics
        """
        performance = {
            BaselineType.MYSHAKE_LIKE: {
                'precision': 0.754,
                'recall': 0.920,
                'fpr': 0.152,
                'f1': 0.830
            },
            BaselineType.IOT_ONLY: {
                'precision': 0.951,
                'recall': 0.780,
                'fpr': 0.018,
                'f1': 0.857
            },
            BaselineType.UNWEIGHTED: {
                'precision': 0.882,
                'recall': 0.880,
                'fpr': 0.076,
                'f1': 0.881
            }
        }
        
        return performance.get(baseline_type, {})


def compare_all_baselines(
    triggers_list: List[List[Dict]],
    ground_truths: List[bool]
) -> Dict[str, Dict[str, float]]:
    """
    Compare all baseline methods on a dataset.
    
    Args:
        triggers_list: List of trigger lists (one per scenario)
        ground_truths: Ground truth labels
        
    Returns:
        Dictionary mapping method name to metrics
    """
    from evaluation.metrics import MetricsCalculator
    
    results = {}
    
    for baseline_type in BaselineType:
        detector = BaselineDetector(baseline_type)
        calculator = MetricsCalculator()
        
        for triggers, gt in zip(triggers_list, ground_truths):
            detected, score = detector.detect(triggers)
            calculator.add_result(gt, detected, score)
        
        metrics = calculator.calculate_all_metrics()
        results[baseline_type.value] = metrics
    
    return results
