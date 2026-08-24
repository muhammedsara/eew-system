"""
Consensus Engine - Main Orchestrator

Combines all stages of the weighted spatiotemporal consensus protocol:
1. Spatial Clustering (DBSCAN)
2. Temporal Windowing (2s sliding window)
3. Weighted Voting (mobile=0.3, IoT=0.7)
4. Adaptive Thresholding (time-of-day)
5. Validation Rules (n_IoT ≥ 1, n_total ≥ 3)

From Paper Section III.B:
"An earthquake is declared if: Score ≥ θ ∧ n_IoT ≥ 1 ∧ n_total ≥ 3"

Author: Muhammed Şara
"""

from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from consensus.spatial_clustering import SpatialClusterer, SpatialCluster
from consensus.temporal_windowing import TemporalWindower, TemporalWindow
from consensus.weighted_voting import WeightedVoter, VotingResult
from consensus.adaptive_threshold import AdaptiveThreshold
from consensus.st_graph import check as st_graph_check
from utils.haversine import estimate_epicenter


@dataclass
class ConsensusDecision:
    """
    Final consensus decision for earthquake detection.
    
    Contains all information needed for alert generation.
    """
    is_earthquake: bool
    score: float
    threshold: float
    confidence: float
    
    # Trigger counts
    mobile_count: int
    iot_count: int
    total_count: int
    
    # Location estimate
    estimated_lat: float
    estimated_lon: float
    estimated_radius_km: float
    
    # Timing
    timestamp: float
    processing_time_ms: float
    
    # Validation details
    passes_score: bool
    passes_iot_requirement: bool
    passes_total_requirement: bool
    
    # Cluster info
    num_clusters: int
    primary_cluster_id: int
    
    # Debug info
    raw_triggers: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_earthquake': self.is_earthquake,
            'score': round(self.score, 4),
            'threshold': self.threshold,
            'confidence': round(self.confidence, 4),
            'mobile_count': self.mobile_count,
            'iot_count': self.iot_count,
            'total_count': self.total_count,
            'estimated_epicenter': {
                'lat': round(self.estimated_lat, 4),
                'lon': round(self.estimated_lon, 4)
            },
            'estimated_radius_km': round(self.estimated_radius_km, 2),
            'timestamp': self.timestamp,
            'processing_time_ms': round(self.processing_time_ms, 2),
            'validation': {
                'passes_score': self.passes_score,
                'passes_iot': self.passes_iot_requirement,
                'passes_total': self.passes_total_requirement
            },
            'num_clusters': self.num_clusters
        }
    
    def to_alert_message(self) -> str:
        """Format as alert message"""
        if not self.is_earthquake:
            return "No earthquake detected"
        
        return (
            f"🚨 DEPREM TESPİT EDİLDİ!\n"
            f"Skor: {self.score:.2f} (Eşik: {self.threshold})\n"
            f"Konum: {self.estimated_lat:.3f}°N, {self.estimated_lon:.3f}°E\n"
            f"Cihazlar: {self.mobile_count} mobil + {self.iot_count} IoT = {self.total_count}\n"
            f"Güven: {self.confidence:.1%}\n"
            f"Zaman: {datetime.fromtimestamp(self.timestamp).strftime('%H:%M:%S')}"
        )


class ConsensusEngine:
    """
    Main consensus engine for earthquake detection.
    
    Orchestrates the 5-stage weighted spatiotemporal consensus protocol.
    
    Processing Pipeline:
    1. Receive triggers from devices
    2. Spatial clustering (DBSCAN, ε=5km)
    3. Temporal windowing (2s sliding window)
    4. Weighted voting (mobile=0.3, IoT=0.7)
    5. Adaptive thresholding + validation rules
    6. Generate decision
    
    Attributes:
        spatial_clusterer: DBSCAN spatial clustering
        temporal_windower: 2s sliding window
        weighted_voter: Reliability-based voting
        adaptive_threshold: Time-of-day thresholds
    """
    
    def __init__(
        self,
        mobile_weight: float = None,
        iot_weight: float = None,
        spatial_eps_km: float = None,
        temporal_window_s: float = None,
        threshold: float = None
    ):
        """
        Initialize consensus engine.
        
        Args:
            mobile_weight: Weight for mobile votes (default: 0.3)
            iot_weight: Weight for IoT votes (default: 0.7)
            spatial_eps_km: DBSCAN epsilon in km (default: 5)
            temporal_window_s: Temporal window in seconds (default: 2)
            threshold: Base threshold for detection (default: from config)
        """
        self.spatial_clusterer = SpatialClusterer(eps_km=spatial_eps_km)
        self.temporal_windower = TemporalWindower(window_seconds=temporal_window_s)
        self.weighted_voter = WeightedVoter(
            mobile_weight=mobile_weight,
            iot_weight=iot_weight
        )
        self.adaptive_threshold = AdaptiveThreshold(threshold_default=threshold)
        
        # Ablation flags — when True the corresponding phase is bypassed
        self.disable_spatial = False
        self.disable_temporal = False
        self.disable_adaptive_threshold = False
        self.disable_iot_validation = False
        self.disable_graph = False
        self.disable_weighting = False  # uses equal weights (0.5/0.5)
        
        # Statistics
        self.total_decisions = 0
        self.earthquake_detections = 0
        self.false_alarms = 0  # Updated by external validation
        
        # Pending triggers for streaming mode
        self._pending_triggers = []
    
    def process(
        self,
        triggers: List[Dict],
        timestamp: float = None,
        hour: int = None
    ) -> ConsensusDecision:
        """
        Process a batch of triggers through the consensus pipeline.
        
        This is the main entry point for earthquake detection.
        
        Args:
            triggers: List of trigger dictionaries from devices
            timestamp: Current timestamp (default: now)
            hour: Hour for adaptive threshold (default: current hour)
            
        Returns:
            ConsensusDecision with detection result
        """
        start_time = time.time()
        timestamp = timestamp or start_time
        hour = hour if hour is not None else datetime.now().hour
        
        # Handle empty triggers
        if not triggers:
            return self._create_negative_decision(
                timestamp=timestamp,
                processing_time_ms=(time.time() - start_time) * 1000,
                reason="No triggers received"
            )
        
        # Stage 1: Spatial Clustering
        if self.disable_spatial:
            # Ablation: skip clustering — treat all triggers as a single cluster
            from consensus.spatial_clustering import SpatialCluster
            clusters = [SpatialCluster(
                cluster_id=0,
                triggers=triggers,
                center_lat=np.mean([t.get('lat', t.get('latitude', 0)) for t in triggers]),
                center_lon=np.mean([t.get('lon', t.get('longitude', 0)) for t in triggers]),
                radius_km=50.0,
                mobile_count=sum(1 for t in triggers if t.get('device_type') == 'mobile'),
                iot_count=sum(1 for t in triggers if t.get('device_type') == 'iot_anchor')
            )]
        else:
            clusters = self.spatial_clusterer.cluster(triggers)
        
        if not clusters:
            return self._create_negative_decision(
                timestamp=timestamp,
                processing_time_ms=(time.time() - start_time) * 1000,
                triggers=triggers,
                reason="No spatial clusters formed"
            )
        
        # Stage 2: Temporal Windowing (applied per cluster)
        best_decision = None
        best_composite_score = -1
        
        # Paper Equation 3: n_IoT ≥ 1 is a GLOBAL check — "at least one
        # IoT anchor confirmed this event", not per-cluster.  DBSCAN may
        # place IoT anchors in a separate cluster from the mobile mass,
        # so we pre-compute global IoT presence here.
        global_has_iot = any(
            t.get('device_type') == 'iot_anchor' for t in triggers
        )
        
        for cluster in clusters:
            # Get temporal windows for this cluster
            if self.disable_temporal:
                # Ablation: skip windowing — treat all cluster triggers as one window
                from consensus.temporal_windowing import TemporalWindow
                windows = [TemporalWindow(
                    window_id=0,
                    triggers=cluster.triggers,
                    start_time=min(t.get('timestamp', 0) for t in cluster.triggers),
                    end_time=max(t.get('timestamp', 0) for t in cluster.triggers),
                    mobile_count=sum(1 for t in cluster.triggers if t.get('device_type') == 'mobile'),
                    iot_count=sum(1 for t in cluster.triggers if t.get('device_type') == 'iot_anchor')
                )]
            else:
                windows = self.temporal_windower.group_triggers(cluster.triggers)
            
            for window in windows:
                # Stage 3: Weighted Voting
                if self.disable_weighting:
                    # Ablation: use equal weights (0.5 / 0.5)
                    original_mw = self.weighted_voter.mobile_weight
                    original_iw = self.weighted_voter.iot_weight
                    self.weighted_voter.mobile_weight = 0.5
                    self.weighted_voter.iot_weight = 0.5
                    voting_result = self.weighted_voter.calculate_score(window.triggers)
                    self.weighted_voter.mobile_weight = original_mw
                    self.weighted_voter.iot_weight = original_iw
                else:
                    voting_result = self.weighted_voter.calculate_score(window.triggers)
                
                # Stage 4: Adaptive Thresholding
                if self.disable_adaptive_threshold:
                    # Ablation: use fixed default threshold (0.85)
                    threshold = 0.85
                else:
                    threshold = self.adaptive_threshold.get_threshold(hour)
                passes_score = voting_result.score >= threshold
                
                # Stage 5: Validation Rules
                if self.disable_iot_validation:
                    # Ablation: skip IoT requirement — any cluster passes
                    passes_iot = True
                else:
                    # Global IoT check (Paper Eq. 3: n_IoT ≥ 1)
                    passes_iot = global_has_iot
                passes_total = self.weighted_voter.validate_total_requirement(window.triggers)

                # Stage 3: spatio-temporal graph consistency
                if self.disable_graph:
                    graph = {'passes': True, 'abstained': True}
                else:
                    graph = st_graph_check(window.triggers)
                passes_graph = graph['passes']

                # Decision
                is_earthquake = (passes_score and passes_iot and passes_total
                                 and passes_graph)
                
                # Calculate confidence
                confidence = self._calculate_confidence(
                    voting_result, passes_iot, passes_total
                )
                
                # Composite score: prioritize IoT presence, then device count, then score
                # This ensures we select clusters that can actually pass validation rules
                composite_score = (
                    voting_result.iot_count * 100 +  # IoT presence is critical
                    voting_result.total_count * 10 +  # More devices = more reliable
                    voting_result.score * 1            # Then by voting score
                )
                
                # Track best cluster/window by composite score
                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    # Estimate epicenter
                    coords = [
                        [t.get('lat', t.get('latitude', 0)),
                         t.get('lon', t.get('longitude', 0))]
                        for t in window.triggers
                    ]
                    coords = np.array(coords)
                    
                    confidences = np.array([
                        t.get('confidence', 0.5) for t in window.triggers
                    ])
                    
                    est_lat, est_lon = estimate_epicenter(coords, confidences)
                    
                    best_decision = ConsensusDecision(
                        is_earthquake=is_earthquake,
                        score=voting_result.score,
                        threshold=threshold,
                        confidence=confidence,
                        mobile_count=voting_result.mobile_count,
                        iot_count=voting_result.iot_count,
                        total_count=voting_result.total_count,
                        estimated_lat=est_lat,
                        estimated_lon=est_lon,
                        estimated_radius_km=cluster.radius_km,
                        timestamp=timestamp,
                        processing_time_ms=0,  # Will be updated
                        passes_score=passes_score,
                        passes_iot_requirement=passes_iot,
                        passes_total_requirement=passes_total,
                        num_clusters=len(clusters),
                        primary_cluster_id=cluster.cluster_id,
                        raw_triggers=window.triggers
                    )
        
        # Finalize
        processing_time = (time.time() - start_time) * 1000
        
        if best_decision:
            best_decision.processing_time_ms = processing_time
            
            # Update statistics
            self.total_decisions += 1
            if best_decision.is_earthquake:
                self.earthquake_detections += 1
            
            return best_decision
        
        return self._create_negative_decision(
            timestamp=timestamp,
            processing_time_ms=processing_time,
            triggers=triggers,
            reason="No valid windows found"
        )
    
    def process_streaming(
        self,
        trigger: Dict,
        current_time: float = None
    ) -> Optional[ConsensusDecision]:
        """
        Process triggers in streaming mode.
        
        For real-time processing where triggers arrive one at a time.
        
        Args:
            trigger: Single trigger dictionary
            current_time: Current timestamp
            
        Returns:
            ConsensusDecision when window completes, None otherwise
        """
        current_time = current_time or time.time()
        
        # Add to pending triggers
        self._pending_triggers.append(trigger)
        
        # Process through temporal windower
        completed_window = self.temporal_windower.process_streaming(
            trigger, current_time
        )
        
        if completed_window:
            # Process the completed window
            decision = self.process(
                completed_window.triggers,
                timestamp=current_time
            )
            
            # Clear old pending triggers
            window_start = completed_window.start_time
            self._pending_triggers = [
                t for t in self._pending_triggers
                if t.get('time', t.get('timestamp', 0)) > window_start
            ]
            
            return decision
        
        return None
    
    def flush_streaming(self) -> Optional[ConsensusDecision]:
        """
        Flush any pending triggers in streaming mode.
        
        Returns:
            ConsensusDecision if there were pending triggers
        """
        window = self.temporal_windower.flush_streaming()
        
        if window and window.triggers:
            return self.process(window.triggers)
        
        return None
    
    def _calculate_confidence(
        self,
        voting_result: VotingResult,
        passes_iot: bool,
        passes_total: bool
    ) -> float:
        """
        Calculate overall confidence in the detection.
        
        Combines:
        - Weighted voting score
        - IoT presence (higher confidence with IoT)
        - Device count (more devices = more confidence)
        """
        base_confidence = voting_result.score
        
        # Boost for IoT presence
        if passes_iot and voting_result.iot_count > 0:
            iot_boost = min(0.15, voting_result.iot_count * 0.05)
            base_confidence += iot_boost
        
        # Boost for device count
        if passes_total:
            count_boost = min(0.1, voting_result.total_count * 0.01)
            base_confidence += count_boost
        
        return min(1.0, base_confidence)
    
    def _create_negative_decision(
        self,
        timestamp: float,
        processing_time_ms: float,
        triggers: List[Dict] = None,
        reason: str = ""
    ) -> ConsensusDecision:
        """Create a negative (no earthquake) decision"""
        return ConsensusDecision(
            is_earthquake=False,
            score=0.0,
            threshold=self.adaptive_threshold.get_threshold(),
            confidence=0.0,
            mobile_count=0,
            iot_count=0,
            total_count=len(triggers) if triggers else 0,
            estimated_lat=0.0,
            estimated_lon=0.0,
            estimated_radius_km=0.0,
            timestamp=timestamp,
            processing_time_ms=processing_time_ms,
            passes_score=False,
            passes_iot_requirement=False,
            passes_total_requirement=False,
            num_clusters=0,
            primary_cluster_id=-1,
            raw_triggers=triggers or []
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'total_decisions': self.total_decisions,
            'earthquake_detections': self.earthquake_detections,
            'detection_rate': (
                self.earthquake_detections / self.total_decisions
                if self.total_decisions > 0 else 0
            ),
            'configuration': {
                'mobile_weight': self.weighted_voter.mobile_weight,
                'iot_weight': self.weighted_voter.iot_weight,
                'spatial_eps_km': self.spatial_clusterer.eps_km,
                'temporal_window_s': self.temporal_windower.window_seconds
            }
        }
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.total_decisions = 0
        self.earthquake_detections = 0
        self.false_alarms = 0
    
    def update_weights(self, mobile_weight: float, iot_weight: float):
        """Update voting weights (for ablation studies)"""
        self.weighted_voter.mobile_weight = mobile_weight
        self.weighted_voter.iot_weight = iot_weight
