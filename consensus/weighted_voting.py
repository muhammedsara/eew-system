"""
Weighted Voting Algorithm

Stage 3 of the consensus protocol (Paper Section III.B, Equation 1):
"The consensus score for a spatiotemporal window is:
Score = [Σ(mobile_conf × w_m) + Σ(iot_conf × w_i)] / (|M| + |I|)

where M and I are sets of mobile and IoT devices, w_m = 0.3 and w_i = 0.7."

Author: Muhammed Şara
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


@dataclass
class VotingResult:
    """Result of weighted voting"""
    score: float
    mobile_contribution: float
    iot_contribution: float
    mobile_count: int
    iot_count: int
    total_count: int
    weighted_mobile_sum: float
    weighted_iot_sum: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'score': round(self.score, 4),
            'mobile_contribution': round(self.mobile_contribution, 4),
            'iot_contribution': round(self.iot_contribution, 4),
            'mobile_count': self.mobile_count,
            'iot_count': self.iot_count,
            'total_count': self.total_count
        }


class WeightedVoter:
    """
    Weighted voting for heterogeneous sensor fusion.
    
    Core innovation: Reliability-based weighting (mobile=0.3, IoT=0.7)
    validated through grid search optimization.
    
    From Paper Section III.C:
    "The accuracy ratio 0.95/0.88 ≈ 1.08 suggests approximately equal weighting.
    However, mobile devices exhibit higher variance due to user handling.
    Therefore, we emphasize IoT reliability through differential weighting."
    
    Attributes:
        mobile_weight: Weight for mobile device votes (default: 0.3)
        iot_weight: Weight for IoT anchor votes (default: 0.7)
    """
    
    def __init__(
        self,
        mobile_weight: float = None,
        iot_weight: float = None
    ):
        """
        Initialize weighted voter.
        
        Args:
            mobile_weight: Weight for mobile votes (default: 0.3)
            iot_weight: Weight for IoT votes (default: 0.7)
        """
        self.mobile_weight = mobile_weight or config.consensus.mobile_weight
        self.iot_weight = iot_weight or config.consensus.iot_weight
        
        # Validate weights
        total = self.mobile_weight + self.iot_weight
        if abs(total - 1.0) > 0.01:
            # Normalize weights
            self.mobile_weight /= total
            self.iot_weight /= total
    
    def calculate_score(self, triggers: List[Dict]) -> VotingResult:
        """
        Calculate weighted consensus score.
        
        Formula (Paper Equation 1):
        Score = [Σ(c_i × w_m) + Σ(c_j × w_i)] / [|M| × w_m + |I| × w_i]
        
        Normalised so that when every device reports confidence=1.0
        the score is 1.0 regardless of device mix.
        
        Args:
            triggers: List of trigger dictionaries with 'device_type' and 'confidence'
            
        Returns:
            VotingResult with calculated score
        """
        mobile_triggers = [
            t for t in triggers 
            if t.get('device_type') == 'mobile'
        ]
        iot_triggers = [
            t for t in triggers 
            if t.get('device_type') == 'iot_anchor'
        ]
        
        # Sum weighted confidence scores
        mobile_sum = sum(
            t.get('confidence', 0.5) * self.mobile_weight
            for t in mobile_triggers
        )
        iot_sum = sum(
            t.get('confidence', 0.5) * self.iot_weight
            for t in iot_triggers
        )
        
        total_count = len(mobile_triggers) + len(iot_triggers)
        
        if total_count == 0:
            return VotingResult(
                score=0.0,
                mobile_contribution=0.0,
                iot_contribution=0.0,
                mobile_count=0,
                iot_count=0,
                total_count=0,
                weighted_mobile_sum=0.0,
                weighted_iot_sum=0.0
            )
        
        # Denominator = sum of applied weights per device
        # This normalises score to [0, 1]: score=1 when every device has conf=1
        weight_sum = (
            len(mobile_triggers) * self.mobile_weight
            + len(iot_triggers) * self.iot_weight
        )
        
        score = (mobile_sum + iot_sum) / weight_sum if weight_sum > 0 else 0.0
        
        return VotingResult(
            score=score,
            mobile_contribution=mobile_sum / weight_sum if weight_sum > 0 else 0,
            iot_contribution=iot_sum / weight_sum if weight_sum > 0 else 0,
            mobile_count=len(mobile_triggers),
            iot_count=len(iot_triggers),
            total_count=total_count,
            weighted_mobile_sum=mobile_sum,
            weighted_iot_sum=iot_sum
        )
    
    def calculate_score_normalized(self, triggers: List[Dict]) -> VotingResult:
        """
        Calculate normalized weighted score.
        
        Alternative formulation that normalizes by maximum possible score.
        
        Args:
            triggers: List of trigger dictionaries
            
        Returns:
            VotingResult with normalized score [0, 1]
        """
        result = self.calculate_score(triggers)
        
        # Maximum possible score is when all confidences are 1.0
        if result.total_count == 0:
            return result
        
        max_mobile = result.mobile_count * self.mobile_weight
        max_iot = result.iot_count * self.iot_weight
        max_score = (max_mobile + max_iot) / result.total_count
        
        if max_score > 0:
            normalized_score = result.score / max_score
        else:
            normalized_score = result.score
        
        return VotingResult(
            score=normalized_score,
            mobile_contribution=result.mobile_contribution,
            iot_contribution=result.iot_contribution,
            mobile_count=result.mobile_count,
            iot_count=result.iot_count,
            total_count=result.total_count,
            weighted_mobile_sum=result.weighted_mobile_sum,
            weighted_iot_sum=result.weighted_iot_sum
        )
    
    def get_effective_weight(
        self,
        mobile_count: int,
        iot_count: int
    ) -> Tuple[float, float]:
        """
        Calculate effective weights based on device counts.
        
        When there are no IoT anchors, mobile votes get full weight, etc.
        
        Args:
            mobile_count: Number of mobile triggers
            iot_count: Number of IoT triggers
            
        Returns:
            Tuple of (effective_mobile_weight, effective_iot_weight)
        """
        total = mobile_count + iot_count
        
        if total == 0:
            return 0.0, 0.0
        
        # Calculate contribution ratios
        mobile_ratio = mobile_count / total
        iot_ratio = iot_count / total
        
        # Weighted contributions
        mobile_effective = mobile_ratio * self.mobile_weight
        iot_effective = iot_ratio * self.iot_weight
        
        # Normalize
        effective_total = mobile_effective + iot_effective
        if effective_total > 0:
            return mobile_effective / effective_total, iot_effective / effective_total
        
        return mobile_ratio, iot_ratio
    
    def sensitivity_analysis(
        self,
        triggers: List[Dict],
        weight_pairs: List[Tuple[float, float]] = None
    ) -> Dict[str, float]:
        """
        Analyze score sensitivity to weight changes.
        
        Used for ablation study (Paper Figure 4).
        
        Args:
            triggers: List of trigger dictionaries
            weight_pairs: List of (mobile_weight, iot_weight) pairs to test
            
        Returns:
            Dictionary mapping weight pair strings to scores
        """
        if weight_pairs is None:
            weight_pairs = config.consensus.weight_grid
        
        results = {}
        original_mobile = self.mobile_weight
        original_iot = self.iot_weight
        
        for m_weight, i_weight in weight_pairs:
            self.mobile_weight = m_weight
            self.iot_weight = i_weight
            
            result = self.calculate_score(triggers)
            key = f"({m_weight:.1f}, {i_weight:.1f})"
            results[key] = result.score
        
        # Restore original weights
        self.mobile_weight = original_mobile
        self.iot_weight = original_iot
        
        return results
    
    def validate_iot_requirement(
        self,
        triggers: List[Dict],
        min_iot: int = None
    ) -> bool:
        """
        Check if IoT requirement is met.
        
        From Paper Equation 3:
        "n_IoT ≥ 1: At least one high-reliability sensor confirms the event"
        
        Args:
            triggers: List of trigger dictionaries
            min_iot: Minimum IoT count required (default: 1)
            
        Returns:
            True if requirement is met
        """
        min_iot = min_iot or config.consensus.min_iot_triggers
        
        iot_count = sum(
            1 for t in triggers 
            if t.get('device_type') == 'iot_anchor'
        )
        
        return iot_count >= min_iot
    
    def validate_total_requirement(
        self,
        triggers: List[Dict],
        min_total: int = None
    ) -> bool:
        """
        Check if total device requirement is met.
        
        From Paper Equation 3:
        "n_total ≥ 3: At least 3 total devices for consensus"
        
        Args:
            triggers: List of trigger dictionaries
            min_total: Minimum total count required (default: 3)
            
        Returns:
            True if requirement is met
        """
        min_total = min_total or config.consensus.min_total_triggers
        return len(triggers) >= min_total
