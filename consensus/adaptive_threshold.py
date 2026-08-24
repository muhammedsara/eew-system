"""
Adaptive Threshold

Stage 4 of the consensus protocol (Paper Section III.B, Equation 2):
"The decision threshold θ adapts to time-of-day:
θ(h) = 0.75 if 2 ≤ h < 6   (night)
θ(h) = 0.90 if 8 ≤ h < 18  (day)
θ(h) = 0.85 otherwise

Night hours have lower ambient noise (no traffic), permitting higher sensitivity.
Daytime hours require stricter thresholds to filter urban vibrations."

Author: Muhammed Şara
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class AdaptiveThreshold:
    """
    Adaptive thresholding based on time-of-day.
    
    Adjusts detection thresholds based on expected ambient noise levels:
    - Night (2-6 AM): Lower threshold (0.75) - less noise, more sensitive
    - Day (8 AM - 6 PM): Higher threshold (0.90) - filter traffic/construction
    - Other times: Default threshold (0.85)
    
    This reduces false alarms during high-activity periods while
    maintaining sensitivity during quiet hours.
    """
    
    def __init__(
        self,
        threshold_night: float = None,
        threshold_day: float = None,
        threshold_default: float = None
    ):
        """
        Initialize adaptive threshold.
        
        Args:
            threshold_night: Threshold for 2-6 AM (default: 0.75)
            threshold_day: Threshold for 8 AM - 6 PM (default: 0.90)
            threshold_default: Threshold for other hours (default: 0.85)
        """
        self.threshold_night = threshold_night or config.consensus.threshold_night
        self.threshold_day = threshold_day or config.consensus.threshold_day
        self.threshold_default = threshold_default or config.consensus.threshold_default
    
    def get_threshold(self, hour: int = None) -> float:
        """
        Get threshold for given hour.
        
        Args:
            hour: Hour of day (0-23). If None, uses current hour.
            
        Returns:
            Detection threshold value
        """
        if hour is None:
            hour = datetime.now().hour
        
        if 2 <= hour < 6:
            return self.threshold_night
        elif 8 <= hour < 18:
            return self.threshold_day
        else:
            return self.threshold_default
    
    def get_threshold_with_context(
        self,
        hour: int = None
    ) -> Tuple[float, str, str]:
        """
        Get threshold with context information.
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            Tuple of (threshold, period_name, reason)
        """
        if hour is None:
            hour = datetime.now().hour
        
        if 2 <= hour < 6:
            return (
                self.threshold_night,
                "night",
                "Low ambient noise - more sensitive detection"
            )
        elif 8 <= hour < 18:
            return (
                self.threshold_day,
                "day",
                "High urban activity - stricter filtering"
            )
        else:
            return (
                self.threshold_default,
                "transition",
                "Moderate noise - balanced threshold"
            )
    
    def evaluate(
        self,
        score: float,
        hour: int = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate if score meets threshold.
        
        Args:
            score: Consensus score to evaluate
            hour: Hour of day (optional)
            
        Returns:
            Tuple of (passes_threshold, details_dict)
        """
        threshold, period, reason = self.get_threshold_with_context(hour)
        passes = score >= threshold
        
        return passes, {
            'score': score,
            'threshold': threshold,
            'passes': passes,
            'margin': score - threshold,
            'period': period,
            'reason': reason,
            'hour': hour or datetime.now().hour
        }
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """Get all configured thresholds"""
        return {
            'night': self.threshold_night,
            'day': self.threshold_day,
            'default': self.threshold_default
        }
    
    def get_hourly_thresholds(self) -> Dict[int, float]:
        """Get threshold for each hour of day"""
        return {hour: self.get_threshold(hour) for hour in range(24)}
    
    def set_thresholds(
        self,
        night: float = None,
        day: float = None,
        default: float = None
    ):
        """
        Update threshold values.
        
        Args:
            night: New night threshold
            day: New day threshold
            default: New default threshold
        """
        if night is not None:
            self.threshold_night = night
        if day is not None:
            self.threshold_day = day
        if default is not None:
            self.threshold_default = default
    
    def sensitivity_analysis(
        self,
        score: float,
        threshold_variations: list = None
    ) -> Dict[str, bool]:
        """
        Analyze how score would fare under different thresholds.
        
        Args:
            score: Score to evaluate
            threshold_variations: List of thresholds to test
            
        Returns:
            Dictionary mapping threshold to pass/fail
        """
        if threshold_variations is None:
            threshold_variations = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        return {
            f"θ={t}": score >= t
            for t in threshold_variations
        }


class DynamicThreshold(AdaptiveThreshold):
    """
    Extended adaptive threshold with learning capability.
    
    Can adjust thresholds based on observed false alarm rates.
    For future implementation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Historical data for learning
        self._history = {
            'night': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
            'day': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
            'transition': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        }
    
    def record_outcome(
        self,
        hour: int,
        predicted: bool,
        actual: bool
    ):
        """
        Record detection outcome for learning.
        
        Args:
            hour: Hour of detection
            predicted: Whether earthquake was predicted
            actual: Whether it was actually an earthquake
        """
        _, period, _ = self.get_threshold_with_context(hour)
        
        if predicted and actual:
            self._history[period]['tp'] += 1
        elif predicted and not actual:
            self._history[period]['fp'] += 1
        elif not predicted and actual:
            self._history[period]['fn'] += 1
        else:
            self._history[period]['tn'] += 1
    
    def get_period_metrics(self, period: str) -> Dict[str, float]:
        """Get performance metrics for a period"""
        h = self._history[period]
        total = h['tp'] + h['fp'] + h['fn'] + h['tn']
        
        if total == 0:
            return {'precision': 0, 'recall': 0, 'fpr': 0}
        
        precision = h['tp'] / (h['tp'] + h['fp']) if (h['tp'] + h['fp']) > 0 else 0
        recall = h['tp'] / (h['tp'] + h['fn']) if (h['tp'] + h['fn']) > 0 else 0
        fpr = h['fp'] / (h['fp'] + h['tn']) if (h['fp'] + h['tn']) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'total_samples': total
        }
