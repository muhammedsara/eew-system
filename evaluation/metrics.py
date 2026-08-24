"""
Evaluation Metrics

Calculates performance metrics for earthquake detection systems.
Implements metrics from Paper Table II.

Author: Muhammed Şara
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve,
    confusion_matrix
)


class MetricsCalculator:
    """
    Calculates evaluation metrics for EEW systems.
    
    Metrics from Paper Table II:
    - Precision: TP / (TP + FP)
    - Recall (TPR): TP / (TP + FN)
    - FPR: FP / (FP + TN)
    - F1-Score: 2 * Precision * Recall / (Precision + Recall)
    """
    
    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.y_scores = []
    
    def add_result(
        self,
        ground_truth: bool,
        prediction: bool,
        score: float = None
    ):
        """
        Add a single prediction result.
        
        Args:
            ground_truth: True if actual earthquake
            prediction: True if system detected earthquake
            score: Confidence score for ROC curve
        """
        self.y_true.append(int(ground_truth))
        self.y_pred.append(int(prediction))
        if score is not None:
            self.y_scores.append(score)
    
    def add_batch(
        self,
        ground_truths: List[bool],
        predictions: List[bool],
        scores: List[float] = None
    ):
        """Add batch of results"""
        for i, (gt, pred) in enumerate(zip(ground_truths, predictions)):
            score = scores[i] if scores else None
            self.add_result(gt, pred, score)
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Returns:
            Dictionary with all metrics
        """
        if len(self.y_true) == 0:
            return {}
        
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        
        # Basic metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y_true)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'tpr': recall,  # Alias
            'fpr': fpr,
            'f1_score': f1,
            'accuracy': accuracy,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'total': len(y_true)
        }
        
        # AUC if scores available
        if len(self.y_scores) == len(self.y_true):
            try:
                fpr_curve, tpr_curve, _ = roc_curve(y_true, self.y_scores)
                metrics['auc'] = auc(fpr_curve, tpr_curve)
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])
    
    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve data.
        
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        if len(self.y_scores) != len(self.y_true):
            raise ValueError("Scores required for ROC curve")
        
        return roc_curve(self.y_true, self.y_scores)
    
    def get_precision_recall_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Precision-Recall curve data.
        
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        if len(self.y_scores) != len(self.y_true):
            raise ValueError("Scores required for PR curve")
        
        return precision_recall_curve(self.y_true, self.y_scores)
    
    def compare_methods(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple detection methods.
        
        Args:
            results: Dict mapping method name to metrics dict
            
        Returns:
            Comparison table
        """
        comparison = {}
        
        for method, metrics in results.items():
            comparison[method] = {
                'Precision': metrics.get('precision', 0) * 100,
                'Recall (TPR)': metrics.get('recall', 0) * 100,
                'FPR': metrics.get('fpr', 0) * 100,
                'F1': metrics.get('f1_score', 0)
            }
        
        return comparison
    
    def bootstrap_confidence_interval(
        self,
        n_iterations: int = 1000,
        confidence: float = 0.95,
        seed: int = 42
    ) -> Dict[str, Dict[str, float]]:
        """
        Bootstrap 95 % confidence intervals for all metrics.
        
        Paper §6:
          "All reported metrics are means with 95% confidence intervals
           computed from 1,000 bootstrap iterations."
        
        Args:
            n_iterations: Number of bootstrap samples (default: 1000)
            confidence: Confidence level (default: 0.95)
            seed: Random seed for reproducibility
            
        Returns:
            Dict mapping metric name to {'mean', 'lower', 'upper', 'ci'}
        """
        rng = np.random.RandomState(seed)
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        n = len(y_true)
        
        if n == 0:
            return {}
        
        boot_metrics = {
            'precision': [], 'recall': [], 'f1_score': [],
            'fpr': [], 'accuracy': []
        }
        
        for _ in range(n_iterations):
            indices = rng.randint(0, n, size=n)
            bt = y_true[indices]
            bp = y_pred[indices]
            
            # Avoid degenerate samples
            if len(np.unique(bt)) < 2:
                continue
            
            tn, fp, fn, tp = confusion_matrix(bt, bp, labels=[0, 1]).ravel()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
            acc = (tp + tn) / len(bt)
            
            boot_metrics['precision'].append(prec)
            boot_metrics['recall'].append(rec)
            boot_metrics['f1_score'].append(f1)
            boot_metrics['fpr'].append(fpr_val)
            boot_metrics['accuracy'].append(acc)
        
        alpha = 1 - confidence
        ci_results = {}
        for metric_name, values in boot_metrics.items():
            values = np.array(values)
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            mean = np.mean(values)
            ci_results[metric_name] = {
                'mean': float(mean),
                'lower': float(lower),
                'upper': float(upper),
                'ci': float((upper - lower) / 2),
            }
        
        return ci_results
    
    def reset(self):
        """Reset accumulated results"""
        self.y_true = []
        self.y_pred = []
        self.y_scores = []
    
    def print_summary(self):
        """Print metrics summary"""
        metrics = self.calculate_all_metrics()
        
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Total samples: {metrics.get('total', 0)}")
        print(f"  TP: {metrics.get('tp', 0)}, FP: {metrics.get('fp', 0)}")
        print(f"  TN: {metrics.get('tn', 0)}, FN: {metrics.get('fn', 0)}")
        print("-"*50)
        print(f"Precision: {metrics.get('precision', 0):.2%}")
        print(f"Recall (TPR): {metrics.get('recall', 0):.2%}")
        print(f"FPR: {metrics.get('fpr', 0):.2%}")
        print(f"F1-Score: {metrics.get('f1_score', 0):.3f}")
        print(f"Accuracy: {metrics.get('accuracy', 0):.2%}")
        if 'auc' in metrics:
            print(f"AUC: {metrics.get('auc', 0):.3f}")
        print("="*50 + "\n")
