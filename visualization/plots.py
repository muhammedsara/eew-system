"""
Visualization Module for EEW System

Creates plots and visualizations for:
- Performance comparison charts
- ROC curves
- Ablation study heatmaps
- Resilience under packet loss

Author: Muhammed Şara
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ResultsVisualizer:
    """Visualizes experiment results"""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'results', 'figures'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    def plot_comparison_table(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot Table II comparison from paper.
        
        Args:
            results: Dict mapping method name to metrics
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        methods = list(results.keys())
        metrics = ['precision', 'recall', 'fpr', 'f1_score']
        metric_labels = ['Precision (%)', 'Recall/TPR (%)', 'FPR (%)', 'F1-Score']
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.suptitle('Method Performance Comparison (Table II)', fontsize=14, fontweight='bold')
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = []
            for method in methods:
                val = results[method].get(metric, 0)
                if metric != 'f1_score':
                    val *= 100  # Convert to percentage
                values.append(val)
            
            bars = axes[i].bar(methods, values, color=self.colors[:len(methods)])
            axes[i].set_title(label)
            axes[i].set_ylabel(label)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[i].annotate(f'{val:.1f}' if metric != 'f1_score' else f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves(
        self,
        roc_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot ROC curves for multiple methods.
        
        Args:
            roc_data: Dict mapping method name to (fpr, tpr) arrays
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for i, (method, (fpr, tpr)) in enumerate(roc_data.items()):
            from sklearn.metrics import auc
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=self.colors[i % len(self.colors)],
                   lw=2, label=f'{method} (AUC = {roc_auc:.3f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_weight_ablation(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot weight ablation study heatmap (Figure 3).
        
        Args:
            ablation_results: Dict mapping weight pair to metrics
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Extract weight pairs and metrics
        weight_pairs = []
        f1_scores = []
        
        for key, metrics in ablation_results.items():
            # Parse "(0.3, 0.7)" format
            mobile_w = float(key.split(',')[0].strip('('))
            f1_scores.append(metrics.get('f1_score', 0))
            weight_pairs.append(mobile_w)
        
        # Create heatmap data
        mobile_weights = sorted(set(weight_pairs))
        iot_weights = [1 - w for w in mobile_weights]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar chart for F1 vs weight
        bars = ax.bar(range(len(weight_pairs)), f1_scores, color=self.colors[0])
        ax.set_xticks(range(len(weight_pairs)))
        ax.set_xticklabels([f'M:{w:.1f}\nI:{1-w:.1f}' for w in weight_pairs])
        ax.set_xlabel('Weight Configuration (Mobile:IoT)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Weight Ablation Study (Figure 3)')
        
        # Highlight optimal
        max_idx = np.argmax(f1_scores)
        bars[max_idx].set_color(self.colors[2])
        ax.annotate('Optimal', xy=(max_idx, f1_scores[max_idx]),
                   xytext=(max_idx, f1_scores[max_idx] + 0.02),
                   ha='center', fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_resilience_curve(
        self,
        resilience_results: Dict[float, Dict[str, float]],
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot resilience under packet loss (Figure 4).
        
        Args:
            resilience_results: Dict mapping packet loss rate to metrics
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        loss_rates = sorted(resilience_results.keys())
        tprs = [resilience_results[r].get('recall', 0) * 100 for r in loss_rates]
        fprs = [resilience_results[r].get('fpr', 0) * 100 for r in loss_rates]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot([r * 100 for r in loss_rates], tprs, 'o-',
               color=self.colors[0], linewidth=2, markersize=8, label='TPR')
        ax.plot([r * 100 for r in loss_rates], fprs, 's--',
               color=self.colors[2], linewidth=2, markersize=8, label='FPR')
        
        # Target lines
        ax.axhline(y=94, color=self.colors[0], linestyle=':', alpha=0.5, label='Target TPR (94%)')
        ax.axhline(y=2, color=self.colors[2], linestyle=':', alpha=0.5, label='Target FPR (2%)')
        
        ax.set_xlabel('Packet Loss Rate (%)')
        ax.set_ylabel('Rate (%)')
        ax.set_title('System Resilience Under Packet Loss (Figure 4)')
        ax.legend(loc='center right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 35])
        ax.set_ylim([0, 100])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_time_of_day_threshold(
        self,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot adaptive threshold based on time of day.
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        from consensus.adaptive_threshold import AdaptiveThreshold
        
        thresholder = AdaptiveThreshold()
        hours = range(24)
        thresholds = [thresholder.get_threshold(h) for h in hours]
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.fill_between(hours, thresholds, alpha=0.3, color=self.colors[1])
        ax.plot(hours, thresholds, 'o-', color=self.colors[1], linewidth=2)
        
        # Mark periods
        ax.axvspan(0, 6, alpha=0.1, color='blue', label='Night (Lower)')
        ax.axvspan(6, 8, alpha=0.1, color='orange', label='Transition')
        ax.axvspan(8, 18, alpha=0.1, color='yellow', label='Day (Higher)')
        ax.axvspan(18, 22, alpha=0.1, color='orange')
        ax.axvspan(22, 24, alpha=0.1, color='blue')
        
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Detection Threshold')
        ax.set_title('Adaptive Threshold by Time of Day (Equation 2)')
        ax.set_xticks(range(0, 24, 2))
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_all_figures(
        self,
        comparison_results: Dict = None,
        ablation_results: Dict = None,
        resilience_results: Dict = None
    ):
        """Generate all paper figures"""
        
        # Default comparison if not provided
        if comparison_results is None:
            comparison_results = {
                'MyShake-like': {'precision': 0.754, 'recall': 0.920, 'fpr': 0.152, 'f1_score': 0.830},
                'IoT-only': {'precision': 0.951, 'recall': 0.780, 'fpr': 0.018, 'f1_score': 0.857},
                'Unweighted': {'precision': 0.882, 'recall': 0.880, 'fpr': 0.076, 'f1_score': 0.881},
                'Proposed': {'precision': 0.960, 'recall': 0.940, 'fpr': 0.020, 'f1_score': 0.950}
            }
        
        # Generate figures
        print("Generating comparison table...")
        self.plot_comparison_table(
            comparison_results,
            os.path.join(self.output_dir, 'table2_comparison.png')
        )
        
        print("Generating time-of-day threshold...")
        self.plot_time_of_day_threshold(
            os.path.join(self.output_dir, 'threshold_time_of_day.png')
        )
        
        if ablation_results:
            print("Generating ablation study...")
            self.plot_weight_ablation(
                ablation_results,
                os.path.join(self.output_dir, 'figure3_ablation.png')
            )
        
        if resilience_results:
            print("Generating resilience curve...")
            self.plot_resilience_curve(
                resilience_results,
                os.path.join(self.output_dir, 'figure4_resilience.png')
            )
        
        print(f"All figures saved to: {self.output_dir}")
        plt.close('all')
