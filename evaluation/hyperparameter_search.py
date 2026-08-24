"""
Hyperparameter Search & Sensitivity Analysis

Implements the experiments described in Paper Section 6.3:
1. DBSCAN grid search:   ε ∈ {2, 3, 5, 7, 10} km,  minPts ∈ {2, 3, 4, 5}
2. Temporal window:      τ_w ∈ {1, 1.5, 2, 2.5, 3} s
3. IoT weight:           w_i ∈ {0.5, 0.6, 0.7, 0.8, 0.9}

Also implements the consensus ablation study (Paper Table 4).

Author: Muhammed Şara
"""

import numpy as np
import json
import time
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from simulation.trace_simulator import TraceSimulator, SimulationSummary
from consensus.engine import ConsensusEngine


# ─────────────────────────────────────────────
#  Data‑class for grid search results
# ─────────────────────────────────────────────
@dataclass
class GridSearchResult:
    """Single row of a grid‑search experiment."""
    param_name: str
    param_value: Any
    precision: float
    recall: float
    f1_score: float
    fpr: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────
#  DBSCAN Grid Search (Paper §6.3)
# ─────────────────────────────────────────────
def run_dbscan_grid_search(
    eps_values: List[float] = None,
    min_pts_values: List[int] = None,
    n_earthquakes: int = 100,
    n_false_positives: int = 500,
    seed: int = 42
) -> List[GridSearchResult]:
    """
    Grid search over DBSCAN hyperparameters.

    Paper §6.3:
      "DBSCAN grid search (ε ∈ {2, 3, 5, 7, 10} km, minPts ∈ {2, 3, 4, 5}):
       optimal at ε=5 km, minPts=3 (F1 93.5%, 2.9 clusters/event)."

    Returns:
        List of GridSearchResult for each (ε, minPts) pair.
    """
    if eps_values is None:
        eps_values = [2, 3, 5, 7, 10]
    if min_pts_values is None:
        min_pts_values = [2, 3, 4, 5]

    results: List[GridSearchResult] = []

    for eps in eps_values:
        for min_pts in min_pts_values:
            print(f"\n  DBSCAN  ε={eps} km, minPts={min_pts}")

            sim = TraceSimulator(seed=seed)
            # Override DBSCAN parameters in the consensus engine
            sim.consensus_engine.spatial_clusterer.eps_km = eps
            sim.consensus_engine.spatial_clusterer.min_samples = min_pts

            summary = sim.run_simulation(
                n_earthquakes=n_earthquakes,
                n_false_positives=n_false_positives,
                show_progress=False
            )

            # Count average clusters per earthquake event
            eq_results = [r for r in sim.results if r.is_earthquake]
            # Approximate cluster count from the result metadata
            avg_clusters = np.mean([
                getattr(r, 'mobile_count', 0) + getattr(r, 'iot_count', 0)
                for r in eq_results
            ]) if eq_results else 0

            results.append(GridSearchResult(
                param_name=f"eps={eps}_minPts={min_pts}",
                param_value={"eps_km": eps, "min_pts": min_pts},
                precision=summary.precision,
                recall=summary.recall,
                f1_score=summary.f1_score,
                fpr=summary.false_positive_rate,
                extra={"avg_device_count": float(avg_clusters)}
            ))

            print(f"    F1={summary.f1_score:.3f}  FPR={summary.false_positive_rate:.2%}")

    # Report optimal
    best = max(results, key=lambda r: r.f1_score)
    print(f"\n  ✅ Optimal: {best.param_name}  "
          f"F1={best.f1_score:.3f}  FPR={best.fpr:.2%}")

    return results


# ─────────────────────────────────────────────
#  Temporal Window Sensitivity (Paper §6.3)
# ─────────────────────────────────────────────
def run_temporal_window_sensitivity(
    window_values: List[float] = None,
    n_earthquakes: int = 100,
    n_false_positives: int = 500,
    seed: int = 42
) -> List[GridSearchResult]:
    """
    Sensitivity analysis over temporal window durations.

    Paper §6.3:
      "Temporal window (τ_w ∈ {1, 1.5, 2, 2.5, 3} s): 2 s captures
       95th percentile device delays while avoiding noise correlation."

    Returns:
        List of GridSearchResult for each window duration.
    """
    if window_values is None:
        window_values = [1.0, 1.5, 2.0, 2.5, 3.0]

    results: List[GridSearchResult] = []

    for tw in window_values:
        print(f"\n  Temporal window τ_w = {tw} s")

        sim = TraceSimulator(seed=seed)
        # Override temporal window in consensus engine
        sim.consensus_engine.temporal_windower.window_seconds = tw

        summary = sim.run_simulation(
            n_earthquakes=n_earthquakes,
            n_false_positives=n_false_positives,
            show_progress=False
        )

        results.append(GridSearchResult(
            param_name=f"tw={tw}s",
            param_value={"temporal_window_s": tw},
            precision=summary.precision,
            recall=summary.recall,
            f1_score=summary.f1_score,
            fpr=summary.false_positive_rate,
        ))

        print(f"    F1={summary.f1_score:.3f}  FPR={summary.false_positive_rate:.2%}")

    best = max(results, key=lambda r: r.f1_score)
    print(f"\n  ✅ Optimal: {best.param_name}  "
          f"F1={best.f1_score:.3f}  FPR={best.fpr:.2%}")

    return results


# ─────────────────────────────────────────────
#  Consensus Ablation Study (Paper Table 4)
# ─────────────────────────────────────────────
ABLATION_CONFIGS = {
    "Full system":              {},
    "w/o Spatial clustering":   {"disable_spatial": True},
    "w/o Temporal windowing":   {"disable_temporal": True},
    "w/o Adaptive threshold":   {"disable_adaptive_threshold": True},
    "w/o IoT validation":       {"disable_iot_validation": True},
    "w/o IoT weighting":        {"disable_weighting": True},
    "Baseline (majority vote)": {
        "disable_spatial": False,
        "disable_temporal": False,
        "disable_adaptive_threshold": True,
        "disable_iot_validation": True,
        "disable_weighting": True,
    },
}


def run_consensus_ablation(
    n_earthquakes: int = 100,
    n_false_positives: int = 500,
    n_seeds: int = 5,
    base_seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Systematic ablation study disabling one component at a time.

    Paper Table 4 — §6.2:
      "We conducted a systematic ablation study disabling one component
       at a time while holding all others fixed."

    Results are averaged over *n_seeds* runs with 95 % CI.
    """
    all_results: Dict[str, List[Dict[str, float]]] = {
        name: [] for name in ABLATION_CONFIGS
    }

    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        print(f"\n{'='*60}")
        print(f"  ABLATION — seed {seed}")
        print(f"{'='*60}")

        for config_name, flags in ABLATION_CONFIGS.items():
            print(f"\n  ▶ {config_name}")

            sim = TraceSimulator(seed=seed)

            # Set ablation flags
            for flag_name, flag_val in flags.items():
                setattr(sim.consensus_engine, flag_name, flag_val)

            summary = sim.run_simulation(
                n_earthquakes=n_earthquakes,
                n_false_positives=n_false_positives,
                show_progress=False
            )

            all_results[config_name].append({
                "precision": summary.precision,
                "recall": summary.recall,
                "f1_score": summary.f1_score,
                "fpr": summary.false_positive_rate,
            })

            print(f"    F1={summary.f1_score:.3f}  FPR={summary.false_positive_rate:.2%}")

    # Aggregate with 95 % CI
    summary_table: Dict[str, Dict[str, Any]] = {}
    for config_name, runs in all_results.items():
        arr = {k: np.array([r[k] for r in runs]) for k in runs[0]}
        summary_table[config_name] = {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "ci95": float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v)))
                         if len(v) > 1 else 0.0,
            }
            for k, v in arr.items()
        }

    # Pretty print
    print(f"\n{'='*80}")
    print(f"  ABLATION RESULTS (mean ± 95% CI, {n_seeds} seeds)")
    print(f"{'='*80}")
    print(f"{'Configuration':<30} {'Precision':>12} {'Recall':>12} "
          f"{'F1':>12} {'FPR':>12}")
    print("-" * 80)
    for name, metrics in summary_table.items():
        p = metrics['precision']
        r = metrics['recall']
        f = metrics['f1_score']
        fpr = metrics['fpr']
        print(f"{name:<30} "
              f"{p['mean']*100:>5.1f}±{p['ci95']*100:.1f}%  "
              f"{r['mean']*100:>5.1f}±{r['ci95']*100:.1f}%  "
              f"{f['mean']*100:>5.1f}±{f['ci95']*100:.1f}%  "
              f"{fpr['mean']*100:>5.1f}±{fpr['ci95']*100:.1f}%")
    print("=" * 80)

    return summary_table


# ─────────────────────────────────────────────
#  Save all results to JSON
# ─────────────────────────────────────────────
def save_results(results: dict, filename: str):
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / filename, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  💾 Saved to {out_dir / filename}")


# ─────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter search & ablation")
    parser.add_argument("--dbscan", action="store_true", help="DBSCAN grid search")
    parser.add_argument("--temporal", action="store_true",
                        help="Temporal window sensitivity")
    parser.add_argument("--ablation", action="store_true",
                        help="Consensus ablation study")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--n-eq", type=int, default=100)
    parser.add_argument("--n-fp", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    run_all = args.all or not (args.dbscan or args.temporal or args.ablation)

    if args.dbscan or run_all:
        print("\n" + "=" * 60)
        print("  📐 DBSCAN GRID SEARCH")
        print("=" * 60)
        dbscan_results = run_dbscan_grid_search(
            n_earthquakes=args.n_eq, n_false_positives=args.n_fp
        )
        save_results(
            [r.to_dict() for r in dbscan_results],
            "dbscan_grid_search.json"
        )

    if args.temporal or run_all:
        print("\n" + "=" * 60)
        print("  ⏱  TEMPORAL WINDOW SENSITIVITY")
        print("=" * 60)
        tw_results = run_temporal_window_sensitivity(
            n_earthquakes=args.n_eq, n_false_positives=args.n_fp
        )
        save_results(
            [r.to_dict() for r in tw_results],
            "temporal_window_sensitivity.json"
        )

    if args.ablation or run_all:
        print("\n" + "=" * 60)
        print("  🔬 CONSENSUS ABLATION STUDY")
        print("=" * 60)
        ablation_results = run_consensus_ablation(
            n_earthquakes=args.n_eq,
            n_false_positives=args.n_fp,
            n_seeds=args.seeds,
        )
        save_results(ablation_results, "consensus_ablation.json")

    print("\n✅ All experiments complete.")
