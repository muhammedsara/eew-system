#!/usr/bin/env python3
"""
Hybrid Mobile-IoT Earthquake Early Warning System

Main entry point for running simulations, experiments, and dashboard.

Based on the research paper:
"Leveraging Smartphone Sensors for Earthquake Early Warning: 
A Weighted Spatiotemporal Consensus Protocol for Advanced Hybrid 
Mobile-IoT Seismic Networks"

Author: Muhammed Şara
Usage:
    python main.py --mode simulation     # Run full simulation
    python main.py --mode dashboard      # Start web dashboard
    python main.py --mode demo           # Quick demo
    python main.py --mode ablation       # Run ablation study
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from utils.logger import get_logger

logger = get_logger('Main')


def run_demo():
    """Run a quick demonstration"""
    print("\n" + "="*60)
    print("🌍 DEPREM ERKEN UYARI SİSTEMİ - DEMO")
    print("="*60 + "\n")
    
    from simulation.earthquake_generator import EarthquakeGenerator
    from simulation.false_positive_generator import FalsePositiveGenerator
    from devices.device_manager import DeviceManager
    from consensus.engine import ConsensusEngine
    
    # Initialize components
    generator = EarthquakeGenerator(seed=42)
    device_manager = DeviceManager()
    consensus_engine = ConsensusEngine()
    
    # Generate earthquake scenario
    print("📍 Deprem senaryosu oluşturuluyor...")
    earthquake = generator.generate_scenario(
        magnitude=5.8,
        epicenter_lat=40.7,
        epicenter_lon=30.5
    )
    print(f"   Büyüklük: M{earthquake.magnitude:.1f}")
    print(f"   Merkez: {earthquake.epicenter_lat:.2f}°N, {earthquake.epicenter_lon:.2f}°E")
    print(f"   Etki yarıçapı: {earthquake.affected_radius_km:.1f} km")
    
    # Setup devices - use realistic city radius (50km) not seismic affected radius
    print("\n📱 Cihazlar dağıtılıyor...")
    distribution_radius_km = 50.0  # Realistic city/urban area radius
    device_manager.setup_scenario(
        center_lat=earthquake.epicenter_lat,
        center_lon=earthquake.epicenter_lon,
        num_mobile=100,
        num_iot=10,
        distribution_radius_km=distribution_radius_km
    )
    stats = device_manager.get_statistics()
    print(f"   Mobil cihazlar: {stats['num_mobile_devices']}")
    print(f"   IoT istasyonları: {stats['num_iot_anchors']}")
    
    # Simulate triggers
    print("\n⚡ Tetikleyiciler simüle ediliyor...")
    triggers = device_manager.simulate_earthquake_triggers(
        earthquake_data=earthquake.waveform_data,
        epicenter_lat=earthquake.epicenter_lat,
        epicenter_lon=earthquake.epicenter_lon,
        magnitude=earthquake.magnitude
    )
    print(f"   Tetiklenen cihaz sayısı: {len(triggers)}")
    
    # Run consensus
    print("\n🔍 Konsensüs protokolü çalıştırılıyor...")
    trigger_dicts = [t.to_dict() for t in triggers]
    decision = consensus_engine.process(trigger_dicts)
    
    print(f"\n{'='*60}")
    print(f"{'DEPREM TESPİT EDİLDİ! 🚨' if decision.is_earthquake else 'Deprem tespit edilmedi'}")
    print(f"{'='*60}")
    print(f"   Skor: {decision.score:.3f}")
    print(f"   Eşik: {decision.threshold:.3f}")
    print(f"   Mobil oy: {decision.mobile_count}")
    print(f"   IoT oy: {decision.iot_count}")
    print(f"   İşlem süresi: {decision.processing_time_ms:.2f} ms")
    
    if decision.estimated_lat and decision.estimated_lon:
        print(f"   Tahmini merkez: {decision.estimated_lat:.2f}°N, {decision.estimated_lon:.2f}°E")
        print(f"   Tahmini yarıçap: {decision.estimated_radius_km:.1f} km")
    
    print("\n✅ Demo tamamlandı!\n")


def run_simulation(n_earthquakes=100, n_false_positives=500):
    """Run full simulation"""
    print("\n" + "="*60)
    print("📊 TAM SİMÜLASYON")
    print("="*60 + "\n")
    
    from simulation.trace_simulator import TraceSimulator
    from evaluation.metrics import MetricsCalculator
    
    simulator = TraceSimulator()
    
    print(f"Deprem senaryoları: {n_earthquakes}")
    print(f"Yanlış pozitif senaryoları: {n_false_positives}")
    print()
    
    summary = simulator.run_simulation(
        n_earthquakes=n_earthquakes,
        n_false_positives=n_false_positives,
        show_progress=True
    )
    
    print("\n" + "="*60)
    print("SONUÇLAR")
    print("="*60)
    print(f"Total scenarios: {summary.total_scenarios}")
    print(f"TP: {summary.true_positives}, FP: {summary.false_positives}")
    print(f"TN: {summary.true_negatives}, FN: {summary.false_negatives}")
    print("-"*60)
    print(f"Precision: {summary.precision:.2%}")
    print(f"Recall (TPR): {summary.recall:.2%}")
    print(f"False Positive Rate: {summary.false_positive_rate:.2%}")
    print(f"F1-Score: {summary.f1_score:.3f}")
    print(f"Accuracy: {summary.accuracy:.2%}")
    print("-"*60)
    print(f"Avg processing time: {summary.avg_processing_time_ms:.2f} ms")
    print(f"Total simulation time: {summary.total_simulation_time_s:.2f} s")
    print("="*60 + "\n")
    
    return summary


def run_ablation_study():
    """Run weight ablation study"""
    print("\n" + "="*60)
    print("🔬 ABLASYON ÇALIŞMASI (Weight Optimization)")
    print("="*60 + "\n")
    
    from simulation.trace_simulator import TraceSimulator
    from visualization.plots import ResultsVisualizer
    
    simulator = TraceSimulator()
    
    # Test weight pairs
    weight_pairs = [
        (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
        (0.4, 0.6), (0.5, 0.5), (0.6, 0.4),
        (0.7, 0.3), (0.8, 0.2)
    ]
    
    results = simulator.run_ablation_study(
        weight_pairs=weight_pairs,
        n_earthquakes=50,
        n_false_positives=250
    )
    
    # Find optimal
    best_key = max(results, key=lambda k: results[k].f1_score)
    best = results[best_key]
    
    print("\n" + "="*60)
    print(f"OPTİMAL AĞIRLIKLAR: {best_key}")
    print(f"F1-Score: {best.f1_score:.3f}")
    print(f"TPR: {best.recall:.2%}, FPR: {best.false_positive_rate:.2%}")
    print("="*60 + "\n")
    
    # Generate visualization
    visualizer = ResultsVisualizer()
    ablation_data = {k: v.to_dict()['metrics'] for k, v in results.items()}
    visualizer.plot_weight_ablation(ablation_data)
    print("Görselleştirme kaydedildi.\n")
    
    return results


def run_dashboard(host='0.0.0.0', port=8080):
    """Start web dashboard"""
    print("\n" + "="*60)
    print("🖥️  WEB DASHBOARD BAŞLATILIYOR")
    print("="*60)
    
    from dashboard.app import run_dashboard as start_dashboard
    start_dashboard(host=host, port=port)


def run_baseline_comparison():
    """Compare all system configurations on the SAME scenarios.
    
    Runs four configurations through the simulation:
      1. Mobile-only (190 phones, no IoT validation)
      2. IoT-only (10 anchors)
      3. Unweighted hybrid (0.5/0.5)
      4. Weighted hybrid (0.3/0.7) — proposed
    """
    print("\n" + "="*60)
    print("📈 BASELINE KARŞILAŞTIRMASI (Gerçek Simülasyon)")
    print("="*60 + "\n")

    from simulation.trace_simulator import TraceSimulator

    configs = {
        'Mobile-only (190 phones)': {
            'disable_iot_validation': True,
            'disable_weighting': True,
        },
        'IoT-only (10 anchors)': {
            'disable_weighting': True,
            # Only IoT triggers considered — handled by filtering in engine
        },
        'Unweighted hybrid': {
            'disable_weighting': True,
        },
        'Weighted hybrid (proposed)': {},  # Full system
    }

    seed = 42
    n_eq = 100
    n_fp = 500
    results = {}

    for name, flags in configs.items():
        print(f"\n  ▶ {name}")
        sim = TraceSimulator(seed=seed)

        for flag_name, flag_val in flags.items():
            setattr(sim.consensus_engine, flag_name, flag_val)

        summary = sim.run_simulation(
            n_earthquakes=n_eq,
            n_false_positives=n_fp,
            show_progress=False
        )

        results[name] = {
            'precision': summary.precision,
            'recall': summary.recall,
            'fpr': summary.false_positive_rate,
            'f1_score': summary.f1_score,
        }
        print(f"    Recall={summary.recall:.1%}  "
              f"FPR={summary.false_positive_rate:.2%}  "
              f"F1={summary.f1_score:.3f}")

    # Print comparison table
    print(f"\n{'-'*70}")
    print(f"{'Method':<30} {'Recall':>10} {'FPR':>10} {'F1':>10}")
    print(f"{'-'*70}")
    for name, m in results.items():
        print(f"{name:<30} {m['recall']*100:>9.1f}%"
              f" {m['fpr']*100:>9.2f}% {m['f1_score']:>10.3f}")
    print(f"{'-'*70}\n")

    return results


def run_hyperparameter_search():
    """Run DBSCAN grid search + temporal window sensitivity (Paper §6.3)."""
    from evaluation.hyperparameter_search import (
        run_dbscan_grid_search,
        run_temporal_window_sensitivity,
        save_results,
    )

    print("\n" + "="*60)
    print("📐 DBSCAN GRID SEARCH")
    print("="*60)
    dbscan = run_dbscan_grid_search()
    save_results([r.to_dict() for r in dbscan], "dbscan_grid_search.json")

    print("\n" + "="*60)
    print("⏱  TEMPORAL WINDOW SENSITIVITY")
    print("="*60)
    tw = run_temporal_window_sensitivity()
    save_results([r.to_dict() for r in tw], "temporal_window_sensitivity.json")


def run_consensus_ablation_study():
    """Run full consensus ablation study (Paper Table 4, §6.2)."""
    from evaluation.hyperparameter_search import (
        run_consensus_ablation, save_results
    )

    results = run_consensus_ablation(n_seeds=5)
    save_results(results, "consensus_ablation.json")


def run_continuous_simulation():
    """Run 600 s continuous simulation (Paper §6.4)."""
    from simulation.continuous_simulation import ContinuousSimulator
    import json
    from pathlib import Path

    sim = ContinuousSimulator()
    result = sim.run()

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "continuous_simulation.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print(f"  💾 Saved → {out_dir / 'continuous_simulation.json'}")



def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Mobile-IoT Earthquake Early Warning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  demo               Quick demonstration with single earthquake
  simulation         Full simulation (100 EQ + 500 FP)
  dashboard          Start interactive web dashboard
  ablation           Run weight ablation study
  compare            Compare with baseline methods (actual simulation)
  hyperparameters    DBSCAN grid search + temporal window sensitivity
  consensus-ablation Per-component consensus ablation (Paper Table 4)
  continuous         600 s continuous simulation (Paper §6.4)
  figures            Generate all paper figures
        """
    )

    parser.add_argument(
        '--mode', '-m',
        choices=[
            'demo', 'simulation', 'dashboard', 'ablation',
            'compare', 'hyperparameters', 'consensus-ablation',
            'continuous', 'figures'
        ],
        default='demo',
        help='Operation mode'
    )

    parser.add_argument('--earthquakes', '-e', type=int, default=100)
    parser.add_argument('--false-positives', '-f', type=int, default=500)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)

    args = parser.parse_args()

    mode_map = {
        'demo': run_demo,
        'simulation': lambda: run_simulation(args.earthquakes, args.false_positives),
        'dashboard': lambda: run_dashboard(host=args.host, port=args.port),
        'ablation': run_ablation_study,
        'compare': run_baseline_comparison,
        'hyperparameters': run_hyperparameter_search,
        'consensus-ablation': run_consensus_ablation_study,
        'continuous': run_continuous_simulation,
        'figures': lambda: (
            __import__('visualization.plots', fromlist=['ResultsVisualizer'])
            .ResultsVisualizer().generate_all_figures()
        ),
    }

    handler = mode_map.get(args.mode)
    if handler:
        handler()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

