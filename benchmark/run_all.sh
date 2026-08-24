#!/usr/bin/env bash
# Reproduce every number and figure in the paper.
# Needs no external download: the evaluation windows and the deployed detector
# ship with the repository. About 20 min on 8 CPU cores.
set -e
cd "$(dirname "$0")"
echo "== 1/7  device decision tables (36 calibration profiles x 4 waveform pools)"
python3 -u 01_build_device_tables.py
echo "== 2/7  consensus ablation, system comparison, hyper-parameter sweeps"
python3 -u 02_run_experiments.py
echo "== 3/7  latency budget, stress-point ablation, weight variants"
python3 -u 03_latency_and_stress.py
echo "== 4/7  model architecture, training curves, quantisation"
python3 -u 04_model_analysis.py
echo "== 5/7  deployed 188 KiB artefact"
python3 -u 05_deployed_model_check.py
echo "== 6/7  protocol figures"
python3 -u 06_make_figures.py
echo "== 7/7  performance figures"
python3 -u 07_performance_figures.py
echo "== reproducibility check"
python3 check_results.py
