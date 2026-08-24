#!/usr/bin/env python3
"""
Reproducibility check.

Collects every headline number from the JSON artefacts in ``outputs/`` and
compares it with ``expected_results.json``, which records the values published
in the paper. A clean run means the pipeline on this machine reproduces the
published results.

    python benchmark/check_results.py              # compare
    python benchmark/check_results.py --update     # re-record the expectations
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs')
EXPECTED = os.path.join(HERE, 'expected_results.json')
TOL = 0.011                      # percentage points / absolute


def collect():
    S = json.load(open(os.path.join(OUT, 'simulation.json')))
    L = json.load(open(os.path.join(OUT, 'latency_stress.json')))
    C = json.load(open(os.path.join(OUT, 'calibration_sensitivity.json')))
    M = json.load(open(os.path.join(OUT, 'model_analysis.json')))
    D = json.load(open(os.path.join(OUT, 'deployed_tflite.json')))
    P = json.load(open(os.path.join(OUT, 'packet_format.json')))
    A, AS = S['ablation'], L['ablation_stress']
    SY, SP, SPS = S['system_comparison'], S['iot_sparse'], L['iot_sparse_stress']
    hp, st = S['hyperparameters'], L['stream_simulation']
    v = {}

    # ---- detector, deployed 188 KiB INT8 artefact
    for k in ('precision', 'recall', 'f1', 'fpr', 'accuracy'):
        v[f'detector.crossdomain.{k}'] = 100 * D['test_crossdomain'][k]
        v[f'detector.indomain.{k}'] = 100 * D['val'][k]
    v['detector.crossdomain.auc'] = D['test_crossdomain']['auc']
    v['detector.indomain.auc'] = D['val']['auc']
    for k in ('tp', 'tn', 'fp', 'fn'):
        v[f'detector.crossdomain.{k}'] = D['test_crossdomain'][k]
    v['detector.tflite_kib'] = D['tflite_kib']
    v['model.total_params'] = M['total_params']
    v['model.best_epoch'] = M['training']['best_epoch_by_val_loss']
    v['model.generalisation_gap_pp'] = 100 * M['training']['generalisation_gap_acc']
    v['model.int8_f1_drop_pp_val'] = M['quantisation']['f1_drop_pp_val']
    v['report.total_bytes'] = P['total_bytes']

    # ---- consensus ablation, both operating points
    for tag, tbl in (('ref', A), ('stress', AS)):
        for name, key in (('full', 'Full system'),
                          ('no_spatial', 'w/o Spatial clustering'),
                          ('no_temporal', 'w/o Temporal windowing'),
                          ('no_graph', 'w/o Spatio-temporal graph'),
                          ('no_threshold', 'w/o Adaptive thresholding'),
                          ('no_anchor', 'w/o IoT anchor validation'),
                          ('no_personalised', 'w/o Personalised weights (class w)'),
                          ('no_weighting', 'w/o Reliability weighting'),
                          ('majority', 'Baseline: unweighted majority vote')):
            for m in ('precision', 'recall', 'f1', 'fpr'):
                v[f'ablation.{tag}.{name}.{m}'] = tbl[key][m]['mean']

    # ---- system comparison
    for name, key in (('mobile_only', 'Mobile-only (190 phones)'),
                      ('iot_only', 'IoT-only (10 anchors)'),
                      ('unweighted', 'Unweighted hybrid'),
                      ('class_weights', 'Weighted hybrid, fixed class weights'),
                      ('plus_graph', '  + spatio-temporal graph'),
                      ('proposed', '  + personalised weights (proposed)')):
        for m in ('recall', 'fpr', 'f1'):
            v[f'system.{name}.{m}'] = SY[key][m]['mean']

    # ---- anchor-free operation
    for tag, tbl, keys in (
        ('ref', SP, ('Mobile-only, no graph, class weights', 'Mobile-only + ST-graph',
                     'Mobile-only + ST-graph + personalised')),
        ('stress', SPS, ('Mobile-only, class weights', 'Mobile-only + ST-graph',
                         'Mobile-only + ST-graph + personalised'))):
        for name, key in zip(('base', 'graph', 'personalised'), keys):
            for m in ('recall', 'fpr', 'f1'):
                v[f'anchorfree.{tag}.{name}.{m}'] = tbl[key][m]['mean']

    # ---- nuisance-hardness sweep
    for q, row in S['nuisance_stress'].items():
        for name in ('mobile_only', 'mobile_only_graph', 'hybrid_fixed', 'proposed'):
            v[f'stress.q{q}.{name}.fpr'] = row[name]['fpr']['mean']

    # ---- recall breakdown
    for k, r in L['recall_cumulative'].items():
        v[f'recall.cumulative.{k}'] = r['recall']
        v[f'recall.cumulative.{k}.n'] = r['n']
    for k, r in S['recall_by_magnitude'].items():
        v[f'recall.bin.{k}'] = r['recall']
    v['recall.gutenberg_richter'] = S['gutenberg_richter_catalogue']['recall']['mean']

    # ---- calibration sensitivity
    for lev, row in C.items():
        tag = lev.split()[0]
        v[f'calibration.{tag}.mobile.fpr'] = 100 * row['mobile']['fpr']
        v[f'calibration.{tag}.mobile.recall'] = 100 * row['mobile']['recall']
        v[f'calibration.{tag}.anchor.fpr'] = 100 * row['anchor']['fpr']
    for lev, row in S['calibration_system'].items():
        v[f'calibration.{lev.split()[0]}.system.f1'] = row['fixed_weights']['f1']['mean']

    # ---- latency, resilience, hyper-parameters
    v['latency.mean'] = L['alert_latency_s']['mean']
    v['latency.std'] = L['alert_latency_s']['std']
    v['latency.p95'] = L['alert_latency_s']['p95']
    v['latency.n_events'] = L['alert_latency_s']['n']
    v['latency.blind_zone_km'] = L['blind_zone_km']
    for k, row in L['latency_budget_s'].items():
        v[f'latency.budget.{k}'] = row['mean']
    for d, row in L['warning_lead_time_s'].items():
        v[f'leadtime.{d}'] = row['mean']
    for pl, row in S['packet_loss'].items():
        v[f'packetloss.{pl}.recall'] = row['recall']['mean']
        v[f'packetloss.{pl}.fpr'] = row['fpr']['mean']
    v['hyper.dbscan.best_f1'] = max(x['f1']['mean'] for x in hp['dbscan'].values())
    v['hyper.dbscan.min_f1'] = min(x['f1']['mean'] for x in hp['dbscan'].values())
    v['hyper.dbscan.operating_f1'] = hp['dbscan']['eps=5,minPts=3']['f1']['mean']
    v['hyper.iot_weight.span_f1'] = (max(x['f1']['mean'] for x in hp['iot_weight'].values())
                                     - min(x['f1']['mean'] for x in hp['iot_weight'].values()))
    v['hyper.iot_weight.f1'] = hp['iot_weight']['0.7']['f1']['mean']
    v['hyper.temporal.2s_f1'] = hp['temporal']['2.0']['f1']['mean']
    v['hyper.rho_min.0.fpr'] = hp['rho_min']['0.0']['fpr']['mean']
    v['hyper.rho_min.1.fpr'] = hp['rho_min']['1.0']['fpr']['mean']
    v['hyper.v_app_min.1.5.fpr'] = hp['v_app_min']['1.5']['fpr']['mean']
    v['hyper.v_app_min.4.fpr'] = hp['v_app_min']['4.0']['fpr']['mean']

    # ---- Tier-1 traffic, residual false alarms, learned reliability, stream
    v['tier1.eq_reports_per_event'] = S['tier1_report_stats']['eq']['mean']
    v['tier1.nuisance_reports_per_event'] = S['tier1_report_stats']['nz']['mean']
    for cls, row in S['fp_by_nuisance_class'].items():
        v[f'falsealarm.{cls.split()[0].lower()}.rate'] = row['fpr']
    rel = S['learned_reliability']['mobile']
    for k in ('min', 'median', 'max'):
        v[f'reliability.mobile.{k}'] = rel[k]
    v['stream.true_positives'] = st['true_positives']
    v['stream.false_positives'] = st['false_positives']
    v['stream.mean_latency'] = st['avg_latency']
    return v


def main():
    cur = collect()
    if '--update' in sys.argv:
        json.dump(cur, open(EXPECTED, 'w'), indent=1, sort_keys=True)
        print(f'recorded {len(cur)} expected values -> {EXPECTED}')
        return 0
    exp = json.load(open(EXPECTED))
    bad, missing = [], []
    for k, e in sorted(exp.items()):
        if k not in cur:
            missing.append(k); continue
        c = cur[k]
        tol = TOL if abs(e) > 0.02 else 0.002
        if abs(c - e) > max(tol, abs(e) * 1e-4):
            bad.append((k, e, c))
    extra = sorted(set(cur) - set(exp))
    print(f'{len(exp)} expected values, {len(bad)} mismatched, '
          f'{len(missing)} missing, {len(extra)} new')
    for k, e, c in bad:
        print(f'  MISMATCH {k}: expected {e}, got {c}')
    for k in missing:
        print(f'  MISSING  {k}')
    if bad or missing:
        return 1
    print('this run reproduces the published results')
    return 0


if __name__ == '__main__':
    sys.exit(main())
