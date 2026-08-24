#!/usr/bin/env python3
"""
Step 2 -- run every simulation experiment and dump the raw numbers.

Tier-1 detection reports are generated once per (seed, packet-loss) and then
replayed through every consensus configuration, so the ablation isolates the
gateway logic exactly.
"""
import os, sys, json, time, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sim_config import (REGION, GM, TRG, NET, CON, SC, CALIBRATION_LEVELS,
                         FIELD_MIX_MOBILE, NUISANCE_CLASSES)
from network_simulator import (DeviceTables, build_scenarios, build_devices, build_hubs,
                       tier1_reports, ConsensusEngine, Flags)

OUT = os.path.join(HERE, 'outputs')
TAB = DeviceTables(os.path.join(OUT, 'device_tables.npz'))


# --------------------------------------------------------------------- utils
def confusion(recs):
    tp = sum(1 for r in recs if r['is_eq'] and r['det'])
    fn = sum(1 for r in recs if r['is_eq'] and not r['det'])
    fp = sum(1 for r in recs if not r['is_eq'] and r['det'])
    tn = sum(1 for r in recs if not r['is_eq'] and not r['det'])
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=100 * pr, recall=100 * rc,
                f1=100 * (2 * pr * rc / max(pr + rc, 1e-12)),
                fpr=100 * fp / max(fp + tn, 1),
                accuracy=100 * (tp + tn) / max(len(recs), 1))


def agg(per_seed, keys=('precision', 'recall', 'f1', 'fpr')):
    o = {}
    for k in keys:
        v = np.array([s[k] for s in per_seed], dtype=float)
        m = v.mean()
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        o[k] = {'mean': float(m), 'ci95': float(ci)}
    for k in ('tp', 'fp', 'tn', 'fn'):
        o[k] = int(np.sum([s[k] for s in per_seed]))
    return o


def gen_burnin(seed, q_hard=0.0, n_repeat=4):
    """
    Operational warm-up trace: the same fleet and the same urban hubs, but an
    independent, longer nuisance-dominated catalogue.  Used only to let the
    per-device reliability estimator converge before evaluation.
    """
    rng = np.random.default_rng(seed + 900000)
    hubs = build_hubs(np.random.default_rng(seed))      # same city layout
    devs = build_devices(TAB, np.random.default_rng(seed), hubs)
    scen, hours = [], []
    for _ in range(n_repeat):
        sc = build_scenarios(rng, hubs)
        scen += sc
    hours = rng.integers(0, 24, len(scen))
    for i, s in enumerate(scen):
        if not s.is_eq:
            hours[i] = int(rng.integers(7, 21))
    reps = [tier1_reports(s, devs, TAB, rng, 0.0, q_hard) for s in scen]
    return scen, hours, reps


def gen_reports(seed, packet_loss, mobile_level=None, q_hard=0.0, mag_dist=None):
    """Tier-1 pass: identical device decisions shared by all consensus variants."""
    rng = np.random.default_rng(seed)
    hubs = build_hubs(rng)
    scen = build_scenarios(rng, hubs, mag_dist)
    devs = build_devices(TAB, rng, hubs)
    if mobile_level is not None:
        pool = TAB.by_level[mobile_level]
        for d in devs:
            if d.kind == 'mobile':
                d.profile = int(rng.choice(pool))
                d.clock_sigma_s = float(np.hypot(NET.clock_jitter_s, TAB.clock[d.profile]))
    hours = rng.integers(0, 24, len(scen))
    # nuisance sources are concentrated in the active part of the day
    for i, s in enumerate(scen):
        if not s.is_eq:
            hours[i] = int(rng.integers(7, 21))
    reps = [tier1_reports(s, devs, TAB, rng, packet_loss, q_hard) for s in scen]
    return scen, devs, hours, reps


def run_consensus(scen, devs, hours, reps, flags, cfg=CON, burnin=None):
    """
    Replay one Tier-1 trace through the gateway.

    ``burnin`` is an optional (scen, hours, reports) triple representing an
    operational warm-up period.  The adaptive reliability estimator is trained
    on it, but none of its decisions enter the reported metrics -- exactly the
    protocol a deployed gateway would follow after installation.
    """
    eng = ConsensusEngine([__import__('copy').copy(d) for d in devs], flags, cfg)
    if burnin is not None:
        bs, bh, br = burnin
        for s, h, rp in zip(bs, bh, br):
            eng.process(rp, int(h), s.t0)
    recs, meta = [], []
    for s, h, rp in zip(scen, hours, reps):
        o = eng.process(rp, int(h), s.t0)
        recs.append({'is_eq': s.is_eq, 'det': o['detected'],
                     'mag': s.magnitude, 'nuis': s.nuisance})
        meta.append(o)
    return recs, meta, eng


# ================================================================== EXPERIMENTS
RESULTS = {}
T0 = time.time()

Q_REF = 0.10          # reference nuisance-hardness operating point (Sec. 5.4)

BASE_CACHE, BURN_CACHE = {}, {}
def cache(seed, pl=0.0, lev=None, q_hard=Q_REF, mag_dist=None):
    k = (seed, pl, lev, q_hard, mag_dist)
    if k not in BASE_CACHE:
        BASE_CACHE[k] = gen_reports(seed, pl, lev, q_hard, mag_dist)
    return BASE_CACHE[k]

def burn(seed, q_hard=Q_REF):
    if (seed, q_hard) not in BURN_CACHE:
        BURN_CACHE[(seed, q_hard)] = gen_burnin(seed, q_hard)
    return BURN_CACHE[(seed, q_hard)]


def evaluate(flags, seeds=SC.seeds, pl=0.0, lev=None, q_hard=Q_REF,
             mag_dist=None, cfg=CON, use_burnin=None, extra=False):
    if use_burnin is None:
        use_burnin = not (flags.no_adaptive_weights or flags.no_weighting
                          or flags.majority_vote)
    per, meta_all, recs_all = [], [], []
    for seed in seeds:
        scen, devs, hours, reps = cache(seed, pl, lev, q_hard, mag_dist)
        bi = burn(seed, q_hard) if use_burnin else None
        recs, meta, eng = run_consensus(scen, devs, hours, reps, flags, cfg, bi)
        per.append(confusion(recs))
        if extra:
            meta_all.append((scen, recs, meta, eng))
    out = agg(per)
    return (out, meta_all) if extra else out


# ---------------------------------------------------- 1. Tier-1 report statistics
print('[1/10] Tier-1 report statistics ...', flush=True)
st = {'eq': [], 'eq_mob': [], 'eq_iot': [], 'nz': [], 'by_class': {}}
for seed in SC.seeds:
    scen, devs, hours, reps = cache(seed)
    for s, r in zip(scen, reps):
        nm = sum(1 for x in r if x['kind'] == 'mobile'); ni = len(r) - nm
        if s.is_eq:
            st['eq'].append(len(r)); st['eq_mob'].append(nm); st['eq_iot'].append(ni)
        else:
            st['nz'].append(len(r))
            st['by_class'].setdefault(s.nuisance, []).append(len(r))
RESULTS['tier1_report_stats'] = {
    **{k: {'mean': float(np.mean(v)), 'median': float(np.median(v)),
           'p95': float(np.percentile(v, 95)), 'max': int(np.max(v))}
       for k, v in st.items() if k != 'by_class'},
    'by_nuisance_class': {k: {'mean': float(np.mean(v)), 'p95': float(np.percentile(v, 95))}
                          for k, v in st['by_class'].items()}}
print('       EQ reports/event mean %.1f  nuisance mean %.2f'
      % (np.mean(st['eq']), np.mean(st['nz'])), flush=True)


# ------------------------------------------------------------- 2. ablation
ABLATIONS = [
    ('Full system',                 Flags()),
    ('w/o Spatial clustering',             Flags(no_spatial=True)),
    ('w/o Temporal windowing',             Flags(no_temporal=True)),
    ('w/o Spatio-temporal graph',          Flags(no_graph=True)),
    ('w/o Adaptive thresholding',          Flags(no_adaptive_thr=True)),
    ('w/o IoT anchor validation',          Flags(no_iot_validation=True)),
    ('w/o Personalised weights (class w)', Flags(no_adaptive_weights=True)),
    ('w/o Reliability weighting',          Flags(no_weighting=True)),
    ('Baseline: unweighted majority vote', Flags(majority_vote=True, no_graph=True,
                                                 no_iot_validation=True,
                                                 no_adaptive_thr=True)),
]
print('[2/10] Consensus ablation ...', flush=True)
abl = {}
for name, fl in ABLATIONS:
    abl[name] = evaluate(fl)
    print(f"       {name:<38} F1={abl[name]['f1']['mean']:6.2f}+-{abl[name]['f1']['ci95']:.2f}"
          f"  P={abl[name]['precision']['mean']:6.2f}"
          f"  R={abl[name]['recall']['mean']:6.2f}  FPR={abl[name]['fpr']['mean']:5.2f}", flush=True)
RESULTS['ablation'] = abl


# ------------------------------------------------- 3. system-level comparison
SYSTEMS = [
    ('Mobile-only (190 phones)',             Flags(mobile_only=True, no_iot_validation=True,
                                                   no_weighting=True, no_graph=True)),
    ('IoT-only (10 anchors)',                Flags(iot_only=True, no_weighting=True,
                                                   no_graph=True)),
    ('Unweighted hybrid',                    Flags(no_weighting=True, no_graph=True)),
    ('Weighted hybrid, fixed class weights', Flags(no_adaptive_weights=True, no_graph=True)),
    ('  + spatio-temporal graph',            Flags(no_adaptive_weights=True)),
    ('  + personalised weights (proposed)',  Flags()),
]
print('[3/10] System-level comparison ...', flush=True)
sysres = {}
for name, fl in SYSTEMS:
    sysres[name] = evaluate(fl)
    print(f"       {name:<40} R={sysres[name]['recall']['mean']:6.2f}"
          f"  FPR={sysres[name]['fpr']['mean']:5.2f}  F1={sysres[name]['f1']['mean']:6.2f}", flush=True)
RESULTS['system_comparison'] = sysres


# ------------------------------------- 4. IoT-sparse / mobile-only deployment
print('[4/10] IoT-sparse deployment (graph + personalised weights) ...', flush=True)
SPARSE = [
    ('Mobile-only, no graph, class weights', Flags(mobile_only=True, no_iot_validation=True,
                                                   no_graph=True, no_adaptive_weights=True)),
    ('Mobile-only + ST-graph',               Flags(mobile_only=True, no_iot_validation=True,
                                                   no_adaptive_weights=True)),
    ('Mobile-only + ST-graph + personalised', Flags(mobile_only=True, no_iot_validation=True)),
]
sparse = {}
for name, fl in SPARSE:
    sparse[name] = evaluate(fl)
    print(f"       {name:<42} R={sparse[name]['recall']['mean']:6.2f}"
          f"  FPR={sparse[name]['fpr']['mean']:5.2f}  F1={sparse[name]['f1']['mean']:6.2f}", flush=True)
RESULTS['iot_sparse'] = sparse


# ---------------------------------------------- 5. nuisance-hardness stress
print('[5/10] Nuisance-hardness (q_hard) stress sweep ...', flush=True)
stress = {}
for q in (0.0, 0.05, 0.10, 0.20, 0.30):
    stress[f'{q:.2f}'] = {
        'mobile_only': evaluate(Flags(mobile_only=True, no_iot_validation=True,
                                      no_weighting=True, no_graph=True), q_hard=q),
        'mobile_only_graph': evaluate(Flags(mobile_only=True, no_iot_validation=True,
                                            no_adaptive_weights=True), q_hard=q),
        'hybrid_fixed': evaluate(Flags(no_adaptive_weights=True, no_graph=True), q_hard=q),
        'proposed': evaluate(Flags(), q_hard=q),
    }
    r = stress[f'{q:.2f}']
    print(f"       q={q:.2f}  mobile-only FPR={r['mobile_only']['fpr']['mean']:5.2f}"
          f"  +graph={r['mobile_only_graph']['fpr']['mean']:5.2f}"
          f"  hybrid={r['hybrid_fixed']['fpr']['mean']:5.2f}"
          f"  proposed={r['proposed']['fpr']['mean']:5.2f}", flush=True)
RESULTS['nuisance_stress'] = stress


# ------------------------------------------------------------- 6. latency
print('[6/10] Latency ...', flush=True)
lat = []
for seed in SC.seeds:
    scen, devs, hours, reps = cache(seed)
    recs, meta, _ = run_consensus(scen, devs, hours, reps, Flags(), CON, burn(seed))
    lat += [m['latency_s'] for r, m in zip(recs, meta)
            if r['is_eq'] and m['detected'] and np.isfinite(m['latency_s'])]
RESULTS['latency_s'] = {'n': len(lat), 'mean': float(np.mean(lat)),
                        'std': float(np.std(lat)), 'p50': float(np.percentile(lat, 50)),
                        'p95': float(np.percentile(lat, 95)),
                        'ci95': float(1.96 * np.std(lat) / np.sqrt(len(lat)))}
print('       ', RESULTS['latency_s'], flush=True)


# --------------------------------------------------------- 7. packet loss
print('[7/10] Packet-loss resilience ...', flush=True)
RESULTS['packet_loss'] = {}
for pl in (0.0, 0.1, 0.2, 0.3):
    RESULTS['packet_loss'][f'{pl:.1f}'] = evaluate(Flags(), pl=pl)
    r = RESULTS['packet_loss'][f'{pl:.1f}']
    print(f"       loss={pl:.0%}  R={r['recall']['mean']:6.2f}  FPR={r['fpr']['mean']:5.2f}", flush=True)


# ------------------------------------------- 8. sensor-calibration sensitivity
print('[8/10] Sensor-calibration sensitivity (system level) ...', flush=True)
cal = {}
for lev in CALIBRATION_LEVELS:
    cal[lev.name] = {
        'fixed_weights': evaluate(Flags(no_adaptive_weights=True), lev=lev.name),
        'personalised':  evaluate(Flags(), lev=lev.name)}
    a, b = cal[lev.name]['fixed_weights'], cal[lev.name]['personalised']
    print(f"       {lev.name:<16} fixed  F1={a['f1']['mean']:6.2f} FPR={a['fpr']['mean']:5.2f}"
          f"  |  personalised F1={b['f1']['mean']:6.2f} FPR={b['fpr']['mean']:5.2f}", flush=True)
RESULTS['calibration_system'] = cal


# ------------------------------------------------------ 9. hyper-parameters
print('[9/10] Hyper-parameter sensitivity ...', flush=True)
import dataclasses
hp = {'dbscan': {}, 'temporal': {}, 'iot_weight': {}, 'rho_min': {},
      'graph_radius': {}, 'v_app_min': {}}
S3 = SC.seeds[:3]
for eps, mp in itertools.product([2, 3, 5, 7, 10], [2, 3, 4, 5]):
    hp['dbscan'][f'eps={eps},minPts={mp}'] = evaluate(
        Flags(), seeds=S3, cfg=dataclasses.replace(CON, eps_km=eps, min_pts=mp))
for tw in [1.0, 1.5, 2.0, 2.5, 3.0]:
    hp['temporal'][f'{tw}'] = evaluate(
        Flags(), seeds=S3, cfg=dataclasses.replace(CON, temporal_window_s=tw))
for wi in [0.5, 0.6, 0.7, 0.8, 0.9]:
    hp['iot_weight'][f'{wi}'] = evaluate(
        Flags(no_adaptive_weights=True), seeds=S3,
        cfg=dataclasses.replace(CON, w_iot=wi, w_mobile=round(1 - wi, 2)))
for rm in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    hp['rho_min'][f'{rm}'] = evaluate(
        Flags(mobile_only=True, no_iot_validation=True, no_adaptive_weights=True),
        seeds=S3, cfg=dataclasses.replace(CON, rho_min=rm))
for gr in [3.0, 5.0, 8.0, 12.0, 20.0]:
    hp['graph_radius'][f'{gr}'] = evaluate(
        Flags(mobile_only=True, no_iot_validation=True, no_adaptive_weights=True),
        seeds=S3, cfg=dataclasses.replace(CON, graph_radius_km=gr))
for vm in [1.5, 2.0, 2.5, 3.0, 4.0]:
    hp['v_app_min'][f'{vm}'] = evaluate(
        Flags(mobile_only=True, no_iot_validation=True, no_adaptive_weights=True),
        seeds=S3, cfg=dataclasses.replace(CON, v_app_min_km_s=vm))
RESULTS['hyperparameters'] = hp
for k in hp:
    b = max(hp[k], key=lambda x: hp[k][x]['f1']['mean'])
    print(f"       best {k:<13} {b:<18} F1={hp[k][b]['f1']['mean']:.2f}", flush=True)


# ------------------------------- 10. breakdowns, catalogue variants, weights
print('[10/10] Breakdowns ...', flush=True)
_, extra = evaluate(Flags(), extra=True)
mag_rec, dist_rec, fp_cls, evid = {}, {}, {}, []
rel_m, rel_i = [], []
for scen, recs, meta, eng in extra:
    dev_xy = np.array([[d.x_km, d.y_km] for d in eng.dev.values()])
    for s, r, m in zip(scen, recs, meta):
        if s.is_eq:
            b = f'M{int(s.magnitude*2)/2:.1f}'
            mag_rec.setdefault(b, [0, 0]); mag_rec[b][0] += 1; mag_rec[b][1] += int(r['det'])
            dmin = float(np.min(np.hypot(dev_xy[:, 0] - s.x_km, dev_xy[:, 1] - s.y_km)))
            db = f'{int(dmin // 10) * 10}-{int(dmin // 10) * 10 + 10}km'
            dist_rec.setdefault(db, [0, 0]); dist_rec[db][0] += 1; dist_rec[db][1] += int(r['det'])
        else:
            fp_cls.setdefault(s.nuisance, [0, 0]); fp_cls[s.nuisance][0] += 1
            fp_cls[s.nuisance][1] += int(r['det'])
    rel_m += [d.reliability for d in eng.dev.values() if d.kind == 'mobile']
    rel_i += [d.reliability for d in eng.dev.values() if d.kind == 'iot']
RESULTS['recall_by_magnitude'] = {k: {'n': v[0], 'tp': v[1], 'recall': 100 * v[1] / v[0]}
                                  for k, v in sorted(mag_rec.items())}
RESULTS['recall_by_nearest_device_distance'] = {
    k: {'n': v[0], 'tp': v[1], 'recall': 100 * v[1] / v[0]} for k, v in sorted(dist_rec.items())}
RESULTS['fp_by_nuisance_class'] = {k: {'n': v[0], 'fp': v[1], 'fpr': 100 * v[1] / v[0]}
                                   for k, v in sorted(fp_cls.items())}
RESULTS['learned_reliability'] = {
    'mobile': {'min': float(np.min(rel_m)), 'p25': float(np.percentile(rel_m, 25)),
               'median': float(np.median(rel_m)), 'p75': float(np.percentile(rel_m, 75)),
               'max': float(np.max(rel_m)), 'values': [float(x) for x in rel_m]},
    'iot': {'min': float(np.min(rel_i)), 'median': float(np.median(rel_i)),
            'max': float(np.max(rel_i)), 'values': [float(x) for x in rel_i]}}

# operationally relevant subset: events felt inside the network footprint
RESULTS['recall_M55plus'] = evaluate(Flags())   # placeholder replaced below
sub = []
for seed in SC.seeds:
    scen, devs, hours, reps = cache(seed)
    recs, meta, _ = run_consensus(scen, devs, hours, reps, Flags(), CON, burn(seed))
    sub.append(confusion([r for r in recs if (not r['is_eq']) or r['mag'] >= 5.5]))
RESULTS['recall_M55plus'] = agg(sub)
print('       M>=5.5 subset: R=%.2f FPR=%.2f F1=%.2f'
      % (RESULTS['recall_M55plus']['recall']['mean'],
         RESULTS['recall_M55plus']['fpr']['mean'],
         RESULTS['recall_M55plus']['f1']['mean']), flush=True)

RESULTS['gutenberg_richter_catalogue'] = evaluate(Flags(), mag_dist='gutenberg-richter')
print('       G-R catalogue: R=%.2f FPR=%.2f'
      % (RESULTS['gutenberg_richter_catalogue']['recall']['mean'],
         RESULTS['gutenberg_richter_catalogue']['fpr']['mean']), flush=True)

RESULTS['config'] = {
    'q_hard_reference': Q_REF, 'seeds': list(SC.seeds),
    'n_earthquakes': SC.n_earthquakes, 'n_nuisance': SC.n_false_positive,
    'n_mobile': REGION.n_mobile, 'n_iot': REGION.n_iot,
    'region_km': REGION.size_km, 'n_hubs': REGION.n_hubs,
    'hub_sigma_km': REGION.hub_sigma_km,
}
RESULTS['runtime_s'] = time.time() - T0
json.dump(RESULTS, open(os.path.join(OUT, 'simulation.json'), 'w'), indent=2)
print(f"\nDONE in {RESULTS['runtime_s']:.0f}s -> outputs/simulation.json")
