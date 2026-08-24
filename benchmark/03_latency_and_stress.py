#!/usr/bin/env python3
"""
Step 3 -- latency budget, warning lead time, stress-point ablation and
personalised-weight design variants.

The device-side detection delay is *not* assumed: it is the empirical
distribution measured by the continuous stream simulation of the trained
model (ModelV5/outputs/stream_simulation.json), i.e. the time from the onset
of shaking at a sensor until three consecutive 3 s windows exceed the
detector threshold.
"""
import os, sys, json, itertools, dataclasses
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, '02_run_experiments.py')).read()
head = src.split('# ================================================================== EXPERIMENTS')[0]
head = head.replace("HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)",
                    "HERE = %r" % HERE)
G = {}; exec(head, G)
Flags, CON, SC, GM, NET = G['Flags'], G['CON'], G['SC'], G['GM'], G['NET']
gen_reports, gen_burnin, run_consensus = G['gen_reports'], G['gen_burnin'], G['run_consensus']
confusion, agg = G['confusion'], G['agg']
OUT = G['OUT']

STREAM = json.load(open(os.path.join(OUT, 'stream_simulation.json')))
DEV_LAT = np.array(STREAM['results']['detection_latencies'])   # measured, seconds
R = {'measured_device_detection_latency_s': {
        'values': DEV_LAT.tolist(), 'mean': float(DEV_LAT.mean()),
        'std': float(DEV_LAT.std()), 'min': float(DEV_LAT.min()),
        'max': float(DEV_LAT.max()),
        'source': 'ModelV5 continuous 600 s stream simulation, 10 events'},
     'stream_simulation': STREAM['results']}

CACHE, BURN = {}, {}
def cache(seed, q):
    if (seed, q) not in CACHE:
        CACHE[(seed, q)] = gen_reports(seed, 0.0, None, q, None)
    return CACHE[(seed, q)]
def burn(seed, q):
    if (seed, q) not in BURN:
        BURN[(seed, q)] = gen_burnin(seed, q)
    return BURN[(seed, q)]

Q_REF, Q_STRESS = 0.10, 0.30


# ─────────────────────────────────────────────────── 1. end-to-end latency
print('[1/5] Latency budget and warning lead time ...', flush=True)
rng = np.random.default_rng(7)
alert_times, budgets, leads = [], [], {d: [] for d in (10, 20, 30, 40, 60)}
for seed in SC.seeds:
    scen, devs, hours, reps = cache(seed, Q_REF)
    recs, meta, _ = run_consensus(scen, devs, hours, reps, Flags(), CON, burn(seed, Q_REF))
    for s, rc, m in zip(scen, recs, meta):
        if not (s.is_eq and rc['detected'] if isinstance(rc, dict) and 'detected' in rc
                else s.is_eq and rc['det']):
            continue
        rp = reps[s.sid]
        if len(rp) < CON.min_total:
            continue
        # replace the placeholder pipeline delay by the measured device latency
        t_dev = sorted(r['t_onset'] for r in rp)[:CON.min_total]
        det = rng.choice(DEV_LAT, size=len(t_dev))
        mu = np.log(NET.rtt_median_ms); sg = (np.log(NET.rtt_p95_ms) - mu) / 1.645
        net = rng.lognormal(mu, sg, size=len(t_dev)) / 1e3
        t_alert = float(np.max(np.array(t_dev) + det + net)) + CON.gateway_compute_s - s.t0
        alert_times.append(t_alert)
        budgets.append({'p_travel_to_3rd': float(max(t_dev) - s.t0),
                        'device_detection': float(det.max()),
                        'network': float(net.max()),
                        'consensus': CON.gateway_compute_s})
        for dkm in leads:
            t_s = dkm / GM.v_s_km_s
            leads[dkm].append(t_s - t_alert)

R['alert_latency_s'] = {'n': len(alert_times), 'mean': float(np.mean(alert_times)),
                        'std': float(np.std(alert_times)),
                        'p50': float(np.percentile(alert_times, 50)),
                        'p95': float(np.percentile(alert_times, 95))}
R['latency_budget_s'] = {k: {'mean': float(np.mean([b[k] for b in budgets])),
                             'p95': float(np.percentile([b[k] for b in budgets], 95))}
                         for k in budgets[0]}
R['warning_lead_time_s'] = {f'{d}km': {'mean': float(np.mean(v)),
                                       'p50': float(np.percentile(v, 50)),
                                       'frac_positive': float(np.mean(np.array(v) > 0))}
                            for d, v in leads.items()}
R['blind_zone_km'] = float(np.mean(alert_times) * GM.v_s_km_s)
print('       alert latency %.2f s (p95 %.2f) | blind zone %.1f km'
      % (R['alert_latency_s']['mean'], R['alert_latency_s']['p95'], R['blind_zone_km']), flush=True)
for k, v in R['warning_lead_time_s'].items():
    print(f"       lead @{k:<5} mean {v['mean']:6.2f} s   positive in {v['frac_positive']:.0%} of events", flush=True)


# ────────────────────────────────────────────── 2. ablation at stress point
print('[2/5] Ablation at the stress operating point q_hard = 0.30 ...', flush=True)
ABL = [('Full system',                 Flags()),
       ('w/o Spatial clustering',             Flags(no_spatial=True)),
       ('w/o Temporal windowing',             Flags(no_temporal=True)),
       ('w/o Spatio-temporal graph',          Flags(no_graph=True)),
       ('w/o Adaptive thresholding',          Flags(no_adaptive_thr=True)),
       ('w/o IoT anchor validation',          Flags(no_iot_validation=True)),
       ('w/o Personalised weights (class w)', Flags(no_adaptive_weights=True)),
       ('w/o Reliability weighting',          Flags(no_weighting=True)),
       ('Baseline: unweighted majority vote', Flags(majority_vote=True, no_graph=True,
                                                    no_iot_validation=True,
                                                    no_adaptive_thr=True))]
abl = {}
for name, fl in ABL:
    ub = not (fl.no_adaptive_weights or fl.no_weighting or fl.majority_vote)
    per = []
    for seed in SC.seeds:
        scen, devs, hours, reps = cache(seed, Q_STRESS)
        per.append(confusion(run_consensus(scen, devs, hours, reps, fl, CON,
                                           burn(seed, Q_STRESS) if ub else None)[0]))
    abl[name] = agg(per)
    print(f"       {name:<38} P={abl[name]['precision']['mean']:6.2f}"
          f" R={abl[name]['recall']['mean']:6.2f} F1={abl[name]['f1']['mean']:6.2f}"
          f" FPR={abl[name]['fpr']['mean']:5.2f}", flush=True)
R['ablation_stress'] = abl


# ─────────────────────────────────── 3. IoT-sparse at the stress point
print('[3/5] IoT-sparse deployment at q_hard = 0.30 ...', flush=True)
SP = [('Mobile-only, class weights',            Flags(mobile_only=True, no_iot_validation=True,
                                                      no_graph=True, no_adaptive_weights=True)),
      ('Mobile-only + ST-graph',                Flags(mobile_only=True, no_iot_validation=True,
                                                      no_adaptive_weights=True)),
      ('Mobile-only + ST-graph + personalised', Flags(mobile_only=True, no_iot_validation=True))]
sp = {}
for name, fl in SP:
    ub = not fl.no_adaptive_weights
    per = []
    for seed in SC.seeds:
        scen, devs, hours, reps = cache(seed, Q_STRESS)
        per.append(confusion(run_consensus(scen, devs, hours, reps, fl, CON,
                                           burn(seed, Q_STRESS) if ub else None)[0]))
    sp[name] = agg(per)
    print(f"       {name:<40} R={sp[name]['recall']['mean']:6.2f}"
          f" FPR={sp[name]['fpr']['mean']:5.2f} F1={sp[name]['f1']['mean']:6.2f}", flush=True)
R['iot_sparse_stress'] = sp


# ───────────────────────────── 4. personalised-weight design variants
print('[4/5] Personalised-weight design variants ...', flush=True)
VAR = {'class weights (submitted)': (Flags(no_adaptive_weights=True), CON, False),
       'personalised, floor 0.20':  (Flags(), CON, True),
       'personalised, floor 0.50':  (Flags(), dataclasses.replace(CON, ratio_floor=0.5), True),
       'personalised, floor 0.70':  (Flags(), dataclasses.replace(CON, ratio_floor=0.7), True)}
var = {}
for name, (fl, cfg, ub) in VAR.items():
    for mode, extra in (('hybrid', {}), ('IoT-sparse', dict(mobile_only=True,
                                                            no_iot_validation=True))):
        f2 = dataclasses.replace(fl, **extra) if extra else fl
        per = []
        for seed in SC.seeds:
            scen, devs, hours, reps = cache(seed, Q_STRESS)
            per.append(confusion(run_consensus(scen, devs, hours, reps, f2, cfg,
                                               burn(seed, Q_STRESS) if ub else None)[0]))
        var[f'{name} | {mode}'] = agg(per)
        a = var[f'{name} | {mode}']
        print(f"       {name:<28} {mode:<11} R={a['recall']['mean']:6.2f}"
              f" FPR={a['fpr']['mean']:5.2f} F1={a['f1']['mean']:6.2f}", flush=True)
R['weight_variants_stress'] = var


# ───────────────────────────────── 5. magnitude-resolved aggregate recall
print('[5/5] Magnitude-resolved recall ...', flush=True)
mag = {}
for seed in SC.seeds:
    scen, devs, hours, reps = cache(seed, Q_REF)
    recs, _, _ = run_consensus(scen, devs, hours, reps, Flags(), CON, burn(seed, Q_REF))
    for s, rc in zip(scen, recs):
        if not s.is_eq: continue
        for lab, ok in (('M>=4.5', True), ('M>=5.0', s.magnitude >= 5.0),
                        ('M>=5.5', s.magnitude >= 5.5), ('M>=6.0', s.magnitude >= 6.0)):
            if ok:
                mag.setdefault(lab, [0, 0]); mag[lab][0] += 1; mag[lab][1] += int(rc['det'])
R['recall_cumulative'] = {k: {'n': v[0], 'tp': v[1], 'recall': 100 * v[1] / v[0]}
                          for k, v in sorted(mag.items())}
for k, v in R['recall_cumulative'].items():
    print(f"       {k}: {v['tp']}/{v['n']} = {v['recall']:.2f}%", flush=True)

json.dump(R, open(os.path.join(OUT, 'latency_stress.json'), 'w'), indent=2)
print('\nsaved -> outputs/latency_stress.json')
