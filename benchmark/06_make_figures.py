#!/usr/bin/env python3
"""Step 4 -- generate every new/updated figure of the revised manuscript."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUTD = os.path.join(HERE, 'outputs')
FIGD = os.path.abspath(os.path.join(HERE, '..', 'figures'))
MA   = OUTD
os.makedirs(FIGD, exist_ok=True)

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 300, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})
C = {'mob': '#1f77b4', 'iot': '#2ca02c', 'bad': '#d62728',
     'warn': '#ff7f0e', 'grey': '#7f7f7f', 'pur': '#9467bd'}

SIM = json.load(open(os.path.join(OUTD, 'simulation.json')))
LST = json.load(open(os.path.join(OUTD, 'latency_stress.json')))
MOD = json.load(open(os.path.join(MA, 'model_analysis.json')))
CAL = json.load(open(os.path.join(OUTD, 'calibration_sensitivity.json')))


# ═══════════════════════════════════ F1  training curves / overfitting
h = MOD['training']['history']; be = MOD['training']['best_epoch_by_val_loss']
ep = np.arange(1, len(h['loss']) + 1)
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
ax[0].plot(ep, h['loss'], label='training', color=C['mob'])
ax[0].plot(ep, h['val_loss'], label='validation', color=C['bad'])
ax[0].axvline(be, ls='--', c='k', lw=.9)
ax[0].annotate(f'best epoch {be}\n(weights restored)', (be, max(h['val_loss']) * .72),
               fontsize=7.5, ha='left', xytext=(be + 1.2, max(h['val_loss']) * .72))
ax[0].set_xlabel('epoch'); ax[0].set_ylabel('binary cross-entropy'); ax[0].legend(frameon=False)
ax[0].set_title('(a) loss')
ax[1].plot(ep, 100 * np.array(h['accuracy']), label='training', color=C['mob'])
ax[1].plot(ep, 100 * np.array(h['val_accuracy']), label='validation', color=C['bad'])
ax[1].axvline(be, ls='--', c='k', lw=.9)
ax[1].set_xlabel('epoch'); ax[1].set_ylabel('accuracy (%)'); ax[1].legend(frameon=False, loc='lower right')
ax[1].set_title('(b) accuracy')
gap = 100 * (np.array(h['accuracy']) - np.array(h['val_accuracy']))
ax[2].plot(ep, gap, color=C['pur'])
ax[2].axhline(0, c='k', lw=.6); ax[2].axvline(be, ls='--', c='k', lw=.9)
ax[2].fill_between(ep, 0, gap, where=gap > 0, alpha=.18, color=C['pur'])
ax[2].set_xlabel('epoch'); ax[2].set_ylabel('train $-$ val accuracy (pp)')
ax[2].set_title('(c) generalisation gap')
fig.savefig(os.path.join(FIGD, 'training_curves.png')); plt.close(fig)
print('F1 training_curves.png')


# ═══════════════════════════════════ F2  detection-report packet format
GROUPS = [
    ('framing', '#c6dbef', [
        ('Protocol version', 1), ('Message type', 1), ('Sequence number', 2),
        ('Rotating device token (SHA-256 prefix)', 8)]),
    ('space-time stamp', '#a1d99b', [
        ('Latitude, 1.1 km grid (int32)', 4), ('Longitude, 1.1 km grid (int32)', 4),
        ('Trigger onset time, ms UTC (int64)', 8), ('Clock-quality flag', 1),
        ('Device class + calibration level', 1)]),
    ('detector output', '#fdd0a2', [
        ('Binary decision', 1), ('Detector confidence (uint8, 1/255)', 1),
        ('Peak ground acceleration (float16, g)', 2),
        ('STA/LTA ratio at trigger (uint8)', 1), ('Model version', 2),
        ('Battery / health status', 1), ('Reserved for extensions', 3)]),
    ('authenticated encryption', '#dadaeb', [
        ('AEAD nonce', 12), ('AEAD authentication tag', 16),
        ('Anchor signature prefix (Ed25519)', 18)]),
]
tot = sum(nb for _, _, fs in GROUPS for _, nb in fs)
fig = plt.figure(figsize=(9.6, 3.4))
axb = fig.add_axes([0.045, 0.60, 0.92, 0.30]); axb.set_axis_off(); axb.grid(False)
x = 0.0
for gname, col, fs in GROUPS:
    g0 = x
    for _, nb in fs:
        axb.add_patch(Rectangle((x, 0.30), nb, 0.55, facecolor=col, edgecolor='k', lw=.6))
        if nb >= 3:
            axb.text(x + nb / 2, 0.575, str(nb), ha='center', va='center', fontsize=7.5)
        x += nb
    axb.plot([g0, x], [0.13, 0.13], color=col, lw=4, solid_capstyle='butt')
    axb.text((g0 + x) / 2, -0.10, f'{gname}\n{int(x - g0)} B', ha='center', va='top', fontsize=7.2)
axb.set_xlim(-1.5, tot + 1.5); axb.set_ylim(-0.55, 1.25)
axb.plot([0, tot], [1.02, 1.02], 'k-', lw=.8)
axb.plot([0, 0], [0.97, 1.07], 'k-', lw=.8); axb.plot([tot, tot], [0.97, 1.07], 'k-', lw=.8)
axb.text(tot / 2, 1.10, f'{tot}-byte detection report (single CoAP / MQTT-SN datagram)',
         ha='center', fontsize=8.5)

axt = fig.add_axes([0.02, 0.0, 0.96, 0.50]); axt.set_axis_off(); axt.grid(False)
rows = [(n, nb, col) for _, col, fs in GROUPS for n, nb in fs]
half = (len(rows) + 1) // 2
for ci, chunk in enumerate((rows[:half], rows[half:])):
    x0 = 0.02 + ci * 0.505
    for ri, (n, nb, col) in enumerate(chunk):
        y = 0.95 - ri * 0.092
        axt.add_patch(Rectangle((x0, y - 0.026), 0.016, 0.052, facecolor=col,
                                edgecolor='k', lw=.4, transform=axt.transAxes,
                                clip_on=False))
        axt.text(x0 + 0.026, y, n, fontsize=6.9, va='center', transform=axt.transAxes)
        axt.text(x0 + 0.455, y, f'{nb} B', fontsize=6.9, va='center', ha='right',
                 transform=axt.transAxes)
fig.savefig(os.path.join(FIGD, 'packet_format.png')); plt.close(fig)
json.dump({'groups': [{'group': g, 'fields': [{'name': n, 'bytes': b} for n, b in fs]}
                      for g, _, fs in GROUPS], 'total_bytes': tot},
          open(os.path.join(OUTD, 'packet_format.json'), 'w'), indent=2)
print('F2 packet_format.png  total bytes =', tot)


# ═══════════════════════════ F3  spatio-temporal graph / moveout evidence
src = open(os.path.join(HERE, '02_run_experiments.py')).read()
head = src.split('# ================================================================== EXPERIMENTS')[0]
head = head.replace("HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)",
                    "HERE = %r" % HERE)
G = {}; exec(head, G)
CON, Flags = G['CON'], G['Flags']
scen, devs, hours, reps = G['gen_reports'](42, 0.0, None, 0.30, None)
pairs = {'earthquake': [], 'Construction / piling': [], 'Heavy-traffic corridor': [],
         'Crowd / stadium': []}
for s, rp in zip(scen, reps):
    key = 'earthquake' if s.is_eq else s.nuisance
    if key not in pairs or len(rp) < 2:
        continue
    P = np.array([[r['x'], r['y']] for r in rp]); T = np.array([r['t_report'] for r in rp])
    iu = np.triu_indices(len(rp), 1)
    d = np.hypot(P[iu[0], 0] - P[iu[1], 0], P[iu[0], 1] - P[iu[1], 1])
    dt = np.abs(T[iu[0]] - T[iu[1]])
    m = (d <= CON.graph_radius_km) & (d > 0)
    if m.sum():
        idx = np.random.default_rng(0).choice(np.where(m)[0], min(400, int(m.sum())), replace=False)
        pairs[key].append(np.column_stack([d[idx], dt[idx]]))
pairs = {k: np.vstack(v) for k, v in pairs.items() if v}

fig = plt.figure(figsize=(10.5, 3.3))
axg = fig.add_subplot(1, 3, 1)
rng = np.random.default_rng(3)
pos = rng.uniform(0, 10, (11, 2)); epi = np.array([-3.0, 5.0])
r = np.hypot(pos[:, 0] - epi[0], pos[:, 1] - epi[1]); tt = r / 6.0
for i in range(len(pos)):
    for j in range(i + 1, len(pos)):
        dd = np.hypot(*(pos[i] - pos[j]))
        if dd <= 5.5:
            axg.plot(*zip(pos[i], pos[j]), color='#bbbbbb', lw=.7, zorder=1)
sc = axg.scatter(pos[:, 0], pos[:, 1], c=tt, cmap='viridis', s=52, zorder=3,
                 edgecolor='k', linewidth=.4)
axg.scatter(*epi, marker='*', s=190, color=C['bad'], zorder=4, edgecolor='k', linewidth=.4)
axg.text(epi[0], epi[1] + .8, 'source', color=C['bad'], ha='center', fontsize=7.5)
plt.colorbar(sc, ax=axg, label='trigger time (s)', fraction=.046)
axg.set_xlabel('x (km)'); axg.set_ylabel('y (km)'); axg.set_title('(a) sensor graph $G=(V,E)$')
axg.set_aspect('equal')

axs = fig.add_subplot(1, 3, 2)
order = [('earthquake', C['iot'], 'o', 'earthquake', 5, .18),
         ('Crowd / stadium', C['pur'], 'v', 'crowd / stadium', 16, .75),
         ('Heavy-traffic corridor', C['warn'], '^', 'heavy-traffic corridor', 16, .75),
         ('Construction / piling', C['bad'], 's', 'construction / piling', 10, .45)]
for k, col, mk, lab, sz, al in order:
    if k in pairs:
        a = pairs[k]
        axs.scatter(a[:, 0], a[:, 1], s=sz, alpha=al, color=col, marker=mk, label=lab,
                    linewidths=0)
dd = np.linspace(0.001, CON.graph_radius_km, 100)
axs.plot(dd, dd / CON.v_app_min_km_s + CON.moveout_tol_s, 'k-', lw=1.6, zorder=6,
         label=r'bound $d_{ij}/v_{\min}+\tau$')
axs.set_yscale('log'); axs.set_ylim(1e-3, 60)
axs.set_xlabel(r'edge distance $d_{ij}$ (km)')
axs.set_ylabel(r'$|\Delta t_{ij}|$ (s)'); axs.set_title('(b) edge moveout test')
axs.text(4.4, 12, 'rejected', fontsize=7, color='#444444')
axs.text(4.4, 3e-3, 'accepted', fontsize=7, color='#444444')
axs.legend(fontsize=6.0, frameon=True, framealpha=.85, loc='lower right', handletextpad=.3)

axr = fig.add_subplot(1, 3, 3)
q = [float(k) for k in sorted(LST and SIM['nuisance_stress'], key=float)]
mo = [SIM['nuisance_stress'][f'{x:.2f}']['mobile_only']['fpr']['mean'] for x in q]
mg = [SIM['nuisance_stress'][f'{x:.2f}']['mobile_only_graph']['fpr']['mean'] for x in q]
hf = [SIM['nuisance_stress'][f'{x:.2f}']['hybrid_fixed']['fpr']['mean'] for x in q]
pr = [SIM['nuisance_stress'][f'{x:.2f}']['proposed']['fpr']['mean'] for x in q]
axr.plot(q, mo, 'o-', color=C['bad'], label='mobile-only')
axr.plot(q, mg, 's-', color=C['warn'], label='mobile-only + ST-graph')
axr.plot(q, hf, '^-', color=C['mob'], label='hybrid (IoT veto)')
axr.plot(q, pr, 'd-', color=C['iot'], label='proposed (all phases)')
axr.axhspan(10, 15, color='grey', alpha=.15)
axr.set_xlabel(r'nuisance hardness $q_{\rm hard}$'); axr.set_ylabel('system false-positive rate (%)')
axr.set_title('(c) false alarms vs. nuisance hardness')
axr.legend(fontsize=6.4, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGD, 'st_graph.png')); plt.close(fig)
print('F3 st_graph.png')


# ═══════════════════════════════════ F4  calibration sensitivity
levels = list(CAL.keys())
xs = np.arange(len(levels))
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
ax[0].plot(xs, [100 * CAL[l]['mobile']['recall'] for l in levels], 'o-', color=C['mob'],
           label='smartphone')
ax[0].plot(xs, [100 * CAL[l]['anchor']['recall'] for l in levels], 's-', color=C['iot'],
           label='IoT anchor')
ax[0].set_ylabel('device-level recall (%)'); ax[0].set_title('(a) recall')
ax[1].plot(xs, [100 * CAL[l]['mobile']['fpr'] for l in levels], 'o-', color=C['mob'])
ax[1].plot(xs, [100 * CAL[l]['anchor']['fpr'] for l in levels], 's-', color=C['iot'])
ax[1].set_ylabel('device-level FPR (%)'); ax[1].set_title('(b) false-positive rate')
sysc = SIM['calibration_system']
ax[2].plot(xs, [sysc[l]['fixed_weights']['f1']['mean'] for l in levels], 'o-',
           color=C['grey'], label='class weights')
ax[2].plot(xs, [sysc[l]['personalised']['f1']['mean'] for l in levels], 'd-',
           color=C['pur'], label='personalised weights')
ax[2].set_ylabel('system F1 (%)'); ax[2].set_title('(c) after consensus')
for a in ax:
    a.set_xticks(xs); a.set_xticklabels([l.split()[0] for l in levels])
    a.set_xlabel('sensor calibration level')
ax[0].legend(frameon=False, fontsize=7.5); ax[2].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIGD, 'calibration_sensitivity.png')); plt.close(fig)
print('F4 calibration_sensitivity.png')


# ═══════════════════════════════════ F5  learned reliability
relm = np.array(SIM['learned_reliability']['mobile']['values'])
reli = np.array(SIM['learned_reliability']['iot']['values'])
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
ax[0].hist(relm, bins=28, color=C['mob'], alpha=.75, label=f'smartphones (n={len(relm)})')
ax[0].hist(reli, bins=14, color=C['iot'], alpha=.75, label=f'IoT anchors (n={len(reli)})')
ax[0].axvline(0.3, ls='--', c=C['mob'], lw=1); ax[0].axvline(0.7, ls='--', c=C['iot'], lw=1)
ax[0].text(0.3, ax[0].get_ylim()[1] * .93, r'  $w_m$=0.3', fontsize=7, color=C['mob'])
ax[0].text(0.7, ax[0].get_ylim()[1] * .80, r'  $w_i$=0.7', fontsize=7, color=C['iot'])
ax[0].set_xlabel(r'learned reliability $r_d$'); ax[0].set_ylabel('sensors')
ax[0].legend(frameon=False, fontsize=7)
ax[0].set_title('(a) posterior after warm-up')
sp = LST['iot_sparse_stress']; names = list(sp.keys())
w = .36; xs = np.arange(len(names))
ax[1].bar(xs - w / 2, [sp[n]['fpr']['mean'] for n in names], w, color=C['bad'], label='FPR (%)')
ax[1].bar(xs + w / 2, [100 - sp[n]['f1']['mean'] for n in names], w, color=C['grey'],
          label='100 $-$ F1 (%)')
ax[1].set_xticks(xs); ax[1].set_xticklabels(['class\nweights', '+ ST-graph',
                                             '+ personalised'], fontsize=7.5)
ax[1].set_ylabel('%'); ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title(r'(b) IoT-sparse mode, $q_{\rm hard}=0.30$')
fig.tight_layout(); fig.savefig(os.path.join(FIGD, 'reliability.png')); plt.close(fig)
print('F5 reliability.png')


# ═══════════════════════════════════ F6  performance breakdown
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
mg = SIM['recall_by_magnitude']
ks = list(mg.keys()); xs = np.arange(len(ks))
ax[0].bar(xs, [mg[k]['recall'] for k in ks], color=C['mob'])
for i, k in enumerate(ks):
    ax[0].text(i, mg[k]['recall'] + 2, f"n={mg[k]['n']}", ha='center', fontsize=6.5)
ax[0].set_xticks(xs); ax[0].set_xticklabels([k.replace('M', '') for k in ks])
ax[0].set_xlabel('magnitude bin'); ax[0].set_ylabel('recall (%)'); ax[0].set_ylim(0, 112)
ax[0].set_title('(a) recall vs. magnitude')

b = LST['latency_budget_s']
order = ['p_travel_to_3rd', 'device_detection', 'network', 'consensus']
lab = ['P-wave travel\nto 3rd sensor', 'on-device\ndetection', 'uplink\ntransport',
       'gateway\nconsensus']
vals = [b[k]['mean'] for k in order]
cols = [C['grey'], C['mob'], C['warn'], C['iot']]
ypos = np.arange(len(vals))[::-1]
ax[1].barh(ypos, vals, height=.62, color=cols)
for y, v in zip(ypos, vals):
    txt = f'{v:.2f} s' if v >= 0.5 else f'{v*1e3:.0f} ms'
    ax[1].text(v * 1.25, y, txt, va='center', fontsize=6.8)
ax[1].set_xscale('log'); ax[1].set_xlim(0.02, 26)
ax[1].set_yticks(ypos); ax[1].set_yticklabels(lab, fontsize=6.5)
ax[1].set_xlabel('contribution to alert latency (s), log scale')
ax[1].axvline(sum(vals), ls='--', c=C['bad'], lw=1)
ax[1].text(sum(vals) * 1.14, ypos[-1],
           f"measured total\n{LST['alert_latency_s']['mean']:.2f} s",
           fontsize=6.5, color=C['bad'], va='center', ha='left')
ax[1].set_title('(b) alert latency budget')

lt = LST['warning_lead_time_s']
d = [int(k.replace('km', '')) for k in lt]
ax[2].plot(d, [lt[k]['mean'] for k in lt], 'o-', color=C['iot'])
ax[2].axhline(0, c='k', lw=.8)
ax[2].axvspan(0, LST['blind_zone_km'], color=C['bad'], alpha=.13)
ax[2].text(LST['blind_zone_km'] / 2, max(lt[k]['mean'] for k in lt) * .75,
           f"blind zone\n{LST['blind_zone_km']:.0f} km", ha='center', fontsize=6.8, color=C['bad'])
ax[2].set_xlabel('epicentral distance of the user (km)')
ax[2].set_ylabel('warning lead time (s)'); ax[2].set_title('(c) lead time before S arrival')
fig.tight_layout(); fig.savefig(os.path.join(FIGD, 'performance_breakdown.png')); plt.close(fig)
print('F6 performance_breakdown.png')


# ═══════════════════════════ F7  consensus hyper-parameter sensitivity
hp = SIM['hyperparameters']
fig, ax = plt.subplots(1, 4, figsize=(12.6, 2.9))

# (a) DBSCAN grid
d = hp['dbscan']
eps = sorted({float(k.split(',')[0].split('=')[1]) for k in d})
mps = sorted({int(k.split('minPts=')[1]) for k in d})
for mp, mk in zip(mps, ['o', 's', '^', 'v']):
    y = [d[f'eps={int(e)},minPts={mp}']['f1']['mean'] for e in eps]
    ax[0].plot(eps, y, mk + '-', label=f'minPts={mp}', ms=4)
ax[0].axvline(5, ls='--', c=C['bad'], lw=1)
ax[0].text(5.15, min(min(y) for y in [[d[f'eps={int(e)},minPts={mp}']['f1']['mean']
           for e in eps] for mp in mps]) + .3, ' operating\n point', fontsize=6.4, color=C['bad'])
ax[0].set_xlabel(r'DBSCAN $\epsilon$ (km)'); ax[0].set_ylabel('system F1 (%)')
ax[0].legend(fontsize=6.2, frameon=False); ax[0].set_title(r'(a) spatial clustering')

# (b) temporal window
t = hp['temporal']; tw = sorted(float(k) for k in t)
ax[1].plot(tw, [t[f'{x}']['f1']['mean'] for x in tw], 'o-', color=C['mob'], ms=4)
ax[1].axvline(2.0, ls='--', c=C['bad'], lw=1)
ax[1].set_xlabel(r'temporal window $\tau_w$ (s)'); ax[1].set_ylabel('system F1 (%)')
ax[1].set_title('(b) temporal windowing')

# (c) IoT class weight -- deliberately shows the flat response
g = hp['iot_weight']; wi = sorted(float(k) for k in g)
f1 = [g[f'{x}']['f1']['mean'] for x in wi]; ci = [g[f'{x}']['f1']['ci95'] for x in wi]
ax[2].errorbar(wi, f1, yerr=ci, fmt='o-', color=C['mob'], capsize=3, ms=4)
_lo = min(f1) - max(ci) - 2.6
_hi = max(f1) + max(ci) + 1.0
ax[2].set_ylim(_lo, _hi)
ax[2].set_xlabel(r'IoT class weight $w_i$   ($w_m{=}1{-}w_i$)')
ax[2].set_ylabel('system F1 (%)')
ax[2].set_title('(c) class weighting')

# (d) graph parameters, measured in anchor-free mode
r = hp['rho_min']; rr = sorted(float(k) for k in r)
ax[3].plot(rr, [r[f'{x}']['fpr']['mean'] for x in rr], 'o-', color=C['bad'],
           label=r'$\rho_{\min}$')
v = hp['v_app_min']; vv = sorted(float(k) for k in v)
ax3b = ax[3].twiny(); ax3b.grid(False)
ax3b.plot(vv, [v[f'{x}']['fpr']['mean'] for x in vv], 's--', color=C['iot'],
          label=r'$v_{\min}$')
ax[3].set_xlabel(r'edge-consistency threshold $\rho_{\min}$', color=C['bad'])
ax3b.set_xlabel(r'minimum apparent velocity $v_{\min}$ (km/s)', color=C['iot'])
ax[3].set_ylabel('anchor-free FPR (%)')
ax[3].set_title('(d) graph parameters', pad=26)
fig.tight_layout()
fig.savefig(os.path.join(FIGD, 'hyperparameters.png')); plt.close(fig)
print('F7 hyperparameters.png')
print('\nAll figures written to', FIGD)
