#!/usr/bin/env python3
"""
Step 5 -- performance-result figures.

Two figures: the detector's threshold-free behaviour on both domains, and a
system-level overview that puts every configuration on one recall/false-alarm
plane.  All inputs are the JSON/NPY artefacts produced by steps 1-3.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'outputs')
FIGD = os.path.abspath(os.path.join(HERE, '..', 'figures'))
MA   = OUTD

plt.rcParams.update({'font.size': 8.6, 'axes.grid': True, 'grid.alpha': .28,
                     'figure.dpi': 300, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})
C = {'ind': '#2c7bb6', 'xdom': '#d7191c', 'g': '#2ca02c', 'o': '#ff7f0e',
     'p': '#7b3294', 'k': '#4d4d4d'}

SIM = json.load(open(os.path.join(OUTD, 'simulation.json')))
LST = json.load(open(os.path.join(OUTD, 'latency_stress.json')))
DEP = json.load(open(os.path.join(MA, 'deployed_tflite.json')))

import sys
sys.path.insert(0, HERE)
import dataset
Xv, yv, Xt, yt, _ = dataset.load()
pv = np.load(os.path.join(MA, 'deployed_conf_val.npy'))
pt = np.load(os.path.join(MA, 'deployed_conf_test.npy'))


# ═════════════════════════════ FIG A: detector performance
fig, ax = plt.subplots(1, 4, figsize=(13.0, 3.05))

# (a) ROC
for y, p, c, lab in ((yv, pv, C['ind'], 'in-domain (STEAD, $n$=4084)'),
                     (yt, pt, C['xdom'], 'cross-domain (MyShake, $n$=384)')):
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(100 * fpr, 100 * tpr, color=c, lw=1.5,
               label=f'{lab}\nAUC = {roc_auc_score(y, p):.4f}')
ax[0].plot([0, 100], [0, 100], ls=':', c=C['k'], lw=.8)
ax[0].set_xlim(0, 20); ax[0].set_ylim(75, 100.5)
ax[0].set_xlabel('false-positive rate (%)'); ax[0].set_ylabel('recall (%)')
ax[0].legend(fontsize=6.6, frameon=False, loc='lower right')
ax[0].set_title('(a) ROC')

# (b) precision-recall
for y, p, c, lab in ((yv, pv, C['ind'], 'in-domain'), (yt, pt, C['xdom'], 'cross-domain')):
    pr, rc, _ = precision_recall_curve(y, p)
    ax[1].plot(100 * rc, 100 * pr, color=c, lw=1.5,
               label=f'{lab}, AP = {average_precision_score(y, p):.4f}')
ax[1].set_xlim(70, 100.5); ax[1].set_ylim(85, 100.5)
ax[1].set_xlabel('recall (%)'); ax[1].set_ylabel('precision (%)')
ax[1].legend(fontsize=6.8, frameon=False, loc='lower left')
ax[1].set_title('(b) precision--recall')

# (c) score distributions, cross-domain
b = np.linspace(0, 1, 34)
ax[2].hist(pt[yt == 1], bins=b, color=C['g'], alpha=.72, label='earthquake ($n$=192)')
ax[2].hist(pt[yt == 0], bins=b, color=C['xdom'], alpha=.72, label='human activity ($n$=192)')
ax[2].axvline(0.5, ls='--', c='k', lw=1)
ax[2].text(0.52, 6.0, ' decision\n threshold', fontsize=6.6, color=C['k'])
ax[2].set_yscale('log'); ax[2].set_xlabel('detector confidence $c$')
ax[2].set_ylabel('windows'); ax[2].legend(fontsize=6.8, frameon=False, loc='upper center')
ax[2].set_title('(c) cross-domain score separation')

# (d) confusion matrices
ax[3].set_axis_off(); ax[3].grid(False)
for k, (tag, m) in enumerate((('in-domain', DEP['val']), ('cross-domain', DEP['test_crossdomain']))):
    x0, y0 = 0.06 + k * 0.52, 0.16
    cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            ax[3].add_patch(plt.Rectangle((x0 + j * .19, y0 + (1 - i) * .30), .19, .30,
                                          facecolor=plt.cm.Blues(0.12 + 0.72 * norm[i, j]),
                                          edgecolor='w', lw=1.2,
                                          transform=ax[3].transAxes))
            ax[3].text(x0 + j * .19 + .095, y0 + (1 - i) * .30 + .15,
                       f'{int(cm[i, j])}\n{100*norm[i,j]:.1f}%', ha='center', va='center',
                       fontsize=7.4, transform=ax[3].transAxes,
                       color='white' if norm[i, j] > .55 else '#1a1a1a')
    ax[3].text(x0 + .19, y0 + .66, tag, ha='center', fontsize=8.0, transform=ax[3].transAxes)
    for j, lab in enumerate(('pred.\nnoise', 'pred.\nquake')):
        ax[3].text(x0 + j * .19 + .095, y0 - .07, lab, ha='center', va='top',
                   fontsize=6.6, transform=ax[3].transAxes)
    for i, lab in enumerate(('noise', 'quake')):
        ax[3].text(x0 - .012, y0 + (1 - i) * .30 + .15, lab, ha='right', va='center',
                   fontsize=6.6, rotation=90, transform=ax[3].transAxes)
ax[3].set_title('(d) confusion matrices')
fig.tight_layout()
fig.savefig(os.path.join(FIGD, 'detector_performance.png')); plt.close(fig)
print('detector_performance.png')


# ═════════════════════════════ FIG B: system performance overview
fig, ax = plt.subplots(1, 4, figsize=(13.0, 3.15))

# (a) ablation: FPR by removed phase, both operating points
ABL = ['Full system', 'w/o Spatial clustering', 'w/o Temporal windowing',
       'w/o Spatio-temporal graph', 'w/o Adaptive thresholding',
       'w/o IoT anchor validation', 'w/o Personalised weights (class w)',
       'Baseline: unweighted majority vote']
SHORT = ['full system', 'spatial', 'temporal', 'ST-graph', 'threshold',
         'anchor veto', 'personal. weights', 'majority vote']
x = np.arange(len(ABL)); w = 0.38
r1 = [SIM['ablation'][k]['fpr']['mean'] for k in ABL]
e1 = [SIM['ablation'][k]['fpr']['ci95'] for k in ABL]
r2 = [LST['ablation_stress'][k]['fpr']['mean'] for k in ABL]
e2 = [LST['ablation_stress'][k]['fpr']['ci95'] for k in ABL]
def clip_err(v, e):
    lo = [min(a, max(a - 0.0, 0.0)) if False else min(a, a) for a in e]
    return np.array([[min(vi, ei) for vi, ei in zip(v, e)], e])
ax[0].bar(x - w/2, r1, w, yerr=clip_err(r1, e1), capsize=2, color=C['ind'],
          label=r'$q_{\rm hard}=0.10$')
ax[0].bar(x + w/2, r2, w, yerr=clip_err(r2, e2), capsize=2, color=C['xdom'],
          label=r'$q_{\rm hard}=0.30$')
ax[0].set_yscale('symlog', linthresh=0.05)
ax[0].set_ylim(0, 20)
ax[0].set_xticks(x); ax[0].set_xticklabels(SHORT, fontsize=6.4, rotation=38, ha='right')
ax[0].set_ylabel('system false-positive rate (%)')
ax[0].legend(fontsize=6.8, frameon=False)
ax[0].set_title('(a) contribution of each phase')

# (b) recall / FPR plane
PTS = [('mobile-only', SIM['system_comparison']['Mobile-only (190 phones)'], C['xdom'], 'o'),
       ('unweighted hybrid', SIM['system_comparison']['Unweighted hybrid'], C['o'], 's'),
       ('class weights', SIM['system_comparison']['Weighted hybrid, fixed class weights'], C['ind'], '^'),
       ('+ ST-graph', SIM['system_comparison']['  + spatio-temporal graph'], C['g'], 'D'),
       ('+ personalised', SIM['system_comparison']['  + personalised weights (proposed)'], C['p'], 'P')]
for lab, v, c, m in PTS:
    ax[1].errorbar(max(v['fpr']['mean'], 0.01), v['recall']['mean'],
                   xerr=v['fpr']['ci95'], yerr=v['recall']['ci95'],
                   fmt=m, color=c, ms=6.5, capsize=2, label=lab)
ax[1].set_xscale('log'); ax[1].set_xlim(0.03, 30); ax[1].set_ylim(80.0, 88.8)
ax[1].set_xlabel('false-positive rate (%), log scale'); ax[1].set_ylabel('recall (%)')
ax[1].legend(fontsize=6.6, frameon=False, loc='upper right')
ax[1].set_title(r'(b) operating points, $q_{\rm hard}=0.10$')

# (c) packet loss
pl = SIM['packet_loss']; ks = sorted(pl, key=float)
xs = [100 * float(k) for k in ks]
ax[2].errorbar(xs, [pl[k]['recall']['mean'] for k in ks],
               yerr=[pl[k]['recall']['ci95'] for k in ks], fmt='o-', color=C['ind'],
               capsize=3, label='recall')
ax[2].set_xlabel('packet loss (%)'); ax[2].set_ylabel('recall (%)', color=C['ind'])
a2 = ax[2].twinx(); a2.grid(False)
a2.errorbar(xs, [pl[k]['fpr']['mean'] for k in ks],
            yerr=[pl[k]['fpr']['ci95'] for k in ks], fmt='s--', color=C['xdom'], capsize=3)
a2.set_ylabel('false-positive rate (%)', color=C['xdom']); a2.set_ylim(0, 1)
ax[2].set_title('(c) network resilience')

# (d) false alarms by nuisance class
cls = SIM['fp_by_nuisance_class']
order = sorted(cls, key=lambda k: -cls[k]['fpr'])
sh = {'Construction / piling': 'construction /\npiling',
      'Heavy-traffic corridor': 'heavy-traffic\ncorridor',
      'Metro / rail passage': 'metro / rail', 'Crowd / stadium': 'crowd /\nstadium',
      'Venue-wide handling burst': 'venue-wide\nhandling',
      'Isolated handling / drop': 'isolated\nhandling'}
y = np.arange(len(order))
ax[3].barh(y, [cls[k]['fpr'] for k in order], color=C['o'])
for i, k in enumerate(order):
    ax[3].text(cls[k]['fpr'] + 0.012, i, f"{cls[k]['fp']}/{cls[k]['n']}", va='center',
               fontsize=6.6)
ax[3].set_yticks(y); ax[3].set_yticklabels([sh[k] for k in order], fontsize=6.6)
ax[3].set_xlabel('false-alarm rate within class (%)')
ax[3].set_xlim(0, max(cls[k]['fpr'] for k in order) * 1.55 + 0.05)
ax[3].set_title(r'(d) residual false alarms, $q_{\rm hard}=0.10$')
fig.tight_layout()
fig.savefig(os.path.join(FIGD, 'system_performance.png')); plt.close(fig)
print('system_performance.png')
