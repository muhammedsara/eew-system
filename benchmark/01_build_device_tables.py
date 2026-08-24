#!/usr/bin/env python3
"""
Step 1 -- build the device-level decision tables used by the network simulator.

For every held-out waveform window and every sampled *device calibration
profile* we run the deployed 188 KiB INT8 TFLite detector and store its
confidence.  The network simulator then only has to look values up, which
makes the consensus ablation exactly comparable across configurations
(identical Tier-1 decisions, only the Tier-2 logic changes).

Pools
-----
  mobile_eq     192 MyShake shake-table windows        (cross-domain, unseen)
  mobile_noise  192 WISDM/UCI-HAR windows              (held-out subjects)
  anchor_eq    2042 STEAD windows                      (held-out events)
  anchor_noise 2042 WISDM/UCI-HAR windows              (held-out subjects)
"""
import os, sys, json
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3'); os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from scipy import signal as sps
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim_config import CALIBRATION_LEVELS, TRG

REPO = os.path.dirname(HERE)
TFL  = os.path.join(REPO, 'models', 'earthquake_detector.tflite')
OUT  = os.path.join(HERE, 'outputs'); os.makedirs(OUT, exist_ok=True)

N_PROFILES_PER_LEVEL = 6      # independent device draws per calibration level
SEED = 20260822


# --------------------------------------------------------------- perturbation
def _rot(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.]])
    Ry = np.array([[cb, 0, sb], [0, 1., 0], [-sb, 0, cb]])
    Rx = np.array([[1., 0, 0], [0, cc, -sc], [0, sc, cc]])
    return Rz @ Ry @ Rx


def make_profile(level, rng):
    """Draw one persistent device calibration profile from a severity level."""
    ang = np.deg2rad(level.misalign_deg)
    return {
        'level': level.name,
        'gain':  1.0 + rng.normal(0, level.gain_sigma, 3),
        'R':     _rot(*rng.normal(0, ang, 3)) if ang > 0 else np.eye(3),
        'bias':  rng.normal(0, level.bias_frac, 3),
        'noise': level.noise_frac,
        'clock_sigma_s': level.clock_sigma_s,
    }


_B, _A = sps.butter(4, [0.5 / 25.0, 20.0 / 25.0], btype='band')


def apply_profile(X, prof, rng):
    """
    Apply a device's sensor imperfections to a batch of windows (N,150,3),
    then re-apply the deployed per-channel z-score front end.

    Order follows the physical chain: true motion -> cross-axis misalignment
    -> per-axis scale-factor error -> offset -> electronic noise floor ->
    on-device standardisation.
    """
    Y = X.astype(np.float64)
    s = Y.std(axis=1, keepdims=True) + 1e-9
    Y = Y @ prof['R'].T                       # misalignment (survives z-scoring)
    Y = Y * prof['gain']                      # scale-factor error
    Y = Y + prof['bias'] * s                  # offset
    if prof['noise'] > 0:                     # band-limited electronic noise
        n = rng.normal(0, 1.0, Y.shape)
        n = sps.filtfilt(_B, _A, n, axis=1)
        n /= (n.std(axis=1, keepdims=True) + 1e-9)
        Y = Y + prof['noise'] * s * n
    m = Y.mean(axis=1, keepdims=True); sd = Y.std(axis=1, keepdims=True) + 1e-8
    return ((Y - m) / sd).astype(np.float32)


# ------------------------------------------------------------------- STA/LTA
def sta_lta(X, fs=50, sta_s=0.5, lta_s=2.0):
    """
    Classic STA/LTA on the window energy envelope.  Scale invariant, hence
    computable on the standardised window.  The LTA length is capped by the
    3 s window; the streaming implementation uses the full 10 s buffer.
    """
    e = (X ** 2).sum(axis=2)
    ns, nl = int(sta_s * fs), int(lta_s * fs)
    c = np.cumsum(np.concatenate([np.zeros((e.shape[0], 1)), e], axis=1), axis=1)
    sta = (c[:, ns:] - c[:, :-ns]) / ns
    lta = (c[:, nl:] - c[:, :-nl]) / nl
    k = min(sta.shape[1], lta.shape[1])
    return np.max(sta[:, -k:] / (lta[:, -k:] + 1e-12), axis=1)


# ---------------------------------------------------------------------- main
def main():
    import dataset
    Xv, yv, Xt, yt, _ = dataset.load()
    pools = {
        'mobile_eq':    Xt[yt == 1],
        'mobile_noise': Xt[yt == 0],
        'anchor_eq':    Xv[yv == 1],
        'anchor_noise': Xv[yv == 0],
    }
    for k, v in pools.items():
        print(f'  pool {k:<13} {v.shape}')

    it = tf.lite.Interpreter(model_path=TFL, num_threads=8); it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]

    def infer(X):
        o = np.empty(len(X), dtype=np.float32)
        for i in range(len(X)):
            it.set_tensor(inp['index'], X[i:i + 1].astype(np.float32))
            it.invoke()
            o[i] = it.get_tensor(out['index'])[0][0]
        return o

    profiles, conf = [], {k: [] for k in pools}
    stalta = {k: sta_lta(v) for k, v in pools.items()}

    for li, lev in enumerate(CALIBRATION_LEVELS):
        for j in range(N_PROFILES_PER_LEVEL):
            # Deterministic, process-independent seed. Python's hash() is
            # randomised per interpreter, so it must not be used here: the
            # device calibration profiles have to be identical on every run
            # for the published numbers to be reproducible.
            rng = np.random.default_rng([SEED, li, j])
            prof = make_profile(lev, rng)
            profiles.append({'level': lev.name, 'idx': j,
                             'clock_sigma_s': prof['clock_sigma_s']})
            for k, X in pools.items():
                Xp = X if lev.name == 'L0 ideal' else apply_profile(X, prof, rng)
                conf[k].append(infer(Xp))
            print(f'  profile {lev.name:<16} #{j}  done', flush=True)

    np.savez_compressed(
        os.path.join(OUT, 'device_tables.npz'),
        profiles_level=np.array([p['level'] for p in profiles]),
        profiles_clock=np.array([p['clock_sigma_s'] for p in profiles]),
        **{f'conf_{k}': np.stack(v) for k, v in conf.items()},
        **{f'stalta_{k}': v for k, v in stalta.items()},
    )

    # summary: per-level device-level detection statistics
    summary = {}
    lv = np.array([p['level'] for p in profiles])
    for lev in CALIBRATION_LEVELS:
        m = lv == lev.name
        rec_mob = float((np.stack(conf['mobile_eq'])[m] >= TRG.cnn_thresh).mean())
        fpr_mob = float((np.stack(conf['mobile_noise'])[m] >= TRG.cnn_thresh).mean())
        rec_iot = float((np.stack(conf['anchor_eq'])[m] >= TRG.cnn_thresh).mean())
        fpr_iot = float((np.stack(conf['anchor_noise'])[m] >= TRG.cnn_thresh).mean())
        pr_m = rec_mob / max(rec_mob + fpr_mob, 1e-9)
        pr_i = rec_iot / max(rec_iot + fpr_iot, 1e-9)
        summary[lev.name] = {
            'mobile': {'recall': rec_mob, 'fpr': fpr_mob,
                       'f1': 2 * pr_m * rec_mob / max(pr_m + rec_mob, 1e-9)},
            'anchor': {'recall': rec_iot, 'fpr': fpr_iot,
                       'f1': 2 * pr_i * rec_iot / max(pr_i + rec_iot, 1e-9)},
        }
        print(f"  {lev.name:<16} mobile R={rec_mob:.4f} FPR={fpr_mob:.4f} | "
              f"anchor R={rec_iot:.4f} FPR={fpr_iot:.4f}")
    json.dump(summary, open(os.path.join(OUT, 'calibration_sensitivity.json'), 'w'), indent=2)
    print('\nsaved ->', os.path.join(OUT, 'device_tables.npz'))


if __name__ == '__main__':
    main()
