"""
Evaluation data for the benchmark.

The repository ships ``data/eval_windows.npz``, which contains everything the
benchmark needs:

    X_val, y_val    4,084 held-out STEAD windows + WISDM/UCI-HAR negatives
                    (validation subjects), the in-domain evaluation set
    X_test, y_test    384 windows: 192 MyShake shake-table recordings and 192
                    human-activity windows from a third group of subjects,
                    the cross-domain evaluation set
    X_calib           500 training windows, used only as the representative
                    dataset for post-training INT8 quantisation

Windows are 150 samples x 3 axes, resampled to 50 Hz, band-pass filtered at
0.5-20 Hz and standardised per channel, i.e. exactly what a device feeds to the
detector.

Regenerating from the raw corpora
---------------------------------
Only needed if you want to change the preprocessing. Set

    EEW_STEAD_DIR, EEW_MYSHAKE_DIR, EEW_HAR_DIR

and run ``python -m benchmark.rebuild_windows`` (not shipped; see the paper's
Section 5 for the full specification). Note that the MyShake shake-table traces
are sampled at 25 Hz -- the value declared in every file header -- and must be
interpolated, not decimated, on the way to 50 Hz.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = os.path.join(HERE, 'data', 'eval_windows.npz')

MYSHAKE_FS = 25          # declared in every shake-table file header
TARGET_FS = 50
N_SAMPLES = 150


def load():
    """Return X_val, y_val, X_test, y_test, X_calib."""
    if not os.path.exists(WINDOWS):
        raise FileNotFoundError(
            f'{WINDOWS} is missing. It ships with the repository; if you are '
            'working from a sparse checkout, fetch it with git-lfs or download '
            'it from the release assets.')
    d = np.load(WINDOWS)
    return d['X_val'], d['y_val'], d['X_test'], d['y_test'], d['X_calib']


if __name__ == '__main__':
    Xv, yv, Xt, yt, Xc = load()
    print(f'validation   {Xv.shape}  ({int(yv.sum())} earthquake / '
          f'{int((yv == 0).sum())} noise)')
    print(f'cross-domain {Xt.shape}  ({int(yt.sum())} earthquake / '
          f'{int((yt == 0).sum())} noise)')
    print(f'calibration  {Xc.shape}')
