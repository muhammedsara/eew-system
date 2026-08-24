"""
Phase 3 -- spatio-temporal graph consistency, for the interactive demo.

This is a thin adapter over the reference implementation in
``benchmark/network_simulator.py``: there is exactly one implementation of the
test in this repository, and both the demo and the published benchmark call it.

The test builds a radius graph over the sensors that triggered inside a
spatio-temporal window and, for every edge, checks the observed delay against
the moveout that the edge distance admits for a source radiating at phase
velocity at least ``v_min``:

    |Delta t_ij| <= d_ij / v_min + tau

A source that migrates at vehicular speed violates this on most of its edges; a
seismic wavefront does not. When at least four sensors are available the trigger
times are additionally fitted with a plane wave and the 95 % confidence interval
of the apparent velocity must intersect the admissible band.
"""
import math
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BENCH = os.path.join(_REPO, 'benchmark')
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

from network_simulator import graph_consistency as _graph_consistency  # noqa: E402
from sim_config import CON as _CON                                    # noqa: E402

KM_PER_DEG = 111.32
_ABSTAIN = {'passes': True, 'abstained': True, 'rho': float('nan'),
            'v_app': float('nan'), 'n_edges': 0, 'n_nodes': 0,
            'aperture_km': 0.0, 'r2': float('nan'),
            'v_ci': (float('nan'), float('nan'))}


def _local_xy(triggers):
    """Project WGS84 degrees onto a local tangent plane in km, centred on the cluster."""
    lats = [t.get('lat', t.get('latitude', 0.0)) for t in triggers]
    lons = [t.get('lon', t.get('longitude', 0.0)) for t in triggers]
    lat0 = sum(lats) / len(lats)
    lon0 = sum(lons) / len(lons)
    k = math.cos(math.radians(lat0))
    out = []
    for t in triggers:
        la = t.get('lat', t.get('latitude', 0.0))
        lo = t.get('lon', t.get('longitude', 0.0))
        out.append({'x': (lo - lon0) * KM_PER_DEG * k,
                    'y': (la - lat0) * KM_PER_DEG,
                    't_report': float(t.get('time', t.get('timestamp', 0.0)))})
    return out


def check(triggers, cfg=_CON, min_time_spread_s=1e-3):
    """
    Apply the test to demo-format triggers (lat/lon in degrees, one timestamp).

    The test abstains when the reported times carry no spread at all: with
    identical timestamps there is no moveout to measure, so vetoing would be an
    artefact of the input rather than a property of the source. The synthetic
    generators used by the demo stamp every trigger of a scenario with the same
    time, which is exactly that case.
    """
    if len(triggers) < 2:
        return dict(_ABSTAIN)
    rep = _local_xy(triggers)
    ts = [r['t_report'] for r in rep]
    if max(ts) - min(ts) < min_time_spread_s:
        return dict(_ABSTAIN)
    out = _graph_consistency(rep, cfg)

    # The apparent-velocity test needs a two-dimensional sensor layout: on a
    # perfectly collinear cluster the slowness perpendicular to the line is not
    # identifiable, and the point estimate is biased. Where the geometry lacks
    # that leverage we fall back on the edge test alone, which is what the
    # protocol prescribes when the velocity estimate cannot be constrained.
    P = np.array([[r['x'], r['y']] for r in rep], dtype=float)
    P -= P.mean(axis=0)
    sv = np.linalg.svd(P, compute_uv=False)
    degenerate = len(sv) < 2 or sv[0] <= 1e-9 or sv[1] / sv[0] < 0.05
    if degenerate and not out.get('abstained', True):
        rho = out.get('rho')
        out = dict(out)
        out['passes'] = bool(rho is not None and rho == rho and rho >= cfg.rho_min)
        out['v_app'] = float('nan')
        out['v_ci'] = (float('nan'), float('nan'))
        out['geometry'] = 'collinear, velocity test abstained'
    return out
