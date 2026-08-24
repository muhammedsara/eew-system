"""
Metropolitan-scale network simulator.

Tier 1 (device)  : real held-out waveform windows -> deployed 188 KiB INT8
                   TFLite detector, with per-device sensor calibration error.
Tier 2 (gateway) : six-phase consensus
                   1 spatial clustering (DBSCAN, haversine)
                   2 temporal windowing (2 s sliding)
                   3 spatio-temporal graph moveout consistency   [new]
                   4 reliability-weighted voting, adaptive weights [new]
                   5 adaptive (diurnal) thresholding
                   6 multi-sensor (IoT anchor) validation
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from sklearn.cluster import DBSCAN

from sim_config import (REGION, GM, TRG, NET, CON, SC, NUISANCE_CLASSES,
                         FIELD_MIX_MOBILE, FIELD_MIX_IOT, Q_HARD_PRIMARY)

KM_PER_DEG = 111.32


# ══════════════════════════════════════════════════════════════ device layer
@dataclass
class Device:
    did: int
    kind: str                # 'mobile' | 'iot'
    x_km: float
    y_km: float
    profile: int             # row index into the confidence tables
    clock_sigma_s: float
    # adaptive reliability (Beta pseudo-counts)
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def reliability(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class DeviceTables:
    """Pre-computed Tier-1 outputs (confidence + STA/LTA) for every profile."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.level = z['profiles_level']
        self.clock = z['profiles_clock']
        self.conf = {k: z[f'conf_{k}'] for k in
                     ('mobile_eq', 'mobile_noise', 'anchor_eq', 'anchor_noise')}
        self.stalta = {k: z[f'stalta_{k}'] for k in
                       ('mobile_eq', 'mobile_noise', 'anchor_eq', 'anchor_noise')}
        self.n_profiles = len(self.level)
        self.by_level = {}
        for i, l in enumerate(self.level):
            self.by_level.setdefault(str(l), []).append(i)


# ══════════════════════════════════════════════════════════ scenario layer
@dataclass
class Scenario:
    sid: int
    is_eq: bool
    t0: float
    x_km: float
    y_km: float
    magnitude: float = 0.0
    depth_km: float = 0.0
    nuisance: str = ''
    move_speed: float = 0.0
    spread_s: float = 0.0
    r_eff_km: float = 0.0
    device_borne: bool = False


def sample_gutenberg_richter(n, m_min, m_max, b, rng):
    u = rng.random(n)
    return np.clip(m_min - (1.0 / b) *
                   np.log10(1 - u * (1 - 10 ** (-b * (m_max - m_min)))), m_min, m_max)


def build_scenarios(rng, hubs, mag_dist=None) -> List[Scenario]:
    """
    Earthquake catalogue  : ``n_earthquakes`` events, magnitudes uniform on
    [4.5, 7.5] (main setting, identical to the submitted version) or drawn
    from a truncated Gutenberg-Richter law; epicentres uniform over the
    metropolitan square extended by a 10 km margin; focal depth U(5, 30) km.

    Nuisance catalogue    : ``n_false_positive`` events split over six source
    classes; each is placed at a randomly chosen urban hub, i.e. exactly where
    the participating phones are.

    Both catalogues are laid on a common time line as a homogeneous Poisson
    process, so the inter-event times are exponentially distributed.
    """
    mag_dist = mag_dist or SC.mag_distribution
    out, t = [], 0.0
    if mag_dist == 'gutenberg-richter':
        mags = sample_gutenberg_richter(SC.n_earthquakes, SC.mag_min,
                                        SC.mag_max, SC.gr_b_value, rng)
    else:
        mags = rng.uniform(SC.mag_min, SC.mag_max, SC.n_earthquakes)
    m0 = SC.epicentre_margin_km
    for i, m in enumerate(mags):
        t += rng.exponential(SC.interevent_mean_s)
        out.append(Scenario(
            sid=i, is_eq=True, t0=t,
            x_km=float(rng.uniform(-m0, REGION.size_km + m0)),
            y_km=float(rng.uniform(-m0, REGION.size_km + m0)),
            magnitude=float(m),
            depth_km=float(rng.uniform(*SC.depth_km))))
    k = SC.n_earthquakes
    for nc in NUISANCE_CLASSES:
        for _ in range(nc.n_scenarios):
            t += rng.exponential(SC.interevent_mean_s * SC.n_earthquakes /
                                 SC.n_false_positive)
            hx, hy = hubs[int(rng.integers(0, len(hubs)))]
            out.append(Scenario(
                sid=k, is_eq=False, t0=t,
                x_km=float(hx + rng.normal(0, 0.4)),
                y_km=float(hy + rng.normal(0, 0.4)),
                nuisance=nc.name,
                move_speed=nc.move_speed_km_s,
                spread_s=nc.temporal_spread_s,
                r_eff_km=float(rng.uniform(*nc.r_eff_km)),
                device_borne=nc.device_borne))
            k += 1
    return out


def build_hubs(rng):
    return [(float(rng.uniform(0, REGION.size_km)),
             float(rng.uniform(0, REGION.size_km))) for _ in range(REGION.n_hubs)]


def build_devices(tables: DeviceTables, rng, hubs) -> List[Device]:
    devs, did = [], 0
    def draw(mix):
        names, p = zip(*mix.items())
        lev = names[rng.choice(len(names), p=np.array(p) / sum(p))]
        return int(rng.choice(tables.by_level[lev]))
    # Thomas cluster process: phones concentrate in urban hubs
    for _ in range(REGION.n_mobile):
        pr = draw(FIELD_MIX_MOBILE)
        hx, hy = hubs[int(rng.integers(0, len(hubs)))]
        devs.append(Device(did, 'mobile',
                           float(hx + rng.normal(0, REGION.hub_sigma_km)),
                           float(hy + rng.normal(0, REGION.hub_sigma_km)), pr,
                           float(np.hypot(NET.clock_jitter_s, tables.clock[pr]))))
        did += 1
    # Anchors are installed on public infrastructure (schools, municipal
    # buildings), i.e. inside the built-up areas, but on structurally isolated
    # mounts.  One anchor per hub, spread over distinct hubs for coverage.
    order = list(rng.permutation(len(hubs)))
    for i in range(REGION.n_iot):
        pr = draw(FIELD_MIX_IOT)
        hx, hy = hubs[order[i % len(hubs)]]
        devs.append(Device(did, 'iot',
                           float(hx + rng.normal(0, REGION.iot_siting_sigma_km)),
                           float(hy + rng.normal(0, REGION.iot_siting_sigma_km)), pr,
                           float(np.hypot(NET.clock_jitter_iot_s, tables.clock[pr]))))
        did += 1
    return devs


# ══════════════════════════════════════════════════════════════ Tier 1
def median_pga(mag, r_epi_km, depth_km):
    r = np.sqrt(r_epi_km ** 2 + depth_km ** 2 + GM.h_km ** 2)
    return np.exp(GM.c1 + GM.c2 * mag - GM.c3 * np.log(r))


def tier1_reports(sc: Scenario, devs: List[Device], tables: DeviceTables,
                  rng, packet_loss: float, q_hard: float = Q_HARD_PRIMARY) -> List[Dict]:
    """
    Generate the Tier-1 detection reports that reach the gateway for one
    scenario.

    A device produces a report only if
      (i)  the wake-up gate fires -- the ground motion at the device exceeds
           its PGA threshold (earthquakes: median GMPE + log-normal
           variability; nuisance: inside the source footprint), and
      (ii) the deployed INT8 detector, evaluated on a *real held-out waveform
           window* transformed by that device's calibration profile, returns a
           confidence above 0.5, and
      (iii) the report survives the uplink.

    ``q_hard`` is the stress parameter of Sec. 5.4: with probability q_hard a
    device inside a source-borne nuisance footprint is fed a window the
    detector cannot distinguish from a seismic one.
    """
    reports = []
    if sc.is_eq:
        cand = devs
    elif sc.device_borne:
        cand = [devs[int(rng.integers(0, len(devs)))]]
    else:
        cand = devs

    for d in cand:
        dx, dy = d.x_km - sc.x_km, d.y_km - sc.y_km
        r = float(np.hypot(dx, dy))

        # ---------------------------------------------- (i) wake-up gate
        if sc.is_eq:
            pga = median_pga(sc.magnitude, r, sc.depth_km) * \
                  np.exp(rng.normal(0, GM.sigma_ln))
            thr = TRG.pga_thresh_mobile_g if d.kind == 'mobile' else TRG.pga_thresh_iot_g
            if pga < thr:
                continue
        else:
            pga = np.nan
            if not sc.device_borne and r > sc.r_eff_km:
                continue
            # anchors are mounted on fixed infrastructure and vibration
            # isolated; they are exposed to a source-borne nuisance only if
            # they happen to sit inside its footprint
            if d.kind == 'iot' and not sc.device_borne and r > sc.r_eff_km:
                continue

        # ------------------------------- (ii) detector on a real window
        kind = 'mobile' if d.kind == 'mobile' else 'anchor'
        if sc.is_eq:
            pool = kind + '_eq'
        elif (not sc.device_borne) and q_hard > 0 and rng.random() < q_hard:
            pool = kind + '_eq'          # stress: nuisance mimics a seismic window
        else:
            pool = kind + '_noise'
        wi = int(rng.integers(0, tables.conf[pool].shape[1]))
        c = float(tables.conf[pool][d.profile, wi])
        if c < TRG.cnn_thresh:
            continue

        # ------------------------------------------------------ timing
        if sc.is_eq:
            r_hypo = np.sqrt(r ** 2 + sc.depth_km ** 2)
            t_onset = sc.t0 + r_hypo / GM.v_p_km_s
        else:
            t_onset = sc.t0 + r / max(sc.move_speed, 1e-6) + \
                      rng.normal(0, sc.spread_s)
        t_report = t_onset + rng.normal(0, d.clock_sigma_s)

        pipeline = max(20.0, rng.normal(NET.edge_pipeline_ms,
                                        NET.edge_pipeline_jitter_ms)) / 1e3
        mu = np.log(NET.rtt_median_ms)
        sg = (np.log(NET.rtt_p95_ms) - mu) / 1.645
        net_s = float(rng.lognormal(mu, sg)) / 1e3

        # ------------------------------------------------ (iii) uplink
        loss = packet_loss * (NET.iot_backhaul_loss_factor if d.kind == 'iot' else 1.0)
        if rng.random() < loss:
            continue

        reports.append({
            'did': d.did, 'kind': d.kind, 'x': d.x_km, 'y': d.y_km,
            'conf': c, 'pga': float(pga),
            't_onset': float(t_onset), 't_report': float(t_report),
            't_arrival': float(t_onset + pipeline + net_s),
        })
    return reports


# ══════════════════════════════════════════════════════════════ Tier 2
def _moveout_fit(P, T):
    """
    Least-squares plane-wave fit  t_i = t0 + s_x x_i + s_y y_i.

    Returns the apparent velocity v = 1/|s|, a 95 % confidence interval on v
    obtained by propagating the covariance of the slowness estimate, the
    coefficient of determination and the residuals.  Working with the interval
    rather than the point estimate matters: a small cluster gives a poorly
    constrained slowness, and vetoing on such an estimate would throw away
    genuine events.
    """
    n = len(T)
    if n < 4:
        return np.nan, (0.0, np.inf), 0.0, np.zeros(n)
    A = np.column_stack([np.ones(n), P[:, 0], P[:, 1]])
    coef, *_ = np.linalg.lstsq(A, T, rcond=None)
    res = T - A @ coef
    dof = n - 3
    ss_tot = float(((T - T.mean()) ** 2).sum())
    r2 = 1.0 - float((res ** 2).sum()) / ss_tot if ss_tot > 1e-12 else 0.0
    sx, sy = float(coef[1]), float(coef[2])
    smag = float(np.hypot(sx, sy))
    if dof <= 0 or smag < 1e-9:
        return (np.inf if smag < 1e-9 else 1.0 / smag), (0.0, np.inf), r2, res
    sigma2 = float((res ** 2).sum()) / dof
    try:
        C = sigma2 * np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        return 1.0 / smag, (0.0, np.inf), r2, res
    g = np.array([sx, sy]) / smag                     # d|s| / ds
    var_s = float(g @ C[1:, 1:] @ g)
    se = np.sqrt(max(var_s, 0.0))
    lo_s, hi_s = smag - 1.96 * se, smag + 1.96 * se
    v_lo = 1.0 / hi_s if hi_s > 1e-9 else np.inf
    v_hi = 1.0 / lo_s if lo_s > 1e-9 else np.inf
    return 1.0 / smag, (v_lo, v_hi), r2, res


def graph_consistency(rep, cfg=CON):
    """
    Phase 3 -- spatio-temporal graph consistency.

    A radius graph G = (V, E) is built over the sensors that triggered inside
    the spatio-temporal window: V is the set of sensors, and (i, j) in E when
    the inter-sensor distance d_ij does not exceed ``graph_radius_km``.  Every
    edge carries the pair (d_ij, Delta t_ij), i.e. the temporal correlation is
    evaluated *relative to the edge distance*, which is what distinguishes a
    seismic wavefront from a co-located anthropogenic disturbance.

    (a) Edge moveout bound.  For any common source radiating at phase velocity
        v >= v_min, the travel-time difference between two sensors obeys the
        triangle inequality

            |Delta t_ij| <= d_ij / v_min + tau ,

        where tau absorbs clock and network jitter.  A source that migrates
        slowly across the city (a truck convoy at 15 m/s, a train, a piling rig
        whose ground roll travels at a few hundred m/s) violates this bound on
        most of its edges.  rho is the fraction of edges that satisfy it.

    (b) Apparent-velocity test.  When at least four sensors are available the
        trigger times are additionally fitted with a plane wave and the 95 %
        confidence interval of the apparent velocity must intersect the
        admissible band [v_min, v_max].  The interval, rather than the point
        estimate, is used so that geometries without enough leverage abstain
        instead of vetoing a genuine event.

    The window is accepted only if rho >= rho_min and (b) does not exclude the
    seismic band.
    """
    n = len(rep)
    P = np.array([[r['x'], r['y']] for r in rep])
    T = np.array([r['t_report'] for r in rep])
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    aperture = float(D.max()) if n > 1 else 0.0
    out = {'v_app': np.nan, 'v_ci': (np.nan, np.nan), 'r2': np.nan,
           'rho': np.nan, 'n_edges': 0, 'n_nodes': n,
           'aperture_km': aperture, 'passes': True, 'abstained': True}
    if n < 2:
        return out

    iu = np.triu_indices(n, 1)
    d = D[iu]
    dt = np.abs(T[iu[0]] - T[iu[1]])
    mask = d <= cfg.graph_radius_km
    n_edges = int(mask.sum())
    if n_edges < cfg.min_edges:
        return out

    # ---- (a) edge moveout bound ----------------------------------------
    bound = d[mask] / cfg.v_app_min_km_s + cfg.moveout_tol_s
    rho = float((dt[mask] <= bound).mean())
    out.update(rho=rho, n_edges=n_edges, abstained=False)

    # ---- (b) plane-wave apparent velocity ------------------------------
    band_excluded = False
    if n >= 4 and aperture >= cfg.min_aperture_km:
        v, (v_lo, v_hi), r2, _ = _moveout_fit(P, T)
        out.update(v_app=float(v), v_ci=(float(v_lo), float(v_hi)), r2=float(r2))
        band_excluded = (v_hi < cfg.v_app_min_km_s) or (v_lo > cfg.v_app_max_km_s)

    out['passes'] = bool((rho >= cfg.rho_min) and not band_excluded)
    return out


def adaptive_threshold(hour, cfg=CON):
    if 2 <= hour < 6:  return cfg.thr_night
    if 8 <= hour < 18: return cfg.thr_day
    return cfg.thr_other


@dataclass
class Flags:
    """Ablation switches -- exactly one phase disabled per run."""
    no_spatial: bool = False
    no_temporal: bool = False
    no_graph: bool = False
    no_weighting: bool = False        # equal weights
    no_adaptive_weights: bool = False # fixed class weights (submitted version)
    no_adaptive_thr: bool = False
    no_iot_validation: bool = False
    majority_vote: bool = False       # unweighted majority baseline
    mobile_only: bool = False
    iot_only: bool = False


class ConsensusEngine:
    def __init__(self, devices: List[Device], flags: Flags = None, cfg=CON):
        self.dev = {d.did: d for d in devices}
        self.f = flags or Flags()
        self.cfg = cfg
        for d in self.dev.values():
            s = cfg.prior_strength_mobile if d.kind == 'mobile' else cfg.prior_strength_iot
            w = cfg.w_mobile if d.kind == 'mobile' else cfg.w_iot
            d.alpha, d.beta = w * s, (1 - w) * s
        self._class_mean = {'mobile': cfg.w_mobile, 'iot': cfg.w_iot}

    def _refresh_class_means(self):
        """Fleet-level normaliser: the average sensor of a class scores 1."""
        for k in ('mobile', 'iot'):
            v = [d.reliability for d in self.dev.values() if d.kind == k]
            if v:
                self._class_mean[k] = float(np.mean(v))

    def rel_ratio(self, d):
        """Personalised, fleet-normalised reliability r_d / mean(r_class)."""
        m = max(self._class_mean.get(d.kind, 1e-6), 1e-6)
        return float(np.clip(d.reliability / m, self.cfg.ratio_floor,
                             self.cfg.ratio_ceiling))

    # ---------------------------------------------------------------- weights
    def weight(self, r):
        """Reliability weight used in the weighted mean of Eq. (1)."""
        d = self.dev[r['did']]
        if self.f.majority_vote or self.f.no_weighting:
            return 1.0
        w0 = self.cfg.w_mobile if d.kind == 'mobile' else self.cfg.w_iot
        if self.f.no_adaptive_weights:
            return w0
        # personalised weight: class prior scaled by how this sensor compares
        # with the rest of its class, so the class ratio 0.3 : 0.7 is preserved
        # while every sensor carries its own normalised reliability
        return float(np.clip(w0 * self.rel_ratio(d),
                             self.cfg.weight_floor, self.cfg.weight_ceiling))

    def evidence(self, r):
        """
        Reliability-normalised evidence contributed by one report.

        A sensor counts as one *nominal* sensor of its own class when its
        estimated reliability equals the cold-start class prior, less when it
        has proved unreliable in the field and more when it has proved better
        than its class.  With this normalisation the quorum rule
        ``sum_d evidence_d >= min_total`` reduces exactly to the fixed-weight
        rule ``|M| + |I| >= min_total`` at cold start, so the personalised
        scheme strictly generalises the submitted one.
        """
        d = self.dev[r['did']]
        if self.f.majority_vote or self.f.no_weighting or self.f.no_adaptive_weights:
            return 1.0
        return self.rel_ratio(d)

    def update_reliability(self, rep, decision):
        """Beta-Bernoulli posterior update with exponential forgetting."""
        if self.f.no_adaptive_weights or self.f.majority_vote or self.f.no_weighting:
            return
        lam = self.cfg.ewma_forget
        self._n_updates = getattr(self, '_n_updates', 0) + 1
        if self._n_updates % 25 == 0:
            self._refresh_class_means()
        for r in rep:
            d = self.dev[r['did']]
            d.alpha *= lam; d.beta *= lam
            if decision:
                d.alpha += r['conf']
                d.beta += (1.0 - r['conf'])
            else:
                d.alpha += (1.0 - r['conf']) * 0.5
                d.beta += r['conf'] * 0.5

    # -------------------------------------------------------------- pipeline
    def process(self, reports, hour, t0=0.0, learn=True):
        info = {'detected': False, 'score': 0.0, 'thr': 0.0, 'n_mobile': 0,
                'n_iot': 0, 'v_app': np.nan, 'rho': np.nan, 'latency_s': np.nan,
                'n_clusters': 0, 'graph_abstained': True, 'evidence': 0.0}
        if self.f.mobile_only:
            reports = [r for r in reports if r['kind'] == 'mobile']
        if self.f.iot_only:
            reports = [r for r in reports if r['kind'] == 'iot']
        if not reports:
            return info

        # -- Phase 1: spatial clustering ----------------------------------
        if self.f.no_spatial:
            clusters = [reports]
        else:
            P = np.array([[r['x'], r['y']] for r in reports])
            lab = DBSCAN(eps=self.cfg.eps_km, min_samples=self.cfg.min_pts).fit_predict(P)
            clusters = [[reports[i] for i in np.where(lab == c)[0][:self.cfg.max_cluster]]
                        for c in sorted(set(lab)) if c != -1]
        info['n_clusters'] = len(clusters)
        if not clusters:
            return info

        best = None
        for cl in clusters:
            # -- Phase 2: temporal windowing -------------------------------
            if self.f.no_temporal:
                windows = [cl]
            else:
                ts = np.array([r['t_report'] for r in cl])
                order = np.argsort(ts); cl_s = [cl[i] for i in order]; ts = ts[order]
                windows, start = [], ts[0]
                while start <= ts[-1]:
                    m = (ts >= start) & (ts < start + self.cfg.temporal_window_s)
                    if m.sum() >= 2:
                        windows.append([cl_s[i] for i in np.where(m)[0]])
                    start += self.cfg.temporal_step_s
            for w in windows:
                if len(w) < 2:
                    continue
                # -- Phase 3: spatio-temporal graph ------------------------
                g = ({'v_app': np.nan, 'rho': np.nan, 'passes': True,
                      'n_edges': 0, 'abstained': True}
                     if self.f.no_graph else graph_consistency(w, self.cfg))
                # -- Phase 4: reliability-weighted voting ------------------
                ws = np.array([self.weight(r) for r in w])
                cs = np.array([r['conf'] for r in w])
                score = float((ws * cs).sum() / max(ws.sum(), 1e-9))
                # -- Phase 5: adaptive threshold ---------------------------
                thr = self.cfg.thr_other if self.f.no_adaptive_thr else adaptive_threshold(hour, self.cfg)
                n_iot = sum(1 for r in w if r['kind'] == 'iot')
                n_mob = len(w) - n_iot
                evid = float(sum(self.evidence(r) for r in w))
                # -- Phase 6: multi-sensor validation ----------------------
                ok_iot = True if (self.f.no_iot_validation or self.f.mobile_only) \
                    else n_iot >= self.cfg.min_iot
                ok_n = evid >= self.cfg.min_total
                det = (score >= thr) and ok_iot and ok_n and g['passes']
                key = (det, len(w), score)
                if best is None or key > best[0]:
                    # operational latency: origin time -> gateway holds every
                    # report needed for the decision (+ consensus compute time)
                    need = sorted(r['t_arrival'] for r in w)[:max(self.cfg.min_total, 1)]
                    lat = need[-1] - t0 + self.cfg.gateway_compute_s
                    best = (key, {'detected': det, 'score': score, 'thr': thr,
                                  'n_mobile': n_mob, 'n_iot': n_iot,
                                  'v_app': g['v_app'], 'rho': g['rho'],
                                  'evidence': evid,
                                  'graph_abstained': g.get('abstained', True),
                                  'latency_s': lat, 'n_clusters': len(clusters),
                                  '_win': w})
        if best is None:
            return info
        out = best[1]; win = out.pop('_win')
        if learn:
            self.update_reliability(win, out['detected'])
        return out
