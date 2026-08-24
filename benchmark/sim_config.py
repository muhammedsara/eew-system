"""
Network-simulator configuration.

Every constant below is either (i) measured from the trained model / datasets,
(ii) taken from a cited published source, or (iii) an explicitly stated design
choice.  Nothing is tuned to reproduce a target number.
"""
from dataclasses import dataclass, field
from typing import Tuple, List

# ----------------------------------------------------------------- geography
@dataclass(frozen=True)
class Region:
    """Metropolitan deployment area (Istanbul-like, 50 x 50 km)."""
    lat0: float = 40.75          # southern edge
    lon0: float = 28.60          # western edge
    size_km: float = 50.0
    n_mobile: int = 190
    n_iot: int = 10
    iot_grid_spacing_km: float = 16.0   # 10 anchors over 50x50 km
    # Participating phones are not uniformly spread: they concentrate in
    # residential / commercial hubs.  Modelled as a Neyman-Scott (Thomas)
    # cluster process; anthropogenic nuisance sources are co-located with the
    # same hubs, which is precisely what makes their triggers *correlated*.
    n_hubs: int = 12
    hub_sigma_km: float = 0.6
    iot_siting_sigma_km: float = 2.0


# ------------------------------------------------------------ ground motion
@dataclass(frozen=True)
class GMPE:
    """
    Generic median PGA attenuation

        ln PGA[g] = c1 + c2*M - c3*ln(sqrt(R^2 + h^2)) + eps,  eps ~ N(0, sigma)

    The four coefficients were fitted to reproduce published median PGA
    values (0.03 g for M4.5 at 10 km, 0.30 g for M6.5 at 10 km, 0.05 g for
    M6.5 at 50 km); sigma is the total log-normal variability reported for
    NGA-West2 style models.
    """
    c1: float = -5.3627
    c2: float = 1.1513
    c3: float = 1.3036
    h_km: float = 8.0
    sigma_ln: float = 0.55
    v_p_km_s: float = 6.0
    v_s_km_s: float = 3.5


# ----------------------------------------------------------------- triggering
@dataclass(frozen=True)
class Trigger:
    """Tier-1 event-trigger gate (identical logic on phones and anchors)."""
    pga_thresh_mobile_g: float = 0.05     # paper Sec. 3.1
    pga_thresh_iot_g: float = 0.02        # anchors: isolated mount, lower floor
    sta_win_s: float = 0.5                # streaming STA/LTA wake-up monitor
    lta_win_s: float = 10.0               # (documented in Sec. 5.2; the
    sta_lta_ratio: float = 3.0            #  simulator models the wake-up stage
    detrigger_ratio: float = 1.5          #  by its PGA gate, see Sec. 5.4)
    window_s: float = 3.0
    fs: int = 50
    cnn_thresh: float = 0.5


# ------------------------------------------------------------------ network
@dataclass(frozen=True)
class Network:
    """Uplink model.  Log-normal RTT calibrated to published LTE measurements."""
    rtt_median_ms: float = 45.0
    rtt_p95_ms: float = 87.0
    packet_loss: float = 0.0
    iot_backhaul_loss_factor: float = 0.25   # wired/fixed backhaul is more reliable
    clock_jitter_s: float = 0.05             # NTP-disciplined phone clock, 1-sigma
    clock_jitter_iot_s: float = 0.01         # GPS-disciplined anchor clock
    edge_pipeline_ms: float = 180.0          # buffer fill + preprocess + inference
    edge_pipeline_jitter_ms: float = 25.0


# --------------------------------------------------------------- consensus
@dataclass(frozen=True)
class Consensus:
    eps_km: float = 5.0
    min_pts: int = 3
    max_cluster: int = 50
    temporal_window_s: float = 2.0
    temporal_step_s: float = 0.5
    w_mobile: float = 0.3
    w_iot: float = 0.7
    thr_night: float = 0.75      # 02:00-06:00
    thr_day: float = 0.90        # 08:00-18:00
    thr_other: float = 0.85
    min_iot: int = 1
    min_total: int = 3
    # ---   spatio-temporal graph -------------------------------------
    graph_radius_km: float = 8.0
    v_app_min_km_s: float = 2.5
    v_app_max_km_s: float = 12.0
    moveout_tol_s: float = 0.35
    rho_min: float = 0.60
    min_aperture_km: float = 2.5
    min_edges: int = 3
    gateway_compute_s: float = 0.05
    # ---   adaptive per-device reliability ---------------------------
    prior_strength_mobile: float = 4.0   # Beta pseudo-counts at cold start
    prior_strength_iot: float = 8.0
    ewma_forget: float = 0.995
    weight_floor: float = 0.02
    weight_ceiling: float = 1.0
    ratio_floor: float = 0.20
    ratio_ceiling: float = 1.60


# ------------------------------------------------- sensor calibration error
@dataclass(frozen=True)
class CalibrationLevel:
    """Per-device, time-invariant sensor imperfection."""
    name: str
    gain_sigma: float       # per-axis scale-factor error (relative)
    misalign_deg: float     # cross-axis misalignment, 1-sigma
    bias_frac: float        # per-axis offset, fraction of window std
    noise_frac: float       # added band-limited noise, fraction of window std
    clock_sigma_s: float    # additional timestamp error


CALIBRATION_LEVELS: List[CalibrationLevel] = [
    CalibrationLevel('L0 ideal',        0.00, 0.0, 0.00, 0.00, 0.000),
    CalibrationLevel('L1 datasheet',    0.02, 1.0, 0.01, 0.02, 0.010),
    CalibrationLevel('L2 typical',      0.05, 3.0, 0.03, 0.05, 0.030),
    CalibrationLevel('L3 uncalibrated', 0.10, 6.0, 0.06, 0.10, 0.060),
    CalibrationLevel('L4 degraded',     0.20, 12.0, 0.12, 0.20, 0.120),
    CalibrationLevel('L5 severe',       0.35, 20.0, 0.20, 0.35, 0.250),
]

# Field population: mixture over calibration levels used in the main runs.
FIELD_MIX_MOBILE = {'L1 datasheet': 0.20, 'L2 typical': 0.45,
                    'L3 uncalibrated': 0.25, 'L4 degraded': 0.10}
FIELD_MIX_IOT    = {'L0 ideal': 0.20, 'L1 datasheet': 0.55, 'L2 typical': 0.25}


# ------------------------------------------------------- nuisance-source model
@dataclass(frozen=True)
class NuisanceClass:
    """
    Anthropogenic nuisance source.

    ``r_eff_km`` is the radius inside which the source raises the device-level
    PGA above the Tier-1 wake-up threshold, i.e. the radius inside which every
    participating device runs the detector.  Values are order-of-magnitude
    consistent with published ground-borne vibration ranges for the
    corresponding source; the results are reported as a function of r_eff in
    the sensitivity analysis.
    """
    name: str
    n_scenarios: int
    r_eff_km: Tuple[float, float]   # footprint radius, uniform range
    temporal_spread_s: float        # trigger-time jitter inside the footprint
    move_speed_km_s: float          # apparent propagation speed of the source
    device_borne: bool = False      # True -> affects one device only


NUISANCE_CLASSES: List[NuisanceClass] = [
    # source-borne, spatially correlated
    NuisanceClass('Construction / piling', 120, (0.60, 1.20), 0.60, 0.60),
    NuisanceClass('Heavy-traffic corridor', 120, (0.10, 0.30), 0.40, 0.015),
    NuisanceClass('Metro / rail passage',   100, (0.15, 0.35), 0.30, 0.020),
    NuisanceClass('Crowd / stadium',         60, (0.20, 0.50), 0.80, 5.000),
    # device-borne, independent across devices
    NuisanceClass('Isolated handling / drop', 60, (0.0, 0.0), 0.10, 0.010, True),
    NuisanceClass('Venue-wide handling burst', 40, (0.20, 0.40), 1.20, 5.000),
]

# Stress parameter q_hard: probability that a device inside a *source-borne*
# nuisance footprint emits a report that the detector cannot distinguish from
# a seismic window.  q_hard = 0 is the measured setting (device decisions come
# from real held-out HAR recordings only) and is used for the primary results.
Q_HARD_PRIMARY: float = 0.0
Q_HARD_STRESS: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30)


@dataclass(frozen=True)
class Scenarios:
    n_earthquakes: int = 100
    mag_min: float = 4.5
    mag_max: float = 7.5
    mag_distribution: str = 'uniform'  # 'uniform' (main) | 'gutenberg-richter'
    gr_b_value: float = 1.0            # truncated Gutenberg-Richter, b = 1.0
    epicentre_margin_km: float = 10.0
    depth_km: Tuple[float, float] = (5.0, 30.0)
    interevent_mean_s: float = 3600.0  # homogeneous Poisson process
    n_false_positive: int = 500
    seeds: Tuple[int, ...] = (42, 43, 44, 45, 46)


REGION = Region(); GM = GMPE(); TRG = Trigger()
NET = Network(); CON = Consensus(); SC = Scenarios()
