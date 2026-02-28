"""Feature extraction for steering event classification (F1–F11 from spec)."""

from typing import Optional

import numpy as np

from nnlc_tools.steering_classifier.config import ClassifierConfig
from nnlc_tools.steering_classifier.filters import compute_freq_energy_ratio


def _safe_pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, returning 0.0 on zero-variance or mismatched inputs."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    a_std = np.std(a)
    b_std = np.std(b)
    if a_std < 1e-9 or b_std < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_features(
    steering_torque: np.ndarray,
    steering_angle_deg: np.ndarray,
    actual_lateral_accel: np.ndarray,
    desired_lateral_accel: np.ndarray,
    a_ego: np.ndarray,
    v_ego: np.ndarray,
    cfg: ClassifierConfig = None,
) -> dict:
    """Extract all F1–F11 features from raw signal arrays for a single event.

    All arrays must be the same length (one entry per frame at cfg.sample_rate_hz).
    Missing/unavailable signals should be passed as arrays of NaN.

    Returns a dict with keys matching FEATURE_COLUMNS in train_rf.py.
    """
    if cfg is None:
        cfg = ClassifierConfig()

    dt = 1.0 / cfg.sample_rate_hz
    n = len(steering_torque)
    duration_s = (n - 1) * dt if n > 1 else n * dt

    # ── F1: Peak torque rate ───────────────────────────────────────────────
    if n > 1:
        torque_diffs = np.diff(steering_torque) / dt
        peak_torque_rate_nm_s = float(np.max(np.abs(torque_diffs)))
    else:
        peak_torque_rate_nm_s = 0.0

    # ── F2: Duration ──────────────────────────────────────────────────────
    # (passed as an output, not re-derived)

    # ── F3: Sign consistency ──────────────────────────────────────────────
    above_noise = np.abs(steering_torque) >= cfg.torque_noise_floor_nm
    torque_above = steering_torque[above_noise]
    if len(torque_above) > 0:
        dominant = max(np.sum(torque_above > 0), np.sum(torque_above < 0))
        sign_consistency = float(dominant / len(torque_above))
    else:
        sign_consistency = 1.0  # no signal → assume consistent (conservative)

    # ── F4: Zero-crossing rate ────────────────────────────────────────────
    if n > 1 and duration_s > 0:
        signs = np.sign(steering_torque)
        crossings = int(np.sum(signs[1:] != signs[:-1]))
        zero_crossing_rate_hz = crossings / duration_s
    else:
        zero_crossing_rate_hz = 0.0

    # ── F5: Kurtosis ──────────────────────────────────────────────────────
    if n >= 4:
        mean_t = np.mean(steering_torque)
        std_t = np.std(steering_torque)
        if std_t > 1e-9:
            torque_kurtosis = float(np.mean(((steering_torque - mean_t) / std_t) ** 4))
        else:
            torque_kurtosis = 3.0  # mesokurtic default
    else:
        torque_kurtosis = 3.0

    # ── F6: Longitudinal shock ────────────────────────────────────────────
    valid_a = a_ego[~np.isnan(a_ego)]
    if len(valid_a) > 0:
        has_longitudinal_shock = bool(
            np.max(np.abs(valid_a)) > cfg.a_ego_shock_threshold
            and duration_s < cfg.shock_max_duration_s
        )
    else:
        has_longitudinal_shock = False

    # ── F7: Torque–angle phase (dT leads dTheta → driver) ─────────────────
    torque_leads_angle: Optional[float] = None
    if n > 1:
        dT = np.diff(steering_torque) / dt
        dTheta = np.diff(steering_angle_deg) / dt
        if len(dT) >= 2:
            torque_leads_angle = _safe_pearsonr(dT, dTheta)

    # ── F8: Torque–lateral-accel correlation ──────────────────────────────
    torque_lat_accel_corr: Optional[float] = None
    min_corr_frames = max(2, int(cfg.min_duration_for_correlation_s * cfg.sample_rate_hz))
    valid_mask = ~np.isnan(actual_lateral_accel)
    t_valid = steering_torque[valid_mask]
    la_valid = actual_lateral_accel[valid_mask]
    if len(t_valid) >= min_corr_frames:
        torque_lat_accel_corr = _safe_pearsonr(t_valid, la_valid)

    # ── F9: Frequency energy ratio ────────────────────────────────────────
    freq_energy_ratio: Optional[float] = None
    min_freq_frames = max(2, int(cfg.min_duration_for_freq_analysis_s * cfg.sample_rate_hz))
    if n >= min_freq_frames:
        freq_energy_ratio = compute_freq_energy_ratio(
            steering_torque,
            fs=cfg.sample_rate_hz,
            low_band=cfg.freq_low_band,
            high_band=cfg.freq_high_band,
            warmup_samples=cfg.freq_filter_warmup_samples,
        )

    # ── F10: Speed-adaptive brevity ───────────────────────────────────────
    valid_v = v_ego[~np.isnan(v_ego)]
    v_mean = float(np.mean(valid_v)) if len(valid_v) > 0 else 0.0
    if v_mean > 1.0:
        max_disturbance_duration = cfg.max_pothole_length_m / v_mean
    else:
        max_disturbance_duration = cfg.max_pothole_length_m  # fallback at standstill
    speed_adjusted_is_brief = bool(duration_s < max_disturbance_duration)

    # ── F11: Lateral accel residual ───────────────────────────────────────
    valid_res_mask = ~(np.isnan(actual_lateral_accel) | np.isnan(desired_lateral_accel))
    if np.sum(valid_res_mask) > 0:
        lat_accel_residual = float(
            np.max(np.abs(actual_lateral_accel[valid_res_mask] - desired_lateral_accel[valid_res_mask]))
        )
    else:
        lat_accel_residual = float("nan")

    # ── Extras (used by RF but not in the 11 spec features) ───────────────
    peak_steering_torque_abs = float(np.max(np.abs(steering_torque))) if n > 0 else 0.0

    return {
        # Tier 1
        "peak_torque_rate_nm_s": peak_torque_rate_nm_s,
        "duration_s": duration_s,
        # Tier 2
        "sign_consistency": sign_consistency,
        "zero_crossing_rate_hz": zero_crossing_rate_hz,
        "torque_kurtosis": torque_kurtosis,
        "has_longitudinal_shock": has_longitudinal_shock,
        "torque_leads_angle": torque_leads_angle,
        # Tier 3
        "torque_lat_accel_corr": torque_lat_accel_corr,
        "freq_energy_ratio": freq_energy_ratio,
        "speed_adjusted_is_brief": speed_adjusted_is_brief,
        "lat_accel_residual": lat_accel_residual,
        # Extras
        "v_ego_mean": v_mean,
        "peak_steering_torque_abs": peak_steering_torque_abs,
    }
