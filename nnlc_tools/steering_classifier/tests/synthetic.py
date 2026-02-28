"""Synthetic event generators for unit and integration tests."""

import numpy as np


def _make_arrays(n: int, fs: float = 100.0):
    """Helper: time axis and blank arrays for a window of n frames."""
    t = np.arange(n) / fs
    return t, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)


def make_pothole_event(speed_mps: float = 25.0, duration_s: float = 0.06,
                       peak_torque_nm: float = 8.0, fs: float = 100.0) -> dict:
    """Short, oscillatory torque with longitudinal shock. Expected: mechanical."""
    n = max(2, int(duration_s * fs))
    t, torque, angle, lat_accel, des_lat, a_ego, v_ego = _make_arrays(n, fs)

    # Damped sinusoid at ~15 Hz
    freq = 15.0
    torque[:] = peak_torque_nm * np.exp(-t * 20) * np.sin(2 * np.pi * freq * t)
    # Small angle change (passive, not commanded)
    angle[:] = 0.5 * np.exp(-t * 30) * np.cos(2 * np.pi * 8 * t)
    # Longitudinal shock
    a_ego[:] = 0.1
    a_ego[n // 3] = 2.5
    # Lateral accel stays near desired (no path deviation)
    des_lat[:] = 0.1
    lat_accel[:] = 0.1 + 0.05 * np.random.default_rng(1).normal(size=n)
    v_ego[:] = speed_mps

    return dict(steering_torque=torque, steering_angle_deg=angle,
                actual_lateral_accel=lat_accel, desired_lateral_accel=des_lat,
                a_ego=a_ego, v_ego=v_ego)


def make_driver_lane_change(speed_mps: float = 25.0, duration_s: float = 3.0,
                             peak_torque_nm: float = 4.0, fs: float = 100.0) -> dict:
    """Smooth half-sine torque, correlated lateral accel. Expected: driver."""
    n = max(2, int(duration_s * fs))
    t, torque, angle, lat_accel, des_lat, a_ego, v_ego = _make_arrays(n, fs)

    torque[:] = peak_torque_nm * np.sin(np.pi * t / duration_s)
    angle[:] = np.cumsum(torque) * 0.02
    # Lateral accel follows torque with ~150 ms lag
    lag = max(1, int(0.15 * fs))
    lat_accel[:lag] = 0.0
    lat_accel[lag:] = 0.4 * torque[: n - lag]
    des_lat[:] = 0.0   # driver is departing from desired
    a_ego[:] = 0.0
    v_ego[:] = speed_mps

    return dict(steering_torque=torque, steering_angle_deg=angle,
                actual_lateral_accel=lat_accel, desired_lateral_accel=des_lat,
                a_ego=a_ego, v_ego=v_ego)


def make_driver_correction(speed_mps: float = 15.0, duration_s: float = 0.8,
                            peak_torque_nm: float = 2.0, fs: float = 100.0) -> dict:
    """Small, unidirectional lane-keeping correction. Expected: driver."""
    n = max(2, int(duration_s * fs))
    t, torque, angle, lat_accel, des_lat, a_ego, v_ego = _make_arrays(n, fs)

    rng = np.random.default_rng(42)
    torque[:] = peak_torque_nm * np.sin(np.pi * t / duration_s) + rng.normal(0, 0.1, n)
    angle[:] = np.cumsum(torque) * 0.01
    lat_accel[:] = 0.2 * torque + rng.normal(0, 0.05, n)
    des_lat[:] = 0.0
    a_ego[:] = 0.0
    v_ego[:] = speed_mps

    return dict(steering_torque=torque, steering_angle_deg=angle,
                actual_lateral_accel=lat_accel, desired_lateral_accel=des_lat,
                a_ego=a_ego, v_ego=v_ego)


def make_curb_impact(speed_mps: float = 8.0, duration_s: float = 0.15,
                     peak_torque_nm: float = 12.0, fs: float = 100.0) -> dict:
    """Extreme torque rate, brief, strong shock. Expected: mechanical."""
    n = max(2, int(duration_s * fs))
    t, torque, angle, lat_accel, des_lat, a_ego, v_ego = _make_arrays(n, fs)

    # Single sharp spike decaying quickly
    torque[:] = peak_torque_nm * np.exp(-t * 40) * np.sign(np.sin(2 * np.pi * 5 * t))
    angle[:] = 1.0 * np.exp(-t * 20)
    a_ego[:] = 0.5
    a_ego[0] = 3.5  # strong shock
    lat_accel[:] = 0.05
    des_lat[:] = 0.05
    v_ego[:] = speed_mps

    return dict(steering_torque=torque, steering_angle_deg=angle,
                actual_lateral_accel=lat_accel, desired_lateral_accel=des_lat,
                a_ego=a_ego, v_ego=v_ego)


def make_rough_road_segment(speed_mps: float = 20.0, duration_s: float = 0.5,
                             rms_torque_nm: float = 1.5, fs: float = 100.0) -> dict:
    """Band-limited noise 8–25 Hz, no single impulse. Expected: mechanical."""
    n = max(2, int(duration_s * fs))
    t, torque, angle, lat_accel, des_lat, a_ego, v_ego = _make_arrays(n, fs)

    rng = np.random.default_rng(7)
    # Compose high-frequency oscillation
    torque[:] = sum(
        rms_torque_nm * 0.5 * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        for f in [10, 14, 18, 22]
    )
    angle[:] = 0.2 * np.sin(2 * np.pi * 10 * t)
    lat_accel[:] = 0.1 + 0.05 * rng.normal(size=n)
    des_lat[:] = 0.1
    a_ego[:] = 0.3 + 0.2 * rng.normal(size=n)
    v_ego[:] = speed_mps

    return dict(steering_torque=torque, steering_angle_deg=angle,
                actual_lateral_accel=lat_accel, desired_lateral_accel=des_lat,
                a_ego=a_ego, v_ego=v_ego)


def make_edge_case_emergency_swerve(speed_mps: float = 30.0, duration_s: float = 0.4,
                                     peak_torque_nm: float = 6.0, fs: float = 100.0) -> dict:
    """Fast driver reaction: high rate but consistent sign, correlated with lat accel.
    Expected: driver (not mechanical).
    """
    n = max(2, int(duration_s * fs))
    t, torque, angle, lat_accel, des_lat, a_ego, v_ego = _make_arrays(n, fs)

    # Fast ramp then hold — high dT/dt at onset but unidirectional
    ramp_end = n // 4
    torque[:ramp_end] = peak_torque_nm * np.linspace(0, 1, ramp_end)
    torque[ramp_end:] = peak_torque_nm * (1.0 - 0.3 * t[ramp_end:] / duration_s)
    torque = np.clip(torque, 0, None)  # all positive

    angle[:] = np.cumsum(torque) * 0.015
    # Strong torque–lat-accel correlation
    lat_accel[:] = 0.5 * torque
    des_lat[:] = 0.0  # actively swerving away from desired
    a_ego[:] = 0.2
    v_ego[:] = speed_mps

    return dict(steering_torque=torque, steering_angle_deg=angle,
                actual_lateral_accel=lat_accel, desired_lateral_accel=des_lat,
                a_ego=a_ego, v_ego=v_ego)
