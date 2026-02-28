import numpy as np
import pytest
from nnlc_tools.steering_classifier.config import ClassifierConfig
from nnlc_tools.steering_classifier.features import extract_features, _safe_pearsonr


FS = 100.0
CFG = ClassifierConfig()


def nan_array(n: int) -> np.ndarray:
    return np.full(n, np.nan)


def zero_array(n: int) -> np.ndarray:
    return np.zeros(n)


class TestSafePearsonr:
    def test_perfect_positive_corr(self):
        a = np.arange(10.0)
        assert _safe_pearsonr(a, a) == pytest.approx(1.0)

    def test_perfect_negative_corr(self):
        a = np.arange(10.0)
        assert _safe_pearsonr(a, -a) == pytest.approx(-1.0)

    def test_constant_array_returns_zero(self):
        a = np.ones(10)
        b = np.arange(10.0)
        assert _safe_pearsonr(a, b) == 0.0

    def test_mismatched_length_returns_zero(self):
        assert _safe_pearsonr(np.ones(5), np.ones(6)) == 0.0


class TestExtractFeatures:
    def _call(self, torque, angle=None, lat=None, des_lat=None, a_ego=None, v_ego=None):
        n = len(torque)
        return extract_features(
            steering_torque=torque,
            steering_angle_deg=angle if angle is not None else zero_array(n),
            actual_lateral_accel=lat if lat is not None else nan_array(n),
            desired_lateral_accel=des_lat if des_lat is not None else nan_array(n),
            a_ego=a_ego if a_ego is not None else zero_array(n),
            v_ego=v_ego if v_ego is not None else np.full(n, 25.0),
            cfg=CFG,
        )

    def test_keys_present(self):
        feats = self._call(np.ones(50))
        expected = [
            "peak_torque_rate_nm_s", "duration_s", "sign_consistency",
            "zero_crossing_rate_hz", "torque_kurtosis", "has_longitudinal_shock",
            "torque_leads_angle", "torque_lat_accel_corr", "freq_energy_ratio",
            "speed_adjusted_is_brief", "lat_accel_residual", "v_ego_mean",
            "peak_steering_torque_abs",
        ]
        for k in expected:
            assert k in feats, f"Missing key: {k}"

    def test_peak_torque_rate_impulse(self):
        # Single-sample step from 0 to 10 Nm → rate = 10 / 0.01 = 1000 Nm/s
        torque = np.zeros(10)
        torque[5] = 10.0
        feats = self._call(torque)
        assert feats["peak_torque_rate_nm_s"] == pytest.approx(1000.0)

    def test_peak_torque_rate_constant(self):
        feats = self._call(np.ones(20))
        assert feats["peak_torque_rate_nm_s"] == pytest.approx(0.0)

    def test_sign_consistency_all_positive(self):
        feats = self._call(np.ones(20))
        assert feats["sign_consistency"] == pytest.approx(1.0)

    def test_sign_consistency_oscillating(self):
        torque = np.array([1.0, -1.0] * 20)
        feats = self._call(torque)
        assert feats["sign_consistency"] == pytest.approx(0.5, abs=0.1)

    def test_zero_crossing_rate(self):
        # Alternating sign at 100 Hz → crosses every sample → ZCR ≈ 100 Hz
        torque = np.array([1.0, -1.0] * 50)
        feats = self._call(torque)
        assert feats["zero_crossing_rate_hz"] > 50

    def test_kurtosis_impulse_is_high(self):
        torque = np.zeros(100)
        torque[50] = 20.0
        feats = self._call(torque)
        assert feats["torque_kurtosis"] > 6

    def test_kurtosis_sine_is_low(self):
        t = np.arange(100) / FS
        torque = np.sin(2 * np.pi * 2 * t)
        feats = self._call(torque)
        # Pure sine kurtosis ≈ 1.5 (lower than normal's 3)
        assert feats["torque_kurtosis"] < 3.5

    def test_longitudinal_shock_fires(self):
        a_ego = np.zeros(20)
        a_ego[5] = 2.0
        feats = self._call(np.ones(20), a_ego=a_ego)
        assert feats["has_longitudinal_shock"] is True

    def test_longitudinal_shock_no_fire_long_event(self):
        n = 100  # 1.0s — exceeds shock_max_duration_s=0.4
        a_ego = np.full(n, 2.0)
        feats = self._call(np.ones(n), a_ego=a_ego)
        assert feats["has_longitudinal_shock"] is False

    def test_torque_lat_corr_none_when_too_short(self):
        feats = self._call(np.ones(5), lat=np.ones(5))
        assert feats["torque_lat_accel_corr"] is None

    def test_torque_lat_corr_high_when_aligned(self):
        n = 50
        sig = np.linspace(0, 1, n)
        feats = self._call(sig, lat=sig)
        assert feats["torque_lat_accel_corr"] == pytest.approx(1.0, abs=0.05)

    def test_freq_energy_ratio_none_when_too_short(self):
        feats = self._call(np.ones(10))  # < 20 samples
        assert feats["freq_energy_ratio"] is None

    def test_speed_adaptive_brevity_high_speed(self):
        # At 25 m/s threshold = 2.5/25 = 0.1s → 5 frames (0.04s) should be brief
        feats = self._call(np.ones(4), v_ego=np.full(4, 25.0))
        assert feats["speed_adjusted_is_brief"] is True

    def test_speed_adaptive_brevity_long_event(self):
        # At 25 m/s threshold = 0.1s → 50 frames (0.49s) should NOT be brief
        feats = self._call(np.ones(50), v_ego=np.full(50, 25.0))
        assert feats["speed_adjusted_is_brief"] is False

    def test_lat_accel_residual_computed(self):
        n = 20
        lat = np.ones(n)
        des = np.zeros(n)
        feats = self._call(np.ones(n), lat=lat, des_lat=des)
        assert feats["lat_accel_residual"] == pytest.approx(1.0)

    def test_lat_accel_residual_nan_when_unavailable(self):
        feats = self._call(np.ones(20))
        assert np.isnan(feats["lat_accel_residual"])
