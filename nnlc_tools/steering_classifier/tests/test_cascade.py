import pytest
from nnlc_tools.steering_classifier.cascade import classify_event
from nnlc_tools.steering_classifier.config import ClassifierConfig


CFG = ClassifierConfig()


def _result(label, confidence, stage):
    """Assert label, confidence range, and stage."""
    assert label in ("driver", "mechanical")
    assert 0.0 <= confidence <= 1.0
    assert stage in (1, 2, 3)


class TestStage1FastGate:
    def test_extreme_rate_very_brief_is_mechanical(self):
        feats = {"peak_torque_rate_nm_s": 150.0, "duration_s": 0.03}
        r = classify_event(feats, CFG)
        assert r.label == "mechanical"
        assert r.stage == 1
        assert r.confidence >= 0.9

    def test_slow_rate_sustained_is_driver(self):
        feats = {"peak_torque_rate_nm_s": 5.0, "duration_s": 1.0}
        r = classify_event(feats, CFG)
        assert r.label == "driver"
        assert r.stage == 1
        assert r.confidence >= 0.9

    def test_ambiguous_bypasses_stage1(self):
        feats = {"peak_torque_rate_nm_s": 50.0, "duration_s": 0.3}
        r = classify_event(feats, CFG)
        assert r.stage > 1

    def test_none_features_bypasses_stage1(self):
        r = classify_event({}, CFG)
        assert r.label == "driver"  # default safe
        assert r.stage == 3


class TestStage2ConfirmationGate:
    def test_chaotic_torque_triggers_mechanical(self):
        feats = {
            "peak_torque_rate_nm_s": 90.0, "duration_s": 0.2,
            "sign_consistency": 0.50, "zero_crossing_rate_hz": 20.0,
            "torque_kurtosis": 8.0, "has_longitudinal_shock": True,
            "torque_leads_angle": -0.2, "speed_adjusted_is_brief": True,
        }
        r = classify_event(feats, CFG)
        assert r.label == "mechanical"

    def test_smooth_torque_triggers_driver(self):
        feats = {
            "peak_torque_rate_nm_s": 5.0, "duration_s": 2.0,
            "sign_consistency": 0.95, "zero_crossing_rate_hz": 1.0,
            "torque_kurtosis": 2.5, "has_longitudinal_shock": False,
            "torque_leads_angle": 0.8, "speed_adjusted_is_brief": False,
        }
        r = classify_event(feats, CFG)
        assert r.label == "driver"

    def test_shock_alone_adds_mechanical_score(self):
        feats = {"has_longitudinal_shock": True, "peak_torque_rate_nm_s": 5.0, "duration_s": 0.2}
        r = classify_event(feats, CFG)
        assert r.mechanical_score > 0


class TestStage3ContextualGate:
    def test_high_corr_pushes_driver(self):
        feats = {
            "peak_torque_rate_nm_s": 60.0, "duration_s": 0.3,
            "sign_consistency": 0.80, "zero_crossing_rate_hz": 6.0,
            "torque_kurtosis": 3.0, "has_longitudinal_shock": False,
            "torque_leads_angle": 0.4, "speed_adjusted_is_brief": True,
            "torque_lat_accel_corr": 0.85,
            "freq_energy_ratio": 5.0,
            "lat_accel_residual": 1.5,
        }
        r = classify_event(feats, CFG)
        assert r.label == "driver"

    def test_low_corr_high_freq_pushes_mechanical(self):
        feats = {
            "peak_torque_rate_nm_s": 70.0, "duration_s": 0.1,
            "sign_consistency": 0.55, "zero_crossing_rate_hz": 18.0,
            "torque_kurtosis": 9.0, "has_longitudinal_shock": False,
            "torque_leads_angle": 0.0, "speed_adjusted_is_brief": True,
            "torque_lat_accel_corr": 0.02,
            "freq_energy_ratio": 0.3,
            "lat_accel_residual": 0.1,
        }
        r = classify_event(feats, CFG)
        assert r.label == "mechanical"

    def test_nan_lat_residual_handled(self):
        import math
        feats = {
            "peak_torque_rate_nm_s": 30.0, "duration_s": 0.5,
            "lat_accel_residual": float("nan"),
        }
        r = classify_event(feats, CFG)
        _result(r.label, r.confidence, r.stage)


class TestDefaultSafe:
    def test_empty_features_returns_driver(self):
        r = classify_event({}, CFG)
        assert r.label == "driver"
        assert r.confidence == 0.5

    def test_tie_returns_driver(self):
        # Equal scores should default to driver (mechanical_score == driver_score == 0)
        r = classify_event({}, CFG)
        assert r.label == "driver"
