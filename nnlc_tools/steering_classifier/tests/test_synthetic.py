"""Integration tests: synthetic event generators → features → cascade classifier."""

import pytest
from nnlc_tools.steering_classifier.cascade import classify_event
from nnlc_tools.steering_classifier.config import ClassifierConfig
from nnlc_tools.steering_classifier.features import extract_features
from nnlc_tools.steering_classifier.tests.synthetic import (
    make_curb_impact,
    make_driver_correction,
    make_driver_lane_change,
    make_edge_case_emergency_swerve,
    make_pothole_event,
    make_rough_road_segment,
)


CFG = ClassifierConfig()


def _classify(signals: dict) -> tuple:
    feats = extract_features(cfg=CFG, **signals)
    result = classify_event(feats, CFG)
    return result.label, result.confidence, feats


class TestSyntheticPothole:
    def test_classified_as_mechanical(self):
        label, conf, feats = _classify(make_pothole_event())
        assert label == "mechanical", (
            f"Pothole classified as {label} (conf={conf:.2f})\n"
            f"  peak_rate={feats['peak_torque_rate_nm_s']:.1f}, "
            f"dur={feats['duration_s']:.3f}, "
            f"sign_cons={feats['sign_consistency']:.2f}, "
            f"zcr={feats['zero_crossing_rate_hz']:.1f}, "
            f"shock={feats['has_longitudinal_shock']}, "
            f"corr={feats['torque_lat_accel_corr']}"
        )


class TestSyntheticDriverLaneChange:
    def test_classified_as_driver(self):
        label, conf, feats = _classify(make_driver_lane_change())
        assert label == "driver", (
            f"Lane change classified as {label} (conf={conf:.2f})\n"
            f"  peak_rate={feats['peak_torque_rate_nm_s']:.1f}, "
            f"dur={feats['duration_s']:.3f}, "
            f"sign_cons={feats['sign_consistency']:.2f}, "
            f"corr={feats['torque_lat_accel_corr']}, "
            f"freq_ratio={feats['freq_energy_ratio']}"
        )


class TestSyntheticDriverCorrection:
    def test_classified_as_driver(self):
        label, conf, feats = _classify(make_driver_correction())
        assert label == "driver", (
            f"Correction classified as {label} (conf={conf:.2f})\n"
            f"  peak_rate={feats['peak_torque_rate_nm_s']:.1f}, "
            f"dur={feats['duration_s']:.3f}, "
            f"sign_cons={feats['sign_consistency']:.2f}"
        )


class TestSyntheticCurbImpact:
    def test_classified_as_mechanical(self):
        label, conf, feats = _classify(make_curb_impact())
        assert label == "mechanical", (
            f"Curb impact classified as {label} (conf={conf:.2f})\n"
            f"  peak_rate={feats['peak_torque_rate_nm_s']:.1f}, "
            f"dur={feats['duration_s']:.3f}, "
            f"shock={feats['has_longitudinal_shock']}"
        )


class TestSyntheticRoughRoad:
    def test_classified_as_mechanical(self):
        label, conf, feats = _classify(make_rough_road_segment())
        assert label == "mechanical", (
            f"Rough road classified as {label} (conf={conf:.2f})\n"
            f"  zcr={feats['zero_crossing_rate_hz']:.1f}, "
            f"sign_cons={feats['sign_consistency']:.2f}, "
            f"freq_ratio={feats['freq_energy_ratio']}"
        )


class TestSyntheticEmergencySwerve:
    def test_classified_as_driver(self):
        label, conf, feats = _classify(make_edge_case_emergency_swerve())
        assert label == "driver", (
            f"Emergency swerve classified as {label} (conf={conf:.2f})\n"
            f"  peak_rate={feats['peak_torque_rate_nm_s']:.1f}, "
            f"dur={feats['duration_s']:.3f}, "
            f"sign_cons={feats['sign_consistency']:.2f}, "
            f"corr={feats['torque_lat_accel_corr']}, "
            f"lat_res={feats['lat_accel_residual']:.2f}"
        )
