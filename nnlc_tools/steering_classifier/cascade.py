"""Three-stage heuristic cascade classifier for steering override events."""

from nnlc_tools.steering_classifier.config import ClassifierConfig
from nnlc_tools.steering_classifier.types import ClassificationResult


def classify_event(
    features: dict,
    cfg: ClassifierConfig = None,
) -> ClassificationResult:
    """Classify an event using the three-stage heuristic cascade.

    Stages are ordered by increasing latency/data requirement:
      Stage 1 — Tier 1 features (peak_torque_rate_nm_s, duration_s)
      Stage 2 — Tier 2 features (sign_consistency, ZCR, kurtosis, shock, phase, brevity)
      Stage 3 — Tier 3 features (corr, freq_ratio, lat_residual)

    `features` may contain None/nan values for unavailable Tier 3 features — those
    scoring blocks are skipped gracefully. This allows calling with partial feature
    sets for real-time/incremental classification.

    Returns ClassificationResult. When evidence is ambiguous the default is "driver"
    (false negatives are more safety-critical than false positives).
    """
    if cfg is None:
        cfg = ClassifierConfig()

    rate = features.get("peak_torque_rate_nm_s", None)
    dur = features.get("duration_s", None)

    # ── Stage 1: Fast Gate ────────────────────────────────────────────────
    if rate is not None and dur is not None:
        if rate > cfg.stage1_mechanical_rate and dur < cfg.stage1_mechanical_duration:
            return ClassificationResult(
                label="mechanical", confidence=0.95,
                mechanical_score=5.0, driver_score=0.0, stage=1, features=features,
            )
        if rate < cfg.stage1_driver_rate and dur > cfg.stage1_driver_duration:
            return ClassificationResult(
                label="driver", confidence=0.95,
                mechanical_score=0.0, driver_score=5.0, stage=1, features=features,
            )

    # ── Stage 2: Confirmation Gate ────────────────────────────────────────
    mechanical_score = 0.0
    driver_score = 0.0

    # Torque rate
    if rate is not None:
        if rate > cfg.torque_rate_definite_mechanical:
            mechanical_score += 1.5
        elif rate > cfg.torque_rate_likely_mechanical:
            mechanical_score += 1.0
        elif rate < cfg.torque_rate_definite_driver:
            driver_score += 1.0

    # Sign consistency
    sc = features.get("sign_consistency", None)
    if sc is not None:
        if sc < cfg.sign_consistency_mechanical:
            mechanical_score += 1.5
        elif sc < cfg.sign_consistency_ambiguous:
            mechanical_score += 0.5
        elif sc > cfg.sign_consistency_driver:
            driver_score += 1.0

    # Zero-crossing rate
    zcr = features.get("zero_crossing_rate_hz", None)
    if zcr is not None:
        if zcr > cfg.zcr_mechanical_hz:
            mechanical_score += 1.0
        elif zcr < cfg.zcr_driver_hz:
            driver_score += 0.5

    # Kurtosis
    kurt = features.get("torque_kurtosis", None)
    if kurt is not None:
        if kurt > cfg.kurtosis_impulsive:
            mechanical_score += 1.0
        elif kurt < cfg.kurtosis_smooth:
            driver_score += 0.5

    # Longitudinal shock
    shock = features.get("has_longitudinal_shock", None)
    if shock:
        mechanical_score += 1.5

    # Torque-angle phase
    phase = features.get("torque_leads_angle", None)
    if phase is not None:
        if phase < cfg.phase_mechanical_threshold:
            mechanical_score += 0.5
        elif phase > cfg.phase_driver_threshold:
            driver_score += 1.0

    # Speed-adaptive brevity
    brief = features.get("speed_adjusted_is_brief", None)
    if brief:
        mechanical_score += 1.0

    # Early exit on strong Stage 2 consensus
    if mechanical_score >= cfg.stage2_mechanical_exit and driver_score < 1.0:
        conf = min(0.95, 0.5 + mechanical_score * 0.1)
        return ClassificationResult(
            label="mechanical", confidence=conf,
            mechanical_score=mechanical_score, driver_score=driver_score,
            stage=2, features=features,
        )
    if driver_score >= cfg.stage2_driver_exit and mechanical_score < 1.0:
        conf = min(0.95, 0.5 + driver_score * 0.1)
        return ClassificationResult(
            label="driver", confidence=conf,
            mechanical_score=mechanical_score, driver_score=driver_score,
            stage=2, features=features,
        )

    # ── Stage 3: Contextual Gate ──────────────────────────────────────────

    # Torque–lateral-accel correlation (strongest single feature)
    corr = features.get("torque_lat_accel_corr", None)
    if corr is not None:
        if corr > cfg.corr_strong_driver:
            driver_score += 2.0
        elif corr > cfg.corr_moderate_driver:
            driver_score += 1.0
        elif corr < cfg.corr_mechanical:
            mechanical_score += 1.5

    # Frequency energy ratio
    fer = features.get("freq_energy_ratio", None)
    if fer is not None:
        if fer > cfg.freq_ratio_driver:
            driver_score += 1.5
        elif fer < 0.5:
            mechanical_score += 2.0
        elif fer < cfg.freq_ratio_mechanical:
            mechanical_score += 1.5

    # Lateral accel residual
    res = features.get("lat_accel_residual", None)
    if res is not None and not (res != res):  # NaN check
        if res > cfg.lat_residual_strong_driver:
            driver_score += 1.5
        elif res > cfg.lat_residual_moderate_driver:
            driver_score += 0.5
        elif res < cfg.lat_residual_mechanical:
            mechanical_score += 0.5

    # ── Final decision ────────────────────────────────────────────────────
    total = mechanical_score + driver_score
    if total == 0.0:
        return ClassificationResult(
            label="driver", confidence=0.5,
            mechanical_score=0.0, driver_score=0.0, stage=3, features=features,
        )

    if mechanical_score > driver_score:
        conf = min(0.95, mechanical_score / total)
        return ClassificationResult(
            label="mechanical", confidence=conf,
            mechanical_score=mechanical_score, driver_score=driver_score,
            stage=3, features=features,
        )
    else:
        conf = min(0.95, driver_score / total)
        return ClassificationResult(
            label="driver", confidence=conf,
            mechanical_score=mechanical_score, driver_score=driver_score,
            stage=3, features=features,
        )
