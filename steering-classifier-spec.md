# Implementation Spec: Enhanced Steering Event Classifier

## Context

We have an ADAS/autopilot override event log classifier that distinguishes **genuine driver steering interventions** from **mechanical disturbances** (potholes, road bumps, curb impacts). Data is sampled at **100 Hz**.

The current classifier uses 5 binary rules (score 0–5; score ≥ 2 → mechanical). Based on a literature review, the classifier needs to be substantially upgraded. This spec defines the enhanced classifier in full.

---

## 1. Input Signal Schema

Each frame (at 100 Hz = 10 ms interval) contains:

```python
@dataclass
class Frame:
    timestamp: float           # seconds
    steering_torque: float     # driver-applied torque, Nm
    torque_output: float       # system-commanded torque, Nm
    actual_lateral_accel: float  # m/s²
    desired_lateral_accel: float # m/s²
    steering_angle_deg: float  # degrees
    steering_rate_deg: float   # deg/s
    v_ego: float               # vehicle speed, m/s
    a_ego: float               # longitudinal acceleration, m/s²
    steering_pressed: bool     # override event flag (the raw trigger)
    lane_change_state: int     # enum for lane change status
```

An **event** is a contiguous window of frames where `steering_pressed == True`. The classifier operates on each event to label it as `"driver"` or `"mechanical"`.

---

## 2. Feature Extraction

Extract the following features from each event. Group them into three tiers matching the cascade architecture.

### Tier 1 — Fast Gate Features (computable within 10 ms / 1 sample)

#### F1: Peak Torque Rate (`peak_torque_rate_nm_s`)
```
dT/dt at each frame = (steering_torque[i] - steering_torque[i-1]) / dt
peak_torque_rate_nm_s = max(|dT/dt|) over the event
```
- `dt = 0.01` (100 Hz)
- Literature says driver inputs produce < 20–50 Nm/s; road impacts produce > 50–100 Nm/s
- **Old threshold was 500 Nm/s — way too conservative**

#### F2: Event Duration (`duration_s`)
```
duration_s = (last_frame.timestamp - first_frame.timestamp)
```

### Tier 2 — Confirmation Gate Features (computable at 50 ms / 5 samples)

#### F3: Torque Sign Consistency (`sign_consistency`)
```
dominant_sign_count = max(count(torque > 0), count(torque < 0))
sign_consistency = dominant_sign_count / total_nonzero_samples
```
- Driver inputs: ~1.0 (unidirectional)
- Road disturbances: ~0.5 (oscillatory)
- Exclude samples where |steering_torque| < 0.3 Nm (noise floor)

#### F4: Zero-Crossing Rate (`zero_crossing_rate_hz`)
```
crossings = count where sign(torque[i]) != sign(torque[i-1])
zero_crossing_rate_hz = crossings / duration_s
```
- Road disturbances oscillate at 8–20+ Hz → ZCR of 16–40 Hz
- Driver inputs: ZCR < 4 Hz

#### F5: Torque Kurtosis (`torque_kurtosis`)
```
Standard excess kurtosis of steering_torque over the event window
```
- Impulsive road events: kurtosis >> 3 (leptokurtic)
- Smooth driver inputs: kurtosis ≈ 3 (mesokurtic)

#### F6: Longitudinal Shock (`has_longitudinal_shock`)
```python
has_longitudinal_shock = (max(|a_ego|) > 1.5) and (duration_s < 0.4)
```
- Retained from original rule 2, validated by pothole detection literature

#### F7: Torque-Angle Phase (`torque_leads_angle`)
```python
# Compute derivatives
dT = diff(steering_torque) / dt
dTheta = diff(steering_angle_deg) / dt

# Cross-correlation at zero lag
correlation = pearsonr(dT, dTheta)

# During driver steering: torque leads angle → positive correlation
# During road disturbance: angle leads torque → negative or near-zero
torque_leads_angle = correlation  # continuous value [-1, 1]
```

### Tier 3 — Contextual Gate Features (computable at 100–200 ms)

#### F8: Torque–Lateral-Accel Correlation (`torque_lat_accel_corr`)
**This is the single most impactful missing feature.**
```python
# Pearson correlation between steering_torque and actual_lateral_accel
# over the event window (minimum 10 samples = 100 ms)
torque_lat_accel_corr = pearsonr(steering_torque, actual_lateral_accel)
```
- Driver steering: r > 0.6 (torque produces lateral accel, they are coupled)
- Road disturbance: r ≈ 0 (torque spike with no corresponding lat accel change)
- **If event is shorter than 10 samples, use NaN and skip this feature**

#### F9: Low/High Frequency Energy Ratio (`freq_energy_ratio`)
```python
# Apply 2nd-order Butterworth IIR filters to steering_torque:
#   low_band: bandpass 0.5–3 Hz (driver content)
#   high_band: bandpass 5–40 Hz (road disturbance content)
# Compute RMS energy of each band over the event window

from scipy.signal import butter, sosfilt

def compute_freq_ratio(torque_signal, fs=100):
    # Low band: 0.5-3 Hz (driver)
    sos_low = butter(2, [0.5, 3.0], btype='bandpass', fs=fs, output='sos')
    low_energy = np.sqrt(np.mean(sosfilt(sos_low, torque_signal)**2))
    
    # High band: 5-40 Hz (road disturbance)
    sos_high = butter(2, [5.0, 40.0], btype='bandpass', fs=fs, output='sos')
    high_energy = np.sqrt(np.mean(sosfilt(sos_high, torque_signal)**2))
    
    if high_energy < 1e-6:
        return 10.0  # cap: essentially all driver content
    return low_energy / high_energy

freq_energy_ratio = compute_freq_ratio(torque_array)
```
- Driver inputs: ratio >> 1 (most energy below 3 Hz)
- Road impacts: ratio ≈ 1 or < 1 (energy concentrated above 5 Hz)
- **Requires minimum ~20 samples (200 ms) for meaningful bandpass filtering; for shorter events, fall back to kurtosis + ZCR**

#### F10: Speed-Adaptive Duration Threshold (`speed_adjusted_is_brief`)
```python
# Maximum expected pothole length (calibratable, default 2.5 m)
L_MAX_POTHOLE = 2.5  # meters

# At current speed, maximum time to cross a pothole
if v_ego > 1.0:
    max_disturbance_duration = L_MAX_POTHOLE / v_ego
else:
    max_disturbance_duration = 2.5  # fallback at very low speed

speed_adjusted_is_brief = duration_s < max_disturbance_duration
```
- At 10 m/s: threshold = 0.25 s
- At 20 m/s: threshold = 0.125 s
- At 30 m/s: threshold = 0.083 s
- Replaces the old fixed 0.15 s and "highway brief" 0.4 s rules

#### F11: Lateral Acceleration Residual (`lat_accel_residual`)
```python
# How much does actual lateral accel deviate from what the system expected?
lat_accel_residual = np.max(np.abs(
    actual_lateral_accel - desired_lateral_accel
)) over the event
```
- Driver override: large residual (driver is steering away from desired path)
- Road disturbance: small residual (no sustained lateral deviation from path plan)
- **Threshold: residual > 0.5 m/s² suggests driver intent**

---

## 3. Classification Architecture

Implement **two** classifiers: a heuristic cascade (primary/production) and a random forest (for offline training and potential production upgrade).

### 3A. Heuristic Cascade Classifier

Three-stage cascade. Each stage can emit a final classification or pass to the next stage.

```python
def classify_event(features: dict) -> tuple[str, float]:
    """
    Returns: (label, confidence)
        label: "mechanical" or "driver"
        confidence: 0.0 to 1.0
    """
    
    # === STAGE 1: Fast Gate (available at 10 ms) ===
    
    # Definite mechanical: extreme torque rate + very brief
    if (features['peak_torque_rate_nm_s'] > 80 
        and features['duration_s'] < 0.05):
        return ("mechanical", 0.95)
    
    # Definite driver: slow torque buildup + sustained
    if (features['peak_torque_rate_nm_s'] < 20 
        and features['duration_s'] > 0.5):
        return ("driver", 0.95)
    
    # === STAGE 2: Confirmation Gate (available at 50 ms) ===
    
    mechanical_score = 0.0
    driver_score = 0.0
    
    # Torque rate scoring (continuous, not binary)
    if features['peak_torque_rate_nm_s'] > 80:
        mechanical_score += 1.5
    elif features['peak_torque_rate_nm_s'] > 50:
        mechanical_score += 1.0
    elif features['peak_torque_rate_nm_s'] < 20:
        driver_score += 1.0
    
    # Sign consistency
    if features['sign_consistency'] < 0.60:
        mechanical_score += 1.5
    elif features['sign_consistency'] < 0.75:
        mechanical_score += 0.5
    elif features['sign_consistency'] > 0.90:
        driver_score += 1.0
    
    # Zero-crossing rate
    if features['zero_crossing_rate_hz'] > 12:
        mechanical_score += 1.0
    elif features['zero_crossing_rate_hz'] < 4:
        driver_score += 0.5
    
    # Kurtosis
    if features['torque_kurtosis'] > 6:
        mechanical_score += 1.0
    elif features['torque_kurtosis'] < 4:
        driver_score += 0.5
    
    # Longitudinal shock
    if features['has_longitudinal_shock']:
        mechanical_score += 1.5
    
    # Torque-angle phase
    if features['torque_leads_angle'] < 0.1:
        mechanical_score += 0.5
    elif features['torque_leads_angle'] > 0.5:
        driver_score += 1.0
    
    # Speed-adjusted brevity
    if features['speed_adjusted_is_brief']:
        mechanical_score += 1.0
    
    # Early exit if strong consensus
    if mechanical_score >= 4.0 and driver_score < 1.0:
        conf = min(0.95, 0.5 + mechanical_score * 0.1)
        return ("mechanical", conf)
    if driver_score >= 3.0 and mechanical_score < 1.0:
        conf = min(0.95, 0.5 + driver_score * 0.1)
        return ("driver", conf)
    
    # === STAGE 3: Contextual Gate (available at 100-200 ms) ===
    
    # Torque-lateral-accel correlation (strongest feature)
    if features['torque_lat_accel_corr'] is not None:
        if features['torque_lat_accel_corr'] > 0.6:
            driver_score += 2.0
        elif features['torque_lat_accel_corr'] > 0.3:
            driver_score += 1.0
        elif features['torque_lat_accel_corr'] < 0.1:
            mechanical_score += 1.5
    
    # Frequency energy ratio
    if features['freq_energy_ratio'] is not None:
        if features['freq_energy_ratio'] > 3.0:
            driver_score += 1.5
        elif features['freq_energy_ratio'] < 1.0:
            mechanical_score += 1.5
        elif features['freq_energy_ratio'] < 0.5:
            mechanical_score += 2.0
    
    # Lateral accel residual
    if features['lat_accel_residual'] > 1.0:
        driver_score += 1.5
    elif features['lat_accel_residual'] > 0.5:
        driver_score += 0.5
    elif features['lat_accel_residual'] < 0.2:
        mechanical_score += 0.5
    
    # Final decision
    total = mechanical_score + driver_score
    if total == 0:
        return ("driver", 0.5)  # default to driver (conservative for safety)
    
    if mechanical_score > driver_score:
        conf = mechanical_score / total
        return ("mechanical", min(0.95, conf))
    else:
        conf = driver_score / total
        return ("driver", min(0.95, conf))
```

### 3B. Random Forest Classifier (Offline Training)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

FEATURE_COLUMNS = [
    'peak_torque_rate_nm_s',
    'duration_s',
    'sign_consistency',
    'zero_crossing_rate_hz',
    'torque_kurtosis',
    'has_longitudinal_shock',      # bool → 0/1
    'torque_leads_angle',
    'torque_lat_accel_corr',
    'freq_energy_ratio',
    'speed_adjusted_is_brief',     # bool → 0/1
    'lat_accel_residual',
    'v_ego_mean',                  # mean speed during event
    'peak_steering_torque_abs',    # max |torque| during event
]

# Hyperparameters (literature shows 10-50 trees, depth 5-8 works well)
clf = RandomForestClassifier(
    n_estimators=30,
    max_depth=7,
    min_samples_leaf=5,
    class_weight='balanced',  # handle class imbalance
    random_state=42,
    n_jobs=-1,
)

# Train/eval with stratified 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_weighted')

# After training, export feature importances for interpretability
clf.fit(X_train, y_train)
importances = dict(zip(FEATURE_COLUMNS, clf.feature_importances_))
```

**For embedded deployment**, export the trained RF to ONNX or implement as a lookup table (sklearn trees → nested if/else in C).

---

## 4. Implementation Requirements

### 4.1. File Structure
```
steering_classifier/
├── __init__.py
├── features.py          # Feature extraction (all F1-F11 + extras)
├── cascade.py           # Heuristic cascade classifier (Section 3A)
├── train_rf.py          # Random forest training pipeline (Section 3B)
├── filters.py           # IIR filter utilities for freq_energy_ratio
├── config.py            # All thresholds and calibration constants
├── types.py             # Frame dataclass, Event dataclass, ClassificationResult
├── evaluate.py          # Metrics: accuracy, precision, recall, F1, confusion matrix
├── tests/
│   ├── test_features.py     # Unit tests for each feature extractor
│   ├── test_cascade.py      # Unit tests for cascade classifier
│   ├── test_filters.py      # Unit tests for IIR filters
│   └── test_synthetic.py    # Integration tests with synthetic events
└── README.md
```

### 4.2. Config / Calibration Constants

Put all thresholds in `config.py` so they can be tuned without code changes:

```python
# config.py
from dataclasses import dataclass

@dataclass
class ClassifierConfig:
    # Sampling
    sample_rate_hz: float = 100.0
    
    # Feature thresholds
    torque_noise_floor_nm: float = 0.3      # ignore torque below this
    
    # Torque rate
    torque_rate_definite_mechanical: float = 80.0   # Nm/s
    torque_rate_likely_mechanical: float = 50.0      # Nm/s
    torque_rate_definite_driver: float = 20.0        # Nm/s
    
    # Duration
    max_pothole_length_m: float = 2.5
    min_duration_for_freq_analysis_s: float = 0.2    # 20 samples
    min_duration_for_correlation_s: float = 0.1       # 10 samples
    
    # Sign consistency
    sign_consistency_mechanical: float = 0.60
    sign_consistency_ambiguous: float = 0.75
    sign_consistency_driver: float = 0.90
    
    # Zero crossing rate
    zcr_mechanical_hz: float = 12.0
    zcr_driver_hz: float = 4.0
    
    # Kurtosis
    kurtosis_impulsive: float = 6.0
    kurtosis_smooth: float = 4.0
    
    # Longitudinal shock
    a_ego_shock_threshold: float = 1.5      # m/s²
    shock_max_duration_s: float = 0.4
    
    # Torque-angle phase
    phase_mechanical_threshold: float = 0.1
    phase_driver_threshold: float = 0.5
    
    # Torque-lat-accel correlation
    corr_strong_driver: float = 0.6
    corr_moderate_driver: float = 0.3
    corr_mechanical: float = 0.1
    
    # Frequency energy ratio
    freq_low_band: tuple = (0.5, 3.0)       # Hz — driver band
    freq_high_band: tuple = (5.0, 40.0)     # Hz — road disturbance band
    freq_ratio_driver: float = 3.0
    freq_ratio_mechanical: float = 1.0
    
    # Lateral accel residual
    lat_residual_strong_driver: float = 1.0  # m/s²
    lat_residual_moderate_driver: float = 0.5
    lat_residual_mechanical: float = 0.2
    
    # Cascade early-exit thresholds
    stage2_mechanical_exit: float = 4.0
    stage2_driver_exit: float = 3.0
```

### 4.3. Synthetic Test Events

Create these synthetic events for unit and integration testing:

```python
def make_pothole_event(speed_mps=25.0, duration_s=0.06, peak_torque_nm=8.0):
    """
    Simulates pothole: short, high dT/dt, oscillatory torque,
    longitudinal shock, no lateral accel change.
    - Torque: damped sinusoid at ~15 Hz
    - a_ego: spike > 2 m/s²
    - actual_lateral_accel ≈ desired_lateral_accel
    """
    ...

def make_driver_lane_change(speed_mps=25.0, duration_s=3.0, peak_torque_nm=4.0):
    """
    Simulates intentional lane change: smooth torque ramp,
    unidirectional, correlated with lateral accel.
    - Torque: half-sine envelope, ~0.5 Hz
    - Lateral accel follows torque with 100-200 ms lag
    - No longitudinal shock
    """
    ...

def make_driver_correction(speed_mps=15.0, duration_s=0.8, peak_torque_nm=2.0):
    """
    Simulates small lane-keeping correction: short but smooth,
    consistent sign, moderate torque.
    """
    ...

def make_curb_impact(speed_mps=8.0, duration_s=0.15, peak_torque_nm=12.0):
    """
    Simulates curb strike: very high torque rate, brief,
    strong longitudinal shock, large yaw disturbance.
    """
    ...

def make_rough_road_segment(speed_mps=20.0, duration_s=0.5, rms_torque_nm=1.5):
    """
    Simulates extended rough road: sustained oscillatory torque,
    moderate ZCR, no single sharp impulse.
    - Torque: band-limited noise 8-25 Hz
    """
    ...

def make_edge_case_emergency_swerve(speed_mps=30.0, duration_s=0.4, peak_torque_nm=6.0):
    """
    Edge case: fast driver reaction that looks mechanical.
    - Short duration, high torque rate
    - BUT: consistent sign, high torque-lat-accel correlation,
      torque leads angle, large lat accel residual
    - Should be classified as "driver"
    """
    ...
```

**Expected classifications:**
| Synthetic Event | Expected Label | Key Discriminating Features |
|---|---|---|
| `make_pothole_event` | mechanical | high dT/dt, oscillatory, brief, longitudinal shock, low corr |
| `make_driver_lane_change` | driver | slow dT/dt, sustained, high sign consistency, high corr |
| `make_driver_correction` | driver | moderate dT/dt, consistent sign, positive phase, moderate corr |
| `make_curb_impact` | mechanical | extreme dT/dt, brief, longitudinal shock, near-zero corr |
| `make_rough_road_segment` | mechanical | high ZCR, oscillatory, low corr, high freq energy |
| `make_edge_case_emergency_swerve` | driver | high dT/dt BUT high corr, consistent sign, torque leads |

### 4.4. Evaluation Metrics

In `evaluate.py`, compute at minimum:
- Accuracy, precision, recall, F1 for each class
- Confusion matrix
- Feature importance ranking (for RF)
- Per-speed-band accuracy (0–10, 10–20, 20–30, 30+ m/s)
- Latency histogram (how many ms into the event before confident classification)

### 4.5. Important Implementation Notes

1. **NaN handling**: For events shorter than `min_duration_for_freq_analysis_s`, set `freq_energy_ratio = None`. For events shorter than `min_duration_for_correlation_s`, set `torque_lat_accel_corr = None`. The cascade classifier must handle `None` features gracefully (skip that scoring block).

2. **Filter edge effects**: The IIR bandpass filters need a few samples of settling time. For the freq energy ratio, discard the first 5 samples of filter output. For events shorter than 20 samples total, skip frequency analysis entirely.

3. **Default to "driver" on ambiguity**: False negatives (classifying a real driver intervention as mechanical) are more safety-critical than false positives. When scores are tied or evidence is weak, return `("driver", 0.5)`.

4. **The cascade should be usable incrementally**: In a real-time system, features become available as more samples arrive. Structure the code so `classify_event` can be called with partial features (Tier 1 only, Tier 1+2, or all tiers) and emit early classifications with lower confidence.

5. **Pearson correlation edge case**: If all values in either array are constant (zero variance), `pearsonr` will return NaN. Handle this by returning 0.0 (no correlation).

6. **Use `scipy.signal.butter(..., output='sos')` for numerical stability** in the IIR filters, not transfer function (`ba`) form.

---

## 5. Comparison to Original 5-Rule Classifier

For reference, here is the mapping from the old rules to the new features:

| Old Rule | New Equivalent | Change |
|---|---|---|
| Rule 1: duration < 0.15s | F10: speed-adaptive threshold | Now `L_max / v_ego` instead of fixed 0.15s |
| Rule 2: \|a_ego\| > 1.5 AND dur < 0.4s | F6: has_longitudinal_shock | Retained as-is |
| Rule 3: sign consistency < 0.65 | F3 + F4: sign consistency + ZCR | Enhanced with zero-crossing rate |
| Rule 4: peak \|dT/dt\| > 500 Nm/s | F1: peak_torque_rate_nm_s | **Threshold lowered from 500 to 50–80 Nm/s** |
| Rule 5: speed > 20 AND dur < 0.4s | F10: speed-adaptive threshold | Continuous scaling replaces binary |
| *(missing)* | F8: torque–lat-accel correlation | **New — strongest single feature** |
| *(missing)* | F9: freq energy ratio | **New — exploits spectral gap** |
| *(missing)* | F7: torque-angle phase | **New — causal direction** |
| *(missing)* | F5: torque kurtosis | **New — impulse detection** |
| *(missing)* | F11: lateral accel residual | **New — path deviation** |

The old system scored 0–5 binary. The new cascade uses weighted continuous scoring with three stages of increasing latency and a configurable threshold structure.
