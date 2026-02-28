from dataclasses import dataclass


@dataclass
class ClassifierConfig:
    # Sampling
    sample_rate_hz: float = 100.0

    # Feature thresholds
    torque_noise_floor_nm: float = 0.3      # ignore torque below this for sign consistency

    # Torque rate (Nm/s)
    torque_rate_definite_mechanical: float = 80.0
    torque_rate_likely_mechanical: float = 50.0
    torque_rate_definite_driver: float = 20.0

    # Duration
    max_pothole_length_m: float = 2.5       # used for speed-adaptive brevity threshold
    min_duration_for_freq_analysis_s: float = 0.2   # 20 samples at 100 Hz
    min_duration_for_correlation_s: float = 0.1      # 10 samples at 100 Hz

    # Sign consistency
    sign_consistency_mechanical: float = 0.60
    sign_consistency_ambiguous: float = 0.75
    sign_consistency_driver: float = 0.90

    # Zero-crossing rate (Hz)
    zcr_mechanical_hz: float = 12.0
    zcr_driver_hz: float = 4.0

    # Kurtosis
    kurtosis_impulsive: float = 6.0
    kurtosis_smooth: float = 4.0

    # Longitudinal shock
    a_ego_shock_threshold: float = 1.5      # m/s²
    shock_max_duration_s: float = 0.4

    # Torque-angle phase (Pearson r of dT vs dTheta)
    phase_mechanical_threshold: float = 0.1
    phase_driver_threshold: float = 0.5

    # Torque–lateral-accel correlation
    corr_strong_driver: float = 0.6
    corr_moderate_driver: float = 0.3
    corr_mechanical: float = 0.1

    # Frequency energy ratio
    freq_low_band: tuple = (0.5, 3.0)       # Hz — driver content
    freq_high_band: tuple = (5.0, 40.0)     # Hz — road disturbance content
    freq_filter_warmup_samples: int = 5     # discard first N samples of IIR output
    freq_ratio_driver: float = 3.0
    freq_ratio_mechanical: float = 1.0

    # Lateral accel residual (m/s²)
    lat_residual_strong_driver: float = 1.0
    lat_residual_moderate_driver: float = 0.5
    lat_residual_mechanical: float = 0.2

    # Stage 2 early-exit thresholds
    stage2_mechanical_exit: float = 4.0
    stage2_driver_exit: float = 3.0

    # Stage 1 early-exit thresholds
    stage1_mechanical_rate: float = 80.0    # Nm/s
    stage1_mechanical_duration: float = 0.05  # s
    stage1_driver_rate: float = 20.0        # Nm/s
    stage1_driver_duration: float = 0.5     # s
