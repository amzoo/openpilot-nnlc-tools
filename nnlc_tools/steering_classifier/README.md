# Steering Event Classifier

**Distinguishing genuine driver steering interventions from mechanical road disturbances in ADAS/autopilot override event logs.**

When a vehicle's autopilot detects steering torque that exceeds its control authority, it flags a `steering_pressed` override event. But not every override is a real driver intervention — potholes, road bumps, curb impacts, and rough surfaces can produce torque spikes that trigger the same flag. Misclassifying a pothole as a driver takeover (or vice versa) corrupts driver engagement metrics, safety analytics, and system tuning. This classifier separates the two.

---

## How It Works

The classifier extracts 11 signal features from each override event and runs them through a **three-stage cascade** that trades off latency against accuracy. Each stage can emit a final decision or pass to the next stage for more evidence.

```
Event frames (100 Hz)
        │
        ▼
┌─────────────────────────────┐
│  STAGE 1 — Fast Gate (10ms) │  Torque rate + duration only
│  Can resolve obvious cases  │  → "definite pothole" or "definite driver"
└──────────┬──────────────────┘
           │ ambiguous
           ▼
┌──────────────────────────────────┐
│  STAGE 2 — Confirmation (50ms)   │  + sign consistency, ZCR, kurtosis,
│  Weighted multi-feature scoring  │    longitudinal shock, torque-angle phase
└──────────┬───────────────────────┘
           │ still ambiguous
           ▼
┌──────────────────────────────────────┐
│  STAGE 3 — Contextual Gate (200ms)   │  + torque–lat-accel correlation,
│  Cross-signal correlation features   │    frequency energy ratio,
│  Resolves edge cases                 │    lateral accel residual
└──────────────────────────────────────┘
           │
           ▼
   ("driver" | "mechanical", confidence)
```

The system defaults to **"driver"** on ambiguity. Misclassifying a real driver intervention as mechanical is more safety-critical than the reverse.

---

## The 11 Features

Each feature's design is grounded in vehicle dynamics research and EPAS (Electric Power Assisted Steering) control literature. The features are grouped by the latency tier at which they become computable.

### Tier 1 — Fast Gate (1 sample / 10 ms)

**F1: Peak Torque Rate (Nm/s)** — The first derivative of steering torque is the fastest discriminator. Human neuromuscular dynamics act as a second-order low-pass filter on steering input, limiting intentional torque rate to roughly 20 Nm/s for normal maneuvers. Road impacts bypass the driver entirely and produce torque rates of 50–100+ Nm/s through direct mechanical force on the steering linkage.

The original classifier used a 500 Nm/s threshold, which is approximately 5–10× too conservative. The literature-calibrated range is 50–80 Nm/s for the mechanical/driver boundary.

*Sources:*
- Pick & Cole, "Neuromuscular dynamics in the driver–vehicle system," *Vehicle System Dynamics*, 44(sup1):511-522, 2006 — established the neuromuscular bandwidth limit constraining driver torque rate.
- Springer chapter on instrumented steering wheel characterization (2024) — measured unconscious reflex torque peaking at 0.13 ± 0.03 s vs. conscious steering at 0.28 ± 0.11 s, with reflex torque only ~20% of maximum voluntary.

**F2: Event Duration (seconds)** — Road impact transients last 20–200 ms depending on speed and obstacle geometry. Lane-keeping corrections span 0.5–3 s; lane changes take 2–5 s. Emergency evasive maneuvers occupy 0.3–2 s with high sustained torque.

*Sources:*
- Schinkel et al., "Driver Intervention Detection via Real-Time Transfer Function Estimation," *IEEE Trans. ITS*, 2021 — found reliable event classification requires minimum 0.2 s of data.

### Tier 2 — Confirmation Gate (5 samples / 50 ms)

**F3: Torque Sign Consistency** — Intentional steering is unidirectional: a driver turning left applies sustained negative (or positive) torque throughout the maneuver. Road disturbances produce oscillatory torque that alternates sign rapidly. The ratio of dominant-sign samples to total samples quantifies this.

*Sources:*
- Moreillon et al., "Hands On/Off Detection Based on EPS Sensors," *JTEKT Engineering Journal*, No. 1017E, 2020 — measured ±1–2 Nm oscillatory torque on cobblestone roads with hands off the wheel; ±3 Nm during automated driving on asphalt. Their Driver Torque Estimator inherently low-pass filters to reject oscillatory content, validating sign consistency as a classification signal.

**F4: Zero-Crossing Rate (Hz)** — A computationally efficient proxy for the oscillation frequency of the torque signal. Road disturbances excite steering at 8–20+ Hz (producing ZCR of 16–40 Hz); driver inputs stay below 2–4 Hz (ZCR < 4 Hz). This feature complements sign consistency by capturing oscillation frequency rather than just polarity balance.

*Sources:*
- Cole et al., "Real-time characterisation of driver steering behaviour," *Vehicle System Dynamics*, 56(10), 2018 — measured intentional driver steering at 0.2–2 Hz with road excitation above 2 Hz.
- Giacomin & Woo, "A study of the human ability to detect road surface type on the basis of steering wheel vibration feedback," *Proc. IMechE Part D*, 219(11), 2005 — found steering wheel vibration from road surface concentrated at 10–60 Hz, with 26–35 Hz most diagnostic.

**F5: Torque Kurtosis** — Excess kurtosis measures the "tailedness" of the torque distribution. A single sharp impulse (pothole strike) produces high kurtosis (>>3, leptokurtic). Smooth driver input produces approximately Gaussian or sub-Gaussian distributions (kurtosis ≈ 3). Extended rough road produces moderate kurtosis between the two extremes.

*Sources:*
- Standard statistical property used in vibration analysis and structural health monitoring. Applied here by analogy to the impulsive vs. smooth signal morphology established in the road excitation literature.

**F6: Longitudinal Shock** — A concurrent spike in longitudinal acceleration (|a_ego| > 1.5 m/s²) during a short event (<0.4 s) is a strong indicator of a vertical road impact. Intentional steering maneuvers do not produce significant longitudinal acceleration unless combined with braking, which can be disambiguated by duration.

*Sources:*
- Mednis et al., "Real Time Pothole Detection Using Android Smartphones with Accelerometers," *IEEE DCOSS*, 2011 — validated accelerometer-based pothole detection using vertical/longitudinal acceleration thresholds.
- Eriksson et al., "The Pothole Patrol," *MobiSys*, 2008 — demonstrated speed-dependent vertical acceleration filtering for road anomaly detection.

**F7: Torque-Angle Phase Relationship** — During intentional steering, the driver applies torque *first* and the wheel turns *in response* (torque leads angle). During a road disturbance, an external force turns the wheel *first* and the driver feels the resulting torque *after* (angle leads torque). The cross-correlation between torque rate and angle rate at zero lag captures this causal direction as a continuous [-1, 1] value.

*Sources:*
- Fundamental EPAS control principle. Validated indirectly by:
  - Chevrel, Mars et al., "Modelling human control of steering for the design of advanced driver assistance systems," *Annual Reviews in Control*, 47:249-261, 2019 — cybernetic driver model showing torque as the *cause* of heading change in intentional control.
  - GM Patent US 8,954,235 (2015) — compares measured vs. model-expected steering torque to detect driver override in lane centering, implicitly relying on the causal torque→angle relationship.

### Tier 3 — Contextual Gate (10–20 samples / 100–200 ms)

**F8: Torque–Lateral-Acceleration Correlation** — **The single most impactful feature missing from the original classifier.** When a driver steers intentionally, their applied torque produces a corresponding change in lateral acceleration — the two signals are tightly coupled (Pearson r > 0.6) with torque leading by 50–200 ms. During a road disturbance, a torque spike occurs *without* producing proportional lateral acceleration change (r ≈ 0), because the disturbance is absorbed by the suspension and steering system before it can alter the vehicle's lateral trajectory.

This feature is powerful because it exploits the physics of the vehicle rather than just the shape of the torque signal. A pothole can produce torque that "looks" intentional in magnitude and even sign consistency, but it will not produce correlated lateral acceleration.

*Sources:*
- comma.ai torqued lateral control documentation — confirms the fundamental torque-to-lateral-acceleration relationship used in production ADAS control.
- Hyundai Patent US2019/0077447 — models the driver torque as proportional to lateral acceleration for ADAS conflict detection.
- Zhou et al., "Driver Steering Intention Prediction for Human-Machine Shared Systems of Intelligent Vehicles Based on CNN-GRU Network," *Sensors*, 25(10):3224, MDPI, 2025 — feature importance analysis ranks steering torque and lateral acceleration as the top two discriminative inputs for intent prediction.

**F9: Frequency Energy Ratio (Low-Band / High-Band)** — The strongest *spectral* discriminator. Driver steering content lives at 0.5–3 Hz; road disturbance content concentrates at 5–40 Hz. Computing the RMS energy ratio of these two bands over a short window provides a single scalar that directly measures where the signal's power lies. Implemented with second-order Butterworth IIR bandpass filters in second-order-sections (SOS) form for numerical stability.

At 100 Hz sampling (Nyquist = 50 Hz), the full separation between driver and road bands is capturable. For events shorter than 200 ms (~20 samples), the bandpass filter edge effects dominate and this feature is not computed.

*Sources:*
- Cole et al. (2018) — 0.2–2 Hz driver band.
- Giacomin & Woo (2005) — 10–60 Hz road surface vibration band.
- Giacomin et al., "Effect of steering wheel acceleration frequency distribution on detection of road type," *Ingeniería Mecánica, Tecnología y Desarrollo*, 2013 — confirmed frequency distribution as the primary cue for road surface type discrimination.
- arXiv submission on steering feedback in dynamic driving simulators (2024) — confirmed 10–30 Hz as the road excitation band, noting content below 10 Hz would be perceived as external steering intervention.
- Giacomin & Onesti, "Frequency weighting for the evaluation of steering wheel rotational vibration," *International Journal of Industrial Ergonomics*, 34(2):89-97, 2004 — established frequency-dependent human sensitivity to steering wheel vibration.

**F10: Speed-Adaptive Duration Threshold** — Replaces the original fixed 0.15 s and "highway brief" (speed > 20 m/s AND duration < 0.4 s) rules. The physics is straightforward: pothole crossing time equals pothole length divided by vehicle speed (T = L/V). A 2.5 m pothole takes 250 ms to cross at 10 m/s but only 83 ms at 30 m/s. Using a calibratable maximum pothole length (default 2.5 m), the threshold adapts continuously to speed rather than using binary cutoffs.

*Sources:*
- SAE Paper 2015-01-0637, "Simulation of Vehicle Pothole Test and Techniques Used" — confirmed pothole size and vehicle speed as the two primary factors in impact severity, with duration scaling as L/V.
- ISO 8608:2016, "Mechanical vibration — Road surface profiles — Reporting of measured data" — classifies road surface roughness via power spectral density at reference spatial frequency, establishing that temporal excitation frequency = spatial frequency × speed.
- Bridgelall & Tolliver, "Characterisation of road bumps using smartphones," *European Transport Research Review*, 8:13, 2016 — demonstrated speed-dependent road anomaly detection thresholds.

**F11: Lateral Acceleration Residual** — The maximum absolute deviation between actual and desired lateral acceleration during the event. When a driver overrides the autopilot, they steer the vehicle away from its planned path, producing a large residual (>0.5 m/s²). A road disturbance produces a torque spike but the vehicle's trajectory remains close to the planned path (small residual), because the suspension absorbs the vertical/lateral perturbation before it significantly alters the vehicle's lateral dynamics.

*Sources:*
- Toyota Patent EP3659878B1 (Mitsumoto, 2021) — uses deviation between actual and model-expected yaw rate to detect external lateral disturbances, the rotational equivalent of the lateral acceleration residual.
- Euro NCAP Assessment Protocol SA v10.4.1 (2024) — specifies lane-keeping assist override at ≤3.5 Nm, implying that driver override produces measurable path deviation.

---

## Why Not Just Use Torque Threshold?

The original classifier and most production ADAS systems use simple torque or torque-rate thresholds. This fails for three reasons:

1. **Cobblestone and rough road surfaces produce sustained torque oscillations of ±1–3 Nm** — well above the hands-on detection threshold of 0.6–1.0 Nm — even with nobody touching the wheel (Moreillon et al., JTEKT 2020). A magnitude-only classifier would flag every cobblestone road as a driver intervention.

2. **Emergency swerves produce torque rates >50 Nm/s and durations <0.5 s**, overlapping with the mechanical disturbance signature in both rate and duration. Only the cross-signal features (torque–lat-accel correlation, torque-angle phase, lateral acceleration residual) can resolve this ambiguity, because an emergency swerve *moves the vehicle laterally* while a pothole does not.

3. **Road disturbance magnitude scales with speed** while duration shrinks. At highway speeds, a pothole impact can produce peak torque comparable to a gentle lane correction — but compressed into 40 ms with oscillatory sign pattern and high-frequency energy. Fixed thresholds optimized for one speed range systematically misclassify at others.

---

## Speed-Dependent Effects

Road disturbance signatures change systematically with vehicle speed through three coupled mechanisms:

**Duration shortens** — Crossing time T = L/V means the same 1 m pothole creates a 100 ms event at 10 m/s but a 33 ms event at 30 m/s.

**Peak magnitude increases** — Impact energy scales with V². At low speeds the tire conforms to pothole geometry, distributing forces over time. At high speeds the tire acts rigidly during the brief contact, concentrating forces into sharper impulses. Research on two-wheeled vehicles has demonstrated "wheel launch" (loss of ground contact) at 60 km/h over potholes that produced only moderate loads at 20 km/h.

**Excitation frequency increases** — A fixed spatial irregularity of wavelength L creates temporal excitation at f = V/L. At 30 m/s over a 0.5 m feature, the excitation frequency is 60 Hz — well above the driver steering band and into the wheel/suspension resonance range. This makes the frequency energy ratio (F9) increasingly discriminative at higher speeds.

*Sources:*
- ISO 8608:2016 — temporal frequency = spatial frequency × speed relationship.
- Wang et al., "Influence of Road Excitation and Steering Wheel Input on Vehicle System Dynamic Responses," *Applied Sciences*, 2017 — showed coupled speed-dependent effects of road excitation on lateral response.

---

## Alternative: Random Forest Classifier

The cascade heuristic is designed for interpretability and real-time production deployment. For offline analysis or as a potential production upgrade, a random forest trained on the same 11+ features typically achieves 84–95% accuracy on vehicle sensor classification tasks.

The implementation includes a training pipeline using scikit-learn's `RandomForestClassifier` with 30 trees at depth 7, balanced class weights, and 5-fold stratified cross-validation. After training, feature importances provide interpretable evidence for which signals carry the most weight, which can feed back into cascade threshold tuning.

*Sources:*
- Das, Khan & Ahmed, "Deep Learning Approach for Detecting Lane Change Maneuvers Using SHRP2 Naturalistic Driving Data," *Transportation Research Record*, 2023 — XGBoost + ResNet-18 achieved 98.8% recall and 95% accuracy on naturalistic driving data.
- Zhou et al. (MDPI *Sensors*, 2025) — CNN-GRU achieved RMSE reductions of ~32% vs. BP, ~21% vs. LSTM, ~25% vs. CNN alone for steering intention prediction. Feature importance: steering torque > steering angle > vehicle speed > lateral acceleration.
- Lightweight CAN-bus driver behavior classification (*Sensors/PMC*, 2020) — depth-wise convolution + augmented RNN deployed on NVIDIA Jetson Nano with real-time inference.

---

## Relevant Standards

| Standard | Relevance |
|---|---|
| **ISO 8608:2016** | Road surface profile classification (Classes A–H) via PSD at reference spatial frequency. Provides the physics for speed-dependent excitation frequency and magnitude scaling. |
| **ISO 11270:2014** | Steering feel — defines test procedures for steering system assessment including torque characteristics. |
| **UN ECE R79** | Lane-keeping assist limits: max lateral acceleration 3 m/s², max lateral jerk 5 m/s³. Hands-off warning escalation at 15 s → 30 s → deactivation. |
| **Euro NCAP SA Protocol v10.4.1 (2024)** | Driver override requirement: ≤3.5 Nm to override lane-keeping assist. |
| **SAE J3016** | Levels of driving automation; defines when driver override authority is required. |

---

## Datasets for Training and Validation

**commaSteeringControl** (comma.ai, Hugging Face) — ~12,500 hours of driving data with openpilot engaged across 275+ car models. Contains `steeringPressed` (binary override flag), `steer` (normalized torque), `steeringAngleDeg`, `vEgo`, `aEgo`, `latAccelDesired`, `latAccelSteeringAngle`. The `steeringPressed` events provide candidate labels but do not distinguish driver intent from road disturbance — which is exactly the classification gap this project addresses.

**comma2k19** (Schafer et al., arXiv:1812.05752) — 33+ hours with CAN bus data and 9-axis IMU on Honda Civic and Toyota RAV4.

**SHRP2 Naturalistic Driving Study** (FHWA/Virginia Tech) — large-scale naturalistic dataset with CAN signals; used in multiple lane change and driver behavior studies but requires data use agreement.

**Smartphone pothole datasets** — Multiple public datasets exist for accelerometer-based road anomaly detection (Kaggle, various research groups) which can provide road surface ground truth labels when cross-referenced with GPS coordinates.

*Note:* No single public dataset combines steering torque data with labeled road surface impact events. Building ground truth requires combining vertical acceleration thresholding on `a_ego`, the `steeringPressed` flag, GPS-linked road quality databases, and selective manual video review for ambiguous cases.

---

## Key Literature References

### Driver steering dynamics and bandwidth
1. Cole, D.J. et al., "Real-time characterisation of driver steering behaviour," *Vehicle System Dynamics*, 56(10), 2018. — Driver steering at 0.2–2 Hz.
2. Pick, A.J. & Cole, D.J., "Neuromuscular dynamics in the driver–vehicle system," *Vehicle System Dynamics*, 44(sup1):511-522, 2006. — Neuromuscular low-pass filtering limits torque rate.
3. Timings, J.P. & Cole, D.J., "A review of human sensory dynamics for application to models of driver steering and speed control," *Biological Cybernetics*, 110(2-3), 2016. — Driver feedback bandwidth ~1–2 Hz.
4. Chevrel, P., Mars, F. et al., "Modelling human control of steering for the design of advanced driver assistance systems," *Annual Reviews in Control*, 47:249-261, 2019. — Cybernetic driver model with visual anticipation + compensatory control.

### Road surface vibration and steering disturbance
5. Giacomin, J. & Woo, Y.J., "A study of the human ability to detect road surface type on the basis of steering wheel vibration feedback," *Proc. IMechE Part D*, 219(11), 2005. — Road vibration at 10–60 Hz, 26–35 Hz most diagnostic.
6. Giacomin, J. et al., "Effect of steering wheel acceleration frequency distribution on detection of road type," *Ingeniería Mecánica, Tecnología y Desarrollo*, 2013. — Frequency distribution as primary road surface cue.
7. Giacomin, J. & Onesti, L., "Frequency weighting for the evaluation of steering wheel rotational vibration," *International Journal of Industrial Ergonomics*, 34(2):89-97, 2004. — Frequency-dependent sensitivity to steering vibration.

### EPAS hands-on detection and disturbance rejection
8. Moreillon, L. et al., "Hands On/Off Detection Based on EPS Sensors," *JTEKT Engineering Journal*, No. 1017E, 2020. — ±1–2 Nm oscillatory torque on cobblestone; Driver Torque Estimator architecture.
9. Dornhege, C., Nolden, P. & Mayer, R., "Steering Torque Disturbance Rejection," *SAE Int. J. Veh. Dyn., Stab., and NVH*, 1(2):165-172, 2017. — Dual rack-force model for disturbance identification.
10. Yamamoto, K. et al., "Driver torque estimation in Electric Power Steering system using an H∞/H2 Proportional Integral Observer," *IEEE CDC*, 2015. — Observer bandwidth as implicit consistency time scale.

### Driver intent and intervention detection
11. Schinkel, W. et al., "Driver Intervention Detection via Real-Time Transfer Function Estimation," *IEEE Trans. ITS*, 2021. — Transfer-function approach; minimum 0.2 s for reliable classification.
12. Zhou, Y. et al., "Driver Steering Intention Prediction for Human-Machine Shared Systems of Intelligent Vehicles Based on CNN-GRU Network," *Sensors*, 25(10):3224, MDPI, 2025. — CNN-GRU RMSE reductions vs. LSTM/Transformer; feature importance ranking.
13. Das, A., Khan, M.N. & Ahmed, M.M., "Deep Learning Approach for Detecting Lane Change Maneuvers Using SHRP2 Naturalistic Driving Data," *Transportation Research Record*, 2023. — XGBoost + ResNet-18, 98.8% recall.

### Road anomaly detection
14. Mednis, A. et al., "Real Time Pothole Detection Using Android Smartphones with Accelerometers," *IEEE DCOSS*, 2011. — STDEV(Z) algorithm, ~90% true positive rate.
15. Bridgelall, R. & Tolliver, D., "Characterisation of road bumps using smartphones," *European Transport Research Review*, 8:13, 2016. — Speed-dependent anomaly detection thresholds.

### Vehicle dynamics and disturbance modeling
16. Abe, M. et al., "A yaw-moment control method based on a vehicle's lateral jerk information," *Vehicle System Dynamics*, 52(10), 2014. — Lateral jerk as intentional maneuver indicator.
17. ISO 8608:2016, "Mechanical vibration — Road surface profiles — Reporting of measured data."
18. Toyota Patent EP3659878B1 (Mitsumoto, 2021) — Yaw rate residual for external disturbance detection.
19. GM Patent US 8,954,235 (2015) — Enhanced steering override detection during automated lane centering.

### Standards
20. UN ECE R79 — Lane-keeping assist lateral acceleration/jerk limits, hands-off detection timing.
21. Euro NCAP Assessment Protocol SA v10.4.1 (2024) — Override force requirements.
22. ISO 11270:2014 — Steering feel test procedures.