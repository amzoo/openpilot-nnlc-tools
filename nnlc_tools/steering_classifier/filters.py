"""IIR bandpass filter utilities for frequency energy ratio computation."""

import numpy as np
from scipy.signal import butter, sosfilt


def _make_bandpass_sos(low_hz: float, high_hz: float, fs: float, order: int = 2):
    """Build a 2nd-order Butterworth bandpass SOS filter."""
    nyq = fs / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 1.0 - 1e-4)
    return butter(order, [low, high], btype="bandpass", output="sos")


def bandpass_rms(signal: np.ndarray, low_hz: float, high_hz: float, fs: float = 100.0,
                 warmup_samples: int = 5) -> float:
    """Apply bandpass filter and return RMS energy of the filtered signal.

    Discards the first `warmup_samples` to reduce IIR edge-effect bias.
    Returns 0.0 if the signal is too short to filter.
    """
    if len(signal) < warmup_samples + 2:
        return 0.0
    sos = _make_bandpass_sos(low_hz, high_hz, fs)
    filtered = sosfilt(sos, signal)
    trimmed = filtered[warmup_samples:]
    if len(trimmed) == 0:
        return 0.0
    return float(np.sqrt(np.mean(trimmed ** 2)))


def compute_freq_energy_ratio(
    signal: np.ndarray,
    fs: float = 100.0,
    low_band: tuple = (0.5, 3.0),
    high_band: tuple = (5.0, 40.0),
    warmup_samples: int = 5,
    cap: float = 10.0,
) -> float:
    """Return low_band_rms / high_band_rms.

    - Driver inputs have most energy < 3 Hz → ratio >> 1
    - Road impacts have energy concentrated 5–40 Hz → ratio ≈ 1 or < 1
    - Returns `cap` when high-band energy is negligible (essentially all driver content)
    """
    low_rms = bandpass_rms(signal, low_band[0], low_band[1], fs, warmup_samples)
    high_rms = bandpass_rms(signal, high_band[0], high_band[1], fs, warmup_samples)
    if high_rms < 1e-6:
        return cap
    return float(low_rms / high_rms)
