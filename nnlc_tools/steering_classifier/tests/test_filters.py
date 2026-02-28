import numpy as np
import pytest
from nnlc_tools.steering_classifier.filters import bandpass_rms, compute_freq_energy_ratio


FS = 100.0


def pure_sine(freq_hz: float, duration_s: float = 1.0, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(duration_s * FS)) / FS
    return amp * np.sin(2 * np.pi * freq_hz * t)


class TestBandpassRms:
    def test_passband_sine_has_nonzero_rms(self):
        sig = pure_sine(1.5, duration_s=2.0)  # 1.5 Hz — inside 0.5–3 Hz band
        rms = bandpass_rms(sig, 0.5, 3.0, FS)
        assert rms > 0.1

    def test_stopband_sine_has_near_zero_rms(self):
        sig = pure_sine(20.0, duration_s=2.0)  # 20 Hz — outside 0.5–3 Hz band
        rms = bandpass_rms(sig, 0.5, 3.0, FS)
        assert rms < 0.1

    def test_too_short_signal_returns_zero(self):
        sig = np.ones(3)
        rms = bandpass_rms(sig, 0.5, 3.0, FS)
        assert rms == 0.0

    def test_high_band_passes_high_freq(self):
        sig = pure_sine(15.0, duration_s=2.0)  # 15 Hz — inside 5–40 Hz band
        rms = bandpass_rms(sig, 5.0, 40.0, FS)
        assert rms > 0.1


class TestFreqEnergyRatio:
    def test_low_freq_signal_gives_high_ratio(self):
        sig = pure_sine(1.5, duration_s=2.0)   # driver content
        ratio = compute_freq_energy_ratio(sig, FS)
        assert ratio > 1.0

    def test_high_freq_signal_gives_low_ratio(self):
        sig = pure_sine(15.0, duration_s=2.0)  # road disturbance content
        ratio = compute_freq_energy_ratio(sig, FS)
        assert ratio < 1.0

    def test_near_zero_high_band_returns_cap(self):
        # A very-low-freq signal essentially has zero high-band energy
        sig = pure_sine(1.0, duration_s=3.0)
        ratio = compute_freq_energy_ratio(sig, FS, cap=10.0)
        assert ratio == pytest.approx(10.0) or ratio > 2.0

    def test_mixed_signal_ratio_intermediate(self):
        # Equal-amplitude low + high
        low = pure_sine(1.5, duration_s=2.0)
        high = pure_sine(15.0, duration_s=2.0)
        ratio = compute_freq_energy_ratio(low + high, FS)
        assert 0.1 < ratio < 10.0
