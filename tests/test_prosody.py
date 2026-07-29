import numpy as np
import parselmouth

from scoring.prosody import analyze_sound


def _pitched_tone(span_st, duration_s=3.0, start_hz=170.0, sample_rate=24000):
    """Sine tone whose instantaneous frequency moves linearly in semitones."""
    n_samples = int(duration_s * sample_rate)
    normalized_time = np.arange(n_samples, dtype=np.float64) / n_samples
    frequency = start_hz * np.power(2.0, span_st * normalized_time / 12.0)
    phase = 2.0 * np.pi * np.cumsum(frequency) / sample_rate
    samples = 0.25 * np.sin(phase)
    return parselmouth.Sound(samples, sampling_frequency=sample_rate)


def test_falling_pitch_has_negative_declination():
    metrics = analyze_sound(_pitched_tone(-7.0))
    assert metrics.n_voiced >= 30
    assert metrics.declination_slope < -3.0


def test_flat_pitch_has_near_zero_declination():
    metrics = analyze_sound(_pitched_tone(0.0))
    assert metrics.n_voiced >= 30
    assert abs(metrics.declination_slope) < 0.2


def test_rising_pitch_has_positive_declination():
    metrics = analyze_sound(_pitched_tone(7.0))
    assert metrics.n_voiced >= 30
    assert metrics.declination_slope > 3.0


def test_too_few_voiced_frames_returns_nan_metrics():
    metrics = analyze_sound(_pitched_tone(-7.0, duration_s=0.15))
    assert metrics.n_voiced < 30
    assert np.isnan(metrics.declination_slope)
    assert np.isnan(metrics.terminal_slope)
    assert np.isnan(metrics.f0_range_st)
    assert np.isnan(metrics.f0_sd_st)
