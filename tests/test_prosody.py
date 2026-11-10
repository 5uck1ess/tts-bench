import numpy as np
import pandas as pd
import parselmouth

from scoring.prosody import _clean_voiced_frames, analyze_sound
from scripts.prosody_report import _language_for_prompt, _per_model_frame


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


def test_terminal_slope_is_in_semitones_per_second():
    # -7 st over 3 s is about -2.33 st/s everywhere, including the last 0.3 s.
    metrics = analyze_sound(_pitched_tone(-7.0, duration_s=3.0))
    assert -3.0 < metrics.terminal_slope < -1.7


def test_terminal_slope_same_units_for_short_and_long_clips():
    # Same st/s rate at two durations must give about the same terminal slope
    # (the old normalized-time unit scaled with clip length).
    short = analyze_sound(_pitched_tone(-3.0, duration_s=1.5))
    long = analyze_sound(_pitched_tone(-12.0, duration_s=6.0))
    assert abs(short.terminal_slope - long.terminal_slope) < 0.6


def test_terminal_slope_nan_when_ending_is_unvoiced():
    # Tone, 0.5 s silence, then a 50 ms voiced blip: the last 0.3 s of the
    # voiced span has too few real frames, so no slope is drawn across the gap.
    sr = 24000
    tone = _pitched_tone(-5.0, duration_s=2.0, sample_rate=sr).values[0]
    gap = np.zeros(int(0.5 * sr))
    blip = _pitched_tone(0.0, duration_s=0.05, sample_rate=sr).values[0]
    sound = parselmouth.Sound(
        np.concatenate([tone, gap, blip]), sampling_frequency=sr
    )
    metrics = analyze_sound(sound)
    # The final 0.3 s of the voiced span holds only the 50 ms blip.
    assert metrics.n_voiced >= 30
    assert np.isnan(metrics.terminal_slope)


def test_multi_frame_octave_plateau_is_fully_rejected():
    f0 = np.full(60, 200.0)
    f0[25:31] = 400.0  # six-frame octave-doubling plateau
    indexes, values, rejected = _clean_voiced_frames(f0)
    assert rejected >= 6
    assert np.all(np.abs(12.0 * np.log2(values / 200.0)) < 1.0)
    assert not np.isin(np.arange(25, 31), indexes).any()


def test_real_intonation_is_not_rejected():
    # A smooth 8-semitone fall over 60 frames is real prosody, not an error.
    f0 = 220.0 * np.power(2.0, np.linspace(0.0, -8.0, 60) / 12.0)
    _, _, rejected = _clean_voiced_frames(f0)
    assert rejected == 0


def test_spanish_prompt_has_a_language():
    assert _language_for_prompt("6") == "es"
    assert _language_for_prompt("5") == "fr"
    assert _language_for_prompt("1") == "en"


def test_per_model_terminal_uses_median_of_abs_not_abs_of_median():
    rows = []
    # Model A alternates big rises and falls: signed median ~0, abs median 20.
    for slope in (-20.0, -20.0, 20.0, 20.0):
        rows.append(("A", "default", slope))
    # Model B is consistently gentle.
    for slope in (-5.0, -5.0, -5.0, -5.0):
        rows.append(("B", "default", slope))
    # Same model in cloning mode must be its own row.
    rows.append(("B", "cloning", -1.0))
    valid = pd.DataFrame(
        [
            {
                "set": "synthetic",
                "language": "en",
                "model": model,
                "mode": mode,
                "terminal_slope": slope,
                "declination_slope": -1.0,
                "join_source": "scores_csv",
            }
            for model, mode, slope in rows
        ]
    )
    frame = _per_model_frame(valid, "en").set_index(["model", "mode"])
    assert frame.loc[("A", "default"), "terminal_abs"] == 20.0
    assert frame.loc[("B", "default"), "terminal_abs"] == 5.0
    assert frame.loc[("B", "cloning"), "n"] == 1
    # A must not rank as the "flattest" model.
    assert (
        frame.loc[("A", "default"), "terminal_rank"]
        > frame.loc[("B", "default"), "terminal_rank"]
    )
