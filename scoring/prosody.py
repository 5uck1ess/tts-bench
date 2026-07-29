"""Temporal F0 metrics for human and synthetic speech.

Praat's autocorrelation pitch tracker is run twice: a broad pass estimates the
speaker's median F0, then a narrower per-clip floor/ceiling is used for the
measurement pass.  Unvoiced frames are discarded.  Voiced runs receive a short
median filter and any remaining adjacent jump larger than one octave is
rejected before the contour is converted to semitones.

Slopes are expressed in semitones per normalized utterance (normalized time
runs from 0 to 1), not semitones per second.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import parselmouth


N_POINTS = 100
MIN_VOICED = 30
TIME_STEP_S = 0.01
OCTAVE_ST = 12.0
_BROAD_FLOOR_HZ = 50.0
_BROAD_CEILING_HZ = 800.0


@dataclass(frozen=True)
class ProsodyMetrics:
    """Per-utterance measurements returned by :func:`analyze_sound`."""

    declination_slope: float
    terminal_slope: float
    f0_range_st: float
    f0_sd_st: float
    duration_s: float
    voiced_frac: float
    n_voiced: int
    n_octave_rejected: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pitch_values(
    sound: parselmouth.Sound,
    pitch_floor: float,
    pitch_ceiling: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return Praat frame times, frequencies, and total pitch-frame count."""
    pitch = sound.to_pitch_ac(
        time_step=TIME_STEP_S,
        pitch_floor=float(pitch_floor),
        pitch_ceiling=float(pitch_ceiling),
    )
    frequencies = np.asarray(
        pitch.selected_array["frequency"], dtype=np.float64
    )
    times = np.asarray(pitch.xs(), dtype=np.float64)
    return times, frequencies, int(frequencies.size)


def infer_pitch_bounds(sound: parselmouth.Sound) -> tuple[float, float]:
    """Infer a conservative Praat floor/ceiling for one voice/clip.

    The broad pass is used only to locate the voice's center.  A 2.5x band on
    either side of the median comfortably spans ordinary intonation while
    discouraging octave-halving/doubling candidates.
    """
    _, broad_f0, _ = _pitch_values(
        sound, _BROAD_FLOOR_HZ, _BROAD_CEILING_HZ
    )
    broad_f0 = broad_f0[np.isfinite(broad_f0) & (broad_f0 > 0.0)]
    if broad_f0.size == 0:
        return 75.0, 600.0

    median_hz = float(np.median(broad_f0))
    floor = float(np.clip(median_hz / 2.5, _BROAD_FLOOR_HZ, 180.0))
    ceiling = float(np.clip(median_hz * 2.5, 250.0, _BROAD_CEILING_HZ))
    if ceiling <= floor * 1.5:
        ceiling = min(_BROAD_CEILING_HZ, floor * 2.5)
    return floor, ceiling


def _median_filter_run(values: np.ndarray, width: int = 5) -> np.ndarray:
    """Median-filter one contiguous voiced run without crossing silence."""
    if values.size < 3:
        return values.copy()
    width = min(width, values.size if values.size % 2 else values.size - 1)
    if width < 3:
        return values.copy()
    radius = width // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=-1)


def _clean_voiced_frames(
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return kept frame indexes/F0 after octave-artifact guarding.

    A short median filter is confined to contiguous voiced runs.  If an
    adjacent-frame jump still exceeds one octave, the member of that pair
    farther from the run median is rejected.
    """
    voiced = np.isfinite(frequencies) & (frequencies > 0.0)
    voiced_indexes = np.flatnonzero(voiced)
    if voiced_indexes.size == 0:
        return voiced_indexes, np.empty(0, dtype=np.float64), 0

    split_at = np.flatnonzero(np.diff(voiced_indexes) > 1) + 1
    runs = np.split(voiced_indexes, split_at)
    kept_indexes: list[np.ndarray] = []
    kept_values: list[np.ndarray] = []
    rejected = 0

    for run in runs:
        raw = frequencies[run]
        filtered = _median_filter_run(raw)
        semitones = 12.0 * np.log2(filtered)
        keep = np.ones(run.size, dtype=bool)
        run_median = float(np.median(semitones))

        for left in np.flatnonzero(np.abs(np.diff(semitones)) > OCTAVE_ST):
            right = left + 1
            if not (keep[left] and keep[right]):
                continue
            left_dev = abs(float(semitones[left]) - run_median)
            right_dev = abs(float(semitones[right]) - run_median)
            keep[right if right_dev >= left_dev else left] = False

        kept_indexes.append(run[keep])
        kept_values.append(filtered[keep])
        rejected += int((~keep).sum())

    return (
        np.concatenate(kept_indexes),
        np.concatenate(kept_values),
        rejected,
    )


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - float(np.mean(x))
    denominator = float(np.dot(x_centered, x_centered))
    if x.size < 2 or denominator <= 0.0:
        return float("nan")
    y_centered = y - float(np.mean(y))
    return float(np.dot(x_centered, y_centered) / denominator)


def _nan_metrics(
    duration_s: float,
    voiced_frac: float,
    n_voiced: int,
    n_octave_rejected: int,
) -> ProsodyMetrics:
    nan = float("nan")
    return ProsodyMetrics(
        declination_slope=nan,
        terminal_slope=nan,
        f0_range_st=nan,
        f0_sd_st=nan,
        duration_s=float(duration_s),
        voiced_frac=float(voiced_frac),
        n_voiced=int(n_voiced),
        n_octave_rejected=int(n_octave_rejected),
    )


def analyze_sound(
    sound: parselmouth.Sound,
    *,
    n_points: int = N_POINTS,
    min_voiced: int = MIN_VOICED,
    pitch_floor: float | None = None,
    pitch_ceiling: float | None = None,
) -> ProsodyMetrics:
    """Compute temporal F0 and dispersion metrics for one utterance.

    When fewer than ``min_voiced`` cleaned voiced frames remain, all four F0
    metrics are NaN.  Duration, voiced fraction, and frame counts remain real
    so callers can count and diagnose those dropped utterances.
    """
    if n_points < 6:
        raise ValueError("n_points must be at least 6")
    if min_voiced < 1:
        raise ValueError("min_voiced must be positive")

    duration_s = float(sound.get_total_duration())
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        return _nan_metrics(duration_s, 0.0, 0, 0)

    if pitch_floor is None or pitch_ceiling is None:
        inferred_floor, inferred_ceiling = infer_pitch_bounds(sound)
        pitch_floor = inferred_floor if pitch_floor is None else pitch_floor
        pitch_ceiling = inferred_ceiling if pitch_ceiling is None else pitch_ceiling
    if pitch_floor <= 0.0 or pitch_ceiling <= pitch_floor:
        raise ValueError("pitch ceiling must be greater than a positive floor")

    times, frequencies, n_frames = _pitch_values(
        sound, float(pitch_floor), float(pitch_ceiling)
    )
    indexes, cleaned_f0, n_octave_rejected = _clean_voiced_frames(frequencies)
    n_voiced = int(cleaned_f0.size)
    voiced_frac = float(n_voiced / n_frames) if n_frames else 0.0

    if n_voiced < min_voiced:
        return _nan_metrics(
            duration_s, voiced_frac, n_voiced, n_octave_rejected
        )

    reference_f0 = float(np.median(cleaned_f0))
    if not np.isfinite(reference_f0) or reference_f0 <= 0.0:
        return _nan_metrics(
            duration_s, voiced_frac, n_voiced, n_octave_rejected
        )

    semitones = 12.0 * np.log2(cleaned_f0 / reference_f0)
    voiced_times = times[indexes]
    voiced_span = float(voiced_times[-1] - voiced_times[0])
    if not np.isfinite(voiced_span) or voiced_span <= 0.0:
        return _nan_metrics(
            duration_s, voiced_frac, n_voiced, n_octave_rejected
        )
    # Normalize the observed F0 span, not container padding before/after it.
    # Internal unvoiced gaps retain their relative duration because the original
    # Praat timestamps are preserved.
    normalized_voiced_time = (voiced_times - voiced_times[0]) / voiced_span
    normalized_time = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    normalized_contour = np.interp(
        normalized_time, normalized_voiced_time, semitones
    )

    terminal = normalized_time >= 0.80
    return ProsodyMetrics(
        declination_slope=_ols_slope(normalized_time, normalized_contour),
        terminal_slope=_ols_slope(
            normalized_time[terminal], normalized_contour[terminal]
        ),
        f0_range_st=float(
            np.percentile(semitones, 95) - np.percentile(semitones, 5)
        ),
        f0_sd_st=float(np.std(semitones)),
        duration_s=duration_s,
        voiced_frac=voiced_frac,
        n_voiced=n_voiced,
        n_octave_rejected=n_octave_rejected,
    )


def analyze_wav(
    wav_path: str | Path,
    **kwargs: Any,
) -> ProsodyMetrics:
    """Load and analyze one WAV without modifying it."""
    return analyze_sound(parselmouth.Sound(str(wav_path)), **kwargs)
