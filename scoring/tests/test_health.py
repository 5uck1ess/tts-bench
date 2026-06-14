import numpy as np

from scoring.health import (CLIP_LEVEL, GAP_SEC, measure, flags_from)

SR = 24000


def _tone(seconds=1.0, amp=0.3, hz=200):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _flags(x):
    return flags_from(measure(x.astype(np.float32), SR))


def test_clean_tone_is_clean():
    assert _flags(_tone()) == ""


def test_clipping_flags_clip():
    overdriven = np.clip(_tone() * 5.0, -1.0, 1.0)
    assert "clip" in _flags(overdriven)


def test_near_silent_flags_silent():
    assert _flags(_tone(amp=0.0005)) == "silent"


def test_interior_gap_flags_gap():
    gap = np.concatenate([_tone(), np.zeros(int(2 * SR), np.float32), _tone()])
    assert _flags(gap) == "gap"


def test_leading_trailing_silence_is_not_a_gap():
    pad = np.zeros(int(2 * SR), np.float32)
    edge = np.concatenate([pad, _tone(), pad])
    assert "gap" not in _flags(edge)


def test_silent_clip_is_not_also_a_gap():
    # an all-silent take is "silent", never "silent;gap"
    assert _flags(_tone(amp=0.0005, seconds=3.0)) == "silent"


def test_measure_reports_peak_and_clip_frac():
    m = measure(np.ones(SR, np.float32) * CLIP_LEVEL, SR)
    assert m["peak"] >= CLIP_LEVEL and m["clip_frac"] == 1.0


def test_gap_threshold_boundary():
    # a gap just under the threshold should not flag
    short = np.concatenate([_tone(), np.zeros(int((GAP_SEC - 0.4) * SR), np.float32), _tone()])
    assert "gap" not in _flags(short)
