"""Deterministic audio-health triage — reference-free, text-free, no ML.

Cheap numpy checks that catch *broken* takes the learned metrics miss: UTMOS
can score a clipped or half-silent clip as "fine", SIM only cares about timbre.
This flags mechanical defects you'd otherwise only catch by ear.

Idea borrowed from JaySpiffy/draft-to-take's audio_quality_judge (SpeechBrain +
deterministic health checks). We keep only the deterministic half; UTMOS/WER/SIM
already cover the learned half.

`HealthScorer.score(wav)` returns a compact flag string ("" = clean, else
";"-joined flags). `HealthScorer.detail(wav)` returns the raw measurements so
thresholds can be tuned against real clips. Flags:

    clip    hard digital clipping (a run of full-scale samples)
    silent  whole take is near-silent (dead generation)
    gap     long silence *inside* the take (dropout / truncation / breakup)

`max_jump` (largest sample-to-sample step) is measured and reported but NOT
flagged: across the published corpus normal speech transients (plosives,
sibilants at 24 kHz) routinely step >0.5, so it can't separate a real click/pop
from ordinary speech without unacceptable false positives. Left in `detail()`
as an informational field only.

WPM (rushed/slow) is intentionally omitted here — it needs the prompt text and
belongs with the WER path, which already has it.
"""

import numpy as np

# Thresholds — tuned against the published gh-pages clips (see __main__ demo).
CLIP_FRAC = 1.5e-3   # >0.15% of samples at full scale ⇒ "clip"
CLIP_LEVEL = 0.999   # |x| at/above this counts as a clipped sample
SILENT_RMS = 3.0e-3  # whole-clip RMS below this (~ -50 dBFS) ⇒ "silent"
GAP_SEC = 1.5        # interior silence longer than this ⇒ "gap" (corpus p99≈1.9s)
FRAME_MS = 20.0      # analysis frame for the gap scan


def _load_mono(wav_path):
    import soundfile as sf  # lazy: keeps measure()/flags_from() importable with only numpy
    data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def _frame_rms(x, sr, frame_ms=FRAME_MS):
    n = max(1, int(sr * frame_ms / 1000.0))
    if x.size < n:
        return np.array([np.sqrt(np.mean(x * x))] if x.size else [0.0])
    trimmed = x[: x.size - (x.size % n)]
    frames = trimmed.reshape(-1, n)
    return np.sqrt(np.mean(frames * frames, axis=1))


def _longest_interior_gap_sec(x, sr):
    """Seconds of the longest low-energy run that is NOT leading/trailing silence
    (those are normal). Returns 0.0 if the clip is effectively all silence."""
    rms = _frame_rms(x, sr)
    if rms.size == 0:
        return 0.0
    speech_floor = 0.06 * float(np.percentile(rms, 90))  # relative to loud frames
    speech_floor = max(speech_floor, 1.5e-3)             # absolute noise guard
    loud = np.where(rms > speech_floor)[0]
    if loud.size < 2:
        return 0.0  # no real speech to bracket — handled by the "silent" check
    interior = rms[loud[0]: loud[-1] + 1] <= speech_floor
    best = run = 0
    for q in interior:
        run = run + 1 if q else 0
        best = max(best, run)
    return best * (FRAME_MS / 1000.0)


def measure(x, sr):
    if x.size == 0:
        return {"peak": 0.0, "rms": 0.0, "clip_frac": 0.0, "max_jump": 0.0, "gap_sec": 0.0}
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    clip_frac = float(np.mean(np.abs(x) >= CLIP_LEVEL))
    max_jump = float(np.max(np.abs(np.diff(x)))) if x.size > 1 else 0.0
    return {"peak": peak, "rms": rms, "clip_frac": clip_frac,
            "max_jump": max_jump, "gap_sec": _longest_interior_gap_sec(x, sr)}


def flags_from(m):
    out = []
    if m["clip_frac"] > CLIP_FRAC:
        out.append("clip")
    if m["rms"] < SILENT_RMS:
        out.append("silent")
    elif m["gap_sec"] > GAP_SEC:          # a silent clip isn't also a "gap"
        out.append("gap")
    return ";".join(out)


class HealthScorer:
    name = "health"

    def score(self, wav_path):
        """Compact flag string; "" means the take passed every check."""
        x, sr = _load_mono(wav_path)
        return flags_from(measure(x, sr))

    def detail(self, wav_path):
        x, sr = _load_mono(wav_path)
        m = measure(x, sr)
        m["flags"] = flags_from(m)
        return m


def _demo(argv):
    """Scan published clips, print the measurement distribution + every flagged
    clip, so thresholds can be set from real data. Not part of scoring."""
    import glob
    import os

    root = argv[0] if argv else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_gh-pages")
    wavs = sorted(w for d in ("windows-default", "windows-cloning",
                              "linux-default", "linux-cloning")
                  for w in glob.glob(os.path.join(root, d, "*.wav")))
    if not wavs:
        print(f"No wavs under {root}")
        return 1
    sc = HealthScorer()
    rows, flagged = [], []
    for w in wavs:
        m = sc.detail(w)
        rows.append(m)
        if m["flags"]:
            rel = os.path.relpath(w, root).replace("\\", "/")
            flagged.append((rel, m))

    def pct(key, p):
        return float(np.percentile([r[key] for r in rows], p))

    print(f"Scanned {len(rows)} clips under {os.path.relpath(root)}\n")
    print("Measurement distribution (p50 / p90 / p99 / max):")
    for k in ("peak", "rms", "clip_frac", "max_jump", "gap_sec"):
        vals = [r[k] for r in rows]
        print(f"  {k:10s} {pct(k,50):.4f} / {pct(k,90):.4f} / "
              f"{pct(k,99):.4f} / {max(vals):.4f}")
    print(f"\nThresholds: clip_frac>{CLIP_FRAC}  rms<{SILENT_RMS}  gap>{GAP_SEC}s "
          f"(max_jump shown for info, not flagged)")
    print(f"\nFlagged {len(flagged)}/{len(rows)} clips:")
    for rel, m in flagged:
        print(f"  [{m['flags']:18s}] {rel}  "
              f"(peak={m['peak']:.3f} rms={m['rms']:.4f} "
              f"clip={m['clip_frac']:.4f} jump={m['max_jump']:.3f} "
              f"gap={m['gap_sec']:.2f}s)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_demo(sys.argv[1:]))
