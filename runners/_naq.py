"""NAQ — Naturalness-Artifact Quotient v2. Per-wav objective quality score.

NAQ = 0.5 * ARTIFACT + 0.5 * NATURALNESS where:
  ARTIFACT    = mean of {HARM, BUZZ}                 (>= 1 non-null required)
  NATURALNESS = mean of {DYN, PROSODY, RHYTHM, PITCH_MVMT}  (>= 2 non-null required)

Sub-features (each normalized to [0, 100], higher = better):
  HARM:       HNR in dB, Praat-style (Boersma 1993), clipped to [0, 30]
  BUZZ:       1 - 4-8kHz spectral flatness (vocoder buzz band)
  DYN:        P95-P5 of frame RMS in dB, clipped to [0, 30]
  PROSODY:    std-dev of voiced F0 in semitones (ref 100 Hz), clipped to [0, 5]
  RHYTHM:     Shannon entropy of IOI histogram, normalized by log2(10)
  PITCH_MVMT: mean |delta F0| across adjacent voiced frames in semitones, clipped to [0, 1.5]

ARTIFACT captures absence-of-artifacts; NATURALNESS captures presence of positive
naturalness cues humans pick up. Both axes are required because acoustic-artifact
metrics alone (the v1 design) can't tell a clean-but-monotone synth from a
clean-and-expressive one.

A learned-MOS predictor (UTMOS/DNSMOS) was considered but dropped: portable
install across 18+ heterogeneous venvs (some no-torch CPU-only) is infeasible.
Voting-system ground truth is the eventual real metric; this is a best-effort
proxy until that ships.

Best-effort: if librosa/scipy are missing or the wav is unloadable,
score() returns nulls and the bench continues.
"""

import os
import sys

# Numba (pulled in transitively by librosa.pyin / onset_detect) segfaults on
# some arm64-macOS venvs when it loads stale or version-mismatched objects from
# its default on-disk cache: running pyin and onset_detect in the same process
# crashes with SIGSEGV (observed in the NeuTTS venv: numba 0.65 / llvmlite 0.47
# / librosa 0.11). Pinning a dedicated, writable cache dir forces a clean
# recompile and makes scoring deterministic. Must run before librosa/numba are
# first imported — safe here because every runner imports _naq before its model
# libs, and the librosa/numba imports below are lazy (inside the feature fns).
# See docs/known-issues.md ("NeuTTS — NAQ scoring segfault on Apple Silicon").
_NUMBA_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".numba_cache"
)
try:
    os.makedirs(_NUMBA_CACHE, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", _NUMBA_CACHE)
except OSError as e:
    # Read-only repo dir: fall back to numba's default cache (no worse than
    # before this guard existed); just note it so the cause isn't invisible.
    print(f"[_naq] could not set NUMBA_CACHE_DIR={_NUMBA_CACHE}: {e}", file=sys.stderr)

_SAFE = (ImportError, RuntimeError, AttributeError, OSError, ValueError)


def _voiced_f0(wav_path):
    """Return (sr, voiced_audio, f0_hz_array, valid_mask), or None.

    voiced_audio: concatenated voiced regions (mono float32).
    f0_hz_array: librosa.pyin output aligned to voiced_audio frames; NaN for unvoiced frames.
    valid_mask: boolean array same length as f0_hz_array, True where F0 is not NaN.

    Returns None if librosa unavailable, no voiced regions, or pyin returned no valid F0.
    Called by HARM, PROSODY, and PITCH_MVMT to avoid redundant pyin calls (~1 sec each).
    """
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        if y.size == 0:
            return None
        voiced_intervals = librosa.effects.split(y, top_db=30)
        if voiced_intervals.size == 0:
            return None
        voiced = np.concatenate([y[s:e] for s, e in voiced_intervals])
        f0, _voiced_flag, _ = librosa.pyin(voiced, fmin=50, fmax=500, sr=sr)
        valid = ~np.isnan(f0)
        if not valid.any():
            return None
        return sr, voiced, f0, valid
    except _SAFE as e:
        print(f"[_naq] voiced_f0 failed: {e}", file=sys.stderr)
        return None


def _hnr_score(sr, voiced, f0, valid):
    """Return median HNR in dB over voiced regions, or None.

    Uses Boersma 1993 formulation: HNR = 10 * log10(r / (1 - r))
    where r is the Praat-style normalized autocorrelation at the F0 lag.
    """
    try:
        import numpy as np
        f0_valid = f0[valid]
        if f0_valid.size == 0:
            return None
        f0_med = float(np.median(f0_valid))
        period = int(sr / f0_med)
        frame_len = period * 3
        if frame_len >= len(voiced):
            return None
        hop = period
        lo = max(1, period - period // 4)
        hi = min(frame_len - 2, period + period // 4)
        hnr_vals = []
        for start in range(0, len(voiced) - frame_len, hop):
            frame = voiced[start : start + frame_len]
            best_r = -999.0
            for lag in range(lo, hi + 1):
                x1 = frame[: frame_len - lag]
                x2 = frame[lag:]
                num = float(np.dot(x1, x2))
                denom = float(np.sqrt(np.dot(x1, x1) * np.dot(x2, x2)))
                if denom > 0:
                    r = num / denom
                    if r > best_r:
                        best_r = r
            if 0.0 < best_r < 1.0:
                hnr_vals.append(10.0 * float(np.log10(best_r / (1.0 - best_r))))
        if not hnr_vals:
            return None
        return float(np.median(hnr_vals))
    except _SAFE as e:
        print(f"[_naq] HNR failed: {e}", file=sys.stderr)
        return None


def _buzz_flatness(wav_path):
    """Return spectral flatness in 4-8 kHz, in [0, 1], or None."""
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import welch
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if y.size == 0 or sr < 16000:
            return None
        f, psd = welch(y, fs=sr, nperseg=min(2048, len(y)))
        band = (f >= 4000) & (f <= 8000)
        psd_band = psd[band]
        if psd_band.size == 0 or np.all(psd_band <= 0):
            return None
        geo = np.exp(np.mean(np.log(psd_band + 1e-12)))
        ari = np.mean(psd_band)
        flat = float(geo / ari) if ari > 0 else None
        return flat
    except _SAFE as e:
        print(f"[_naq] BUZZ failed: {e}", file=sys.stderr)
        return None


def _dyn_range_db(wav_path):
    """Return P95 - P5 of frame-level RMS in dB, or None.

    Captures dynamic range across the wav (loud emphasis vs quiet parts).
    20ms windows, 10ms hop. Requires >=10 frames.
    """
    try:
        import numpy as np
        import soundfile as sf
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if y.size == 0 or sr < 8000:
            return None
        win = int(sr * 0.020)
        hop = int(sr * 0.010)
        if win < 16 or len(y) < win * 10:
            return None
        n_frames = (len(y) - win) // hop + 1
        rms = np.empty(n_frames, dtype=np.float64)
        for i in range(n_frames):
            frame = y[i*hop : i*hop + win]
            rms[i] = float(np.sqrt(np.mean(frame * frame)))
        rms_db = 20.0 * np.log10(rms + 1e-9)
        return float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
    except _SAFE as e:
        print(f"[_naq] DYN failed: {e}", file=sys.stderr)
        return None


def _prosody_std_semi(f0, valid):
    """Return std-dev of voiced F0 in semitones (ref 100 Hz), or None.

    Requires >=10 valid F0 frames.
    """
    try:
        import numpy as np
        f0_valid = f0[valid]
        if f0_valid.size < 10:
            return None
        f0_semi = 12.0 * np.log2(f0_valid / 100.0)
        return float(np.std(f0_semi))
    except _SAFE as e:
        print(f"[_naq] PROSODY failed: {e}", file=sys.stderr)
        return None


def _rhythm_entropy(wav_path):
    """Return entropy of inter-onset intervals normalized to log2(10), or None.

    librosa.onset.onset_detect -> times -> IOI distribution histogram (10 bins,
    range 0.1-2.0 sec) -> Shannon entropy. Requires >=3 IOIs (i.e. >=4 onsets).
    """
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        if y.size == 0:
            return None
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        if len(onset_times) < 4:
            return None
        ioi = np.diff(onset_times)
        if len(ioi) < 3:
            return None
        hist, _ = np.histogram(ioi, bins=10, range=(0.1, 2.0))
        total = hist.sum()
        if total <= 0:
            return None
        p = hist[hist > 0] / total
        entropy = float(-np.sum(p * np.log2(p)))
        return entropy / np.log2(10.0)
    except _SAFE as e:
        print(f"[_naq] RHYTHM failed: {e}", file=sys.stderr)
        return None


def _pitch_movement_semi(f0, valid):
    """Return mean |delta F0| in semitones across adjacent voiced frames, or None.

    Requires >=10 adjacent voiced-pair frames (i.e. 10 transitions where both
    frames are voiced).
    """
    try:
        import numpy as np
        if valid.size < 2:
            return None
        f0_semi = 12.0 * np.log2(np.where(valid, f0, np.nan) / 100.0)
        adj_valid = valid[:-1] & valid[1:]
        if adj_valid.sum() < 10:
            return None
        deltas = np.abs(f0_semi[1:] - f0_semi[:-1])
        valid_deltas = deltas[adj_valid]
        return float(np.mean(valid_deltas))
    except _SAFE as e:
        print(f"[_naq] PITCH_MVMT failed: {e}", file=sys.stderr)
        return None


def score(wav_path):
    """Return {naq, naq_artifact, naq_naturalness} for a wav file.

    Composite NAQ = 0.5 * ARTIFACT + 0.5 * NATURALNESS where:
      ARTIFACT    = mean of {HARM, BUZZ} non-nulls (requires >=1)
      NATURALNESS = mean of {DYN, PROSODY, RHYTHM, PITCH_MVMT} non-nulls (requires >=2)

    All three returned fields are floats in [0, 100] or None.
    The composite is None if either macro is None.
    """
    out = {"naq": None, "naq_artifact": None, "naq_naturalness": None}
    try:
        if not os.path.exists(wav_path):
            return out
    except _SAFE:
        return out

    # Shared pyin output for HARM, PROSODY, PITCH_MVMT
    f0_data = _voiced_f0(wav_path)
    if f0_data is not None:
        sr, voiced, f0, valid = f0_data
        hnr_db = _hnr_score(sr, voiced, f0, valid)
        prosody_raw = _prosody_std_semi(f0, valid)
        pitch_mvmt_raw = _pitch_movement_semi(f0, valid)
    else:
        hnr_db = prosody_raw = pitch_mvmt_raw = None

    flat = _buzz_flatness(wav_path)
    dyn_raw = _dyn_range_db(wav_path)
    rhythm_raw = _rhythm_entropy(wav_path)

    # Normalize each feature to [0, 100]
    def _clip(v, denom):
        return None if v is None else round(max(0.0, min(100.0, v / denom * 100.0)), 1)

    harm       = _clip(hnr_db, 30.0)
    buzz       = None if flat is None else round(max(0.0, min(100.0, (1.0 - flat) * 100.0)), 1)
    dyn        = _clip(dyn_raw, 30.0)
    prosody    = _clip(prosody_raw, 5.0)
    rhythm     = _clip(rhythm_raw, 1.0)  # already in [0, 1] from entropy / log2(10)
    pitch_mvmt = _clip(pitch_mvmt_raw, 1.5)

    # ARTIFACT macro: requires >=1 non-null
    artifact_vals = [v for v in (harm, buzz) if v is not None]
    if len(artifact_vals) >= 1:
        out["naq_artifact"] = round(sum(artifact_vals) / len(artifact_vals), 1)

    # NATURALNESS macro: requires >=2 non-null
    naturalness_vals = [v for v in (dyn, prosody, rhythm, pitch_mvmt) if v is not None]
    if len(naturalness_vals) >= 2:
        out["naq_naturalness"] = round(sum(naturalness_vals) / len(naturalness_vals), 1)

    if out["naq_artifact"] is not None and out["naq_naturalness"] is not None:
        out["naq"] = round(0.5 * out["naq_artifact"] + 0.5 * out["naq_naturalness"], 1)
    return out


def _selftest():
    """Validate scoring on real speech, white noise, silence, and synthetic prosody."""
    import json
    import tempfile
    import numpy as np
    import soundfile as sf

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_wav = os.path.join(repo, "reference", "chris_hemsworth_15s.wav")

    failures = []

    # 1. Real speech
    if os.path.exists(real_wav):
        s = score(real_wav)
        print(f"real speech: {json.dumps(s)}")
        if s["naq"] is None:
            failures.append("real speech NAQ was None")
        elif s["naq"] < 10:
            failures.append(f"real speech NAQ {s['naq']} < 10")
    else:
        print(f"SKIP real speech (no {real_wav})")

    # 2. White noise: HARM None (no F0) so ARTIFACT = BUZZ alone.
    #    NATURALNESS may compute if >=2 of {DYN,RHYTHM} succeed (PROSODY+PITCH_MVMT None).
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        noise_path = f.name
    try:
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(48000 * 3)).astype("float32") * 0.1
        sf.write(noise_path, noise, 48000)
        s = score(noise_path)
        print(f"white noise: {json.dumps(s)}")
        if s["naq"] is not None and s["naq"] > 40:
            failures.append(f"white noise NAQ {s['naq']} > 40 (should be low)")
    finally:
        os.unlink(noise_path)

    # 3. Silence: every voiced/F0/onset feature returns None -> NATURALNESS None -> composite None.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        silence_path = f.name
    try:
        sf.write(silence_path, np.zeros(48000, dtype="float32"), 48000)
        s = score(silence_path)
        print(f"silence:     {json.dumps(s)}")
        if s["naq"] is not None:
            failures.append(f"silence NAQ was {s['naq']} (expected None)")
    finally:
        os.unlink(silence_path)

    # 4. Synthetic monotone tone (220 Hz sine, 3s): high HARM, low PROSODY/PITCH_MVMT.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        mono_path = f.name
    try:
        sr = 24000
        t = np.linspace(0, 3.0, sr * 3, endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype("float32")
        sf.write(mono_path, tone, sr)
        s = score(mono_path)
        print(f"monotone:    {json.dumps(s)}")
        # We expect ARTIFACT to be high (clean sine), NATURALNESS to be low (flat F0)
        if s["naq_artifact"] is not None and s["naq_naturalness"] is not None:
            if s["naq_naturalness"] > 30:
                failures.append(
                    f"monotone NATURALNESS {s['naq_naturalness']} > 30 (should be low for flat sine)"
                )
    finally:
        os.unlink(mono_path)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nself-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
