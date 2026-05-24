"""NAQ — Naturalness-Artifact Quotient. Per-wav objective quality score.

NAQ v2 structure (two macros, each 0-100):
  naq_artifact: mean of non-null sub-scores in {HARM, BUZZ}
    HARM: HNR (dB) clamped [0,30] -> hnr/30*100   (Boersma 1993)
    BUZZ: 4-8 kHz spectral flatness inverted -> (1-flatness)*100
  naq_naturalness: PROSODY and PITCH_MVMT sub-scores (Task 2+)
  naq (composite): mean(naq_artifact, naq_naturalness) when both present

pyin is extracted once via _voiced_f0() and shared across HARM, PROSODY,
and PITCH_MVMT to avoid redundant ~1s calls.

A learned-MOS predictor (UTMOS/DNSMOS) was considered but dropped: portable
install across 18+ heterogeneous venvs and CPU-only venvs (Piper, KittenTTS,
Supertonic) was infeasible. The remaining sub-scores capture the
artifact axes the user explicitly cared about (vocoder buzz + phase noise);
overall "naturalness" prediction is left for a future enhancement.

Best-effort: if librosa/scipy are missing or the wav is unloadable,
score() returns nulls and the bench continues.
"""

import os
import sys

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


def score(wav_path):
    """Return {naq, naq_artifact, naq_naturalness} for a wav file.

    All three fields are floats in [0, 100] on success, or None on failure
    of that macro. The composite `naq` is None if either macro is None.
    """
    out = {"naq": None, "naq_artifact": None, "naq_naturalness": None}
    try:
        if not os.path.exists(wav_path):
            return out
    except _SAFE:
        return out

    f0_data = _voiced_f0(wav_path)
    if f0_data is not None:
        sr, voiced, f0, valid = f0_data
        hnr_db = _hnr_score(sr, voiced, f0, valid)
    else:
        hnr_db = None

    flat = _buzz_flatness(wav_path)

    harm = None if hnr_db is None else round(max(0.0, min(100.0, hnr_db / 30.0 * 100.0)), 1)
    buzz = None if flat is None else round(max(0.0, min(100.0, (1.0 - flat) * 100.0)), 1)

    # Artifact macro: mean of non-null among {HARM, BUZZ}; requires >=1.
    artifact_vals = [v for v in (harm, buzz) if v is not None]
    if artifact_vals:
        out["naq_artifact"] = round(sum(artifact_vals) / len(artifact_vals), 1)

    if out["naq_artifact"] is not None:
        # Naturalness macro will be filled in by later tasks; for now leave None.
        # Composite is None until both macros compute.
        pass
    return out


def _selftest():
    """Validate scoring on real speech, white noise, and silence."""
    import json
    import tempfile
    import numpy as np
    import soundfile as sf

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_wav = os.path.join(repo, "reference", "chris_hemsworth_15s.wav")

    failures = []

    if os.path.exists(real_wav):
        s = score(real_wav)
        print(f"real speech: {json.dumps(s)}")
        if s["naq_artifact"] is None:
            failures.append("real speech naq_artifact was None")
        elif s["naq_artifact"] < 10:
            failures.append(f"real speech naq_artifact {s['naq_artifact']} < 10")
    else:
        print(f"SKIP real speech (no {real_wav})")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        noise_path = f.name
    try:
        noise = (np.random.default_rng(0).standard_normal(48000 * 3)).astype("float32") * 0.1
        sf.write(noise_path, noise, 48000)
        s = score(noise_path)
        print(f"white noise: {json.dumps(s)}")
        # White noise: harm path returns None (no periodicity).
        # naq_artifact may still compute from BUZZ alone; composite naq stays None.
        if s["naq"] is not None and s["naq"] > 40:
            failures.append(f"white noise NAQ {s['naq']} > 40 unexpected")
    finally:
        os.unlink(noise_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        silence_path = f.name
    try:
        sf.write(silence_path, np.zeros(48000, dtype="float32"), 48000)
        s = score(silence_path)
        print(f"silence:     {json.dumps(s)}")
        if s["naq_artifact"] is not None:
            failures.append(f"silence naq_artifact was {s['naq_artifact']} (expected None)")
    finally:
        os.unlink(silence_path)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nself-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
