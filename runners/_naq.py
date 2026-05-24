"""NAQ — Naturalness-Artifact Quotient. Per-wav objective quality score.

NAQ = 0.65 * HARM_score + 0.35 * BUZZ_score
where each sub-score is normalized to [0, 100].

Sub-scores:
  HARM_score: HNR (dB) clamped to [0, 30] -> hnr / 30 * 100
              (Boersma 1993, normalized autocorrelation at the F0 lag)
  BUZZ_score: 4-8 kHz spectral flatness inverted -> (1 - flatness) * 100
              (the vocoder "buzziness" band; flat spectra = artifacted)

A learned-MOS predictor (UTMOS/DNSMOS) was considered but dropped: portable
install across 18+ heterogeneous venvs and CPU-only venvs (Piper, KittenTTS,
Supertonic) was infeasible. The remaining two sub-scores capture the
artifact axes the user explicitly cared about (vocoder buzz + phase noise);
overall "naturalness" prediction is left for a future enhancement.

Best-effort: if librosa/scipy are missing or the wav is unloadable,
score() returns nulls and the bench continues.
"""

import os
import sys

_SAFE = (ImportError, RuntimeError, AttributeError, OSError, ValueError)


def _hnr_score(wav_path):
    """Return median HNR in dB over voiced regions, or None.

    Uses Boersma 1993 formulation: HNR = 10 * log10(r / (1 - r))
    where r is the Praat-style normalized autocorrelation at the F0 lag,
    computed per-frame as dot(x, x_shifted) / sqrt(energy(x) * energy(x_shifted)).
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
        f0_valid = f0[~np.isnan(f0)]
        if f0_valid.size == 0:
            return None
        f0_med = float(np.median(f0_valid))
        period = int(sr / f0_med)
        # Analyse in short frames (3 pitch periods) with Praat-style normalization.
        # r(tau) = dot(x[:N-tau], x[tau:]) / sqrt(energy(x[:N-tau]) * energy(x[tau:]))
        # This bounds r to (-1, 1) regardless of frame length.
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
    """Return {naq, naq_harm, naq_buzz} for a wav file.

    All three fields are floats in [0, 100] on success, or None on failure
    of that sub-score. The composite `naq` is None if either sub-score is None.
    """
    out = {"naq": None, "naq_harm": None, "naq_buzz": None}
    try:
        if not os.path.exists(wav_path):
            return out
    except _SAFE:
        return out

    hnr_db = _hnr_score(wav_path)
    flat = _buzz_flatness(wav_path)

    if hnr_db is not None:
        out["naq_harm"] = round(max(0.0, min(100.0, hnr_db / 30.0 * 100.0)), 1)
    if flat is not None:
        out["naq_buzz"] = round(max(0.0, min(100.0, (1.0 - flat) * 100.0)), 1)

    if out["naq_harm"] is not None and out["naq_buzz"] is not None:
        out["naq"] = round(0.65 * out["naq_harm"] + 0.35 * out["naq_buzz"], 1)
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
        if s["naq"] is None:
            failures.append("real speech NAQ was None")
        elif s["naq"] < 10:
            failures.append(f"real speech NAQ {s['naq']} < 10 (clean speech should beat noise floor)")
    else:
        print(f"SKIP real speech (no {real_wav})")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        noise_path = f.name
    try:
        noise = (np.random.default_rng(0).standard_normal(48000 * 3)).astype("float32") * 0.1
        sf.write(noise_path, noise, 48000)
        s = score(noise_path)
        print(f"white noise: {json.dumps(s)}")
        # White noise: harm path returns None (no periodicity), so composite is None.
        # If composite slips through, it should still be low.
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
        if s["naq"] is not None:
            failures.append(f"silence NAQ was {s['naq']} (expected None)")
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
