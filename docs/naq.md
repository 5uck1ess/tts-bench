# NAQ v2 — Naturalness-Artifact Quotient

A 0-100 objective score for the audio output of any TTS model. Higher = more natural; lower = more roboty / vocoder-artifacted.

The point is to make the quality story as concrete and measurable as the speed story. Listening rankings are subjective, error-prone, and not benchmarkable. NAQ is computed automatically over every cold-run wav and lands in the `results.csv` for every bench run.

---

## Formula

```
NAQ = 0.5 × ARTIFACT  +  0.5 × NATURALNESS
```

Where each macro is the mean of non-null sub-features:

```
ARTIFACT    = mean of {HARM, BUZZ}                          (>= 1 required)
NATURALNESS = mean of {DYN, PROSODY, RHYTHM, PITCH_MVMT}     (>= 2 required)
```

All sub-features and macros are normalized to [0, 100] so the composite stays in [0, 100]. Composite is `None` if either macro is null.

---

## Sub-features

| Sub-feature | Axis | What it captures | How it's computed |
|---|---|---|---|
| **HARM** | artifact | Harmonic-to-noise ratio in voiced regions; catches phase noise + GAN buzziness | Praat-style HNR (Boersma 1993) via librosa pyin + normalized autocorr at the F0 lag. Clipped to 30 dB. |
| **BUZZ** | artifact | 4-8 kHz vocoder hash | 1 − spectral flatness in 4-8 kHz band via `scipy.signal.welch`. |
| **DYN** | naturalness | Dynamic range; captures emphasis / breath / flat-mic feel | P95 − P5 of 20ms-frame RMS in dB. Clipped to 30 dB. |
| **PROSODY** | naturalness | F0 expressiveness; monotone vs alive | std-dev of voiced F0 in semitones (ref 100 Hz). Clipped to 5 semitones. |
| **RHYTHM** | naturalness | Timing variation; metronome vs natural pacing | Shannon entropy of inter-onset-interval distribution (10 bins, 0.1-2.0 sec), normalized by log2(10). |
| **PITCH_MVMT** | naturalness | Pitch contour velocity; flat vs alive | Mean abs frame-to-frame delta F0 across adjacent voiced frames, in semitones. Clipped to 1.5 semitones. |

The two-macro design forces the score to reflect both **absence of artifacts** AND **presence of positive naturalness cues**. Vocoded-but-prosodically-rich output and clean-but-monotone output both lose ground.

---

## Why no learned-MOS predictor

The original design included **UTMOS** as a sub-score (a pretrained model that predicts subjective MOS from a wav). It was dropped because portable install across the 18+ heterogeneous model venvs wasn't workable:

- UTMOS depends on `fairseq`, which has dataclass-compat breakages on Python 3.11
- UTMOS uses `torchaudio.load`, which on torch 2.9+ routes through `torchcodec` requiring FFmpeg shared DLLs not present on Windows
- Three predefined-voice venvs (Piper, KittenTTS, Supertonic) have no torch at all, so UTMOS can't run there

If a portable MOS predictor surfaces later (e.g. DNSMOS via ONNX), it slots back in as a fifth sub-feature with weight redistribution.

---

## Best-effort proxy until votes arrive

NAQ v2's 50/50 macro weighting and the four naturalness features are best-effort acoustic proxies for what human ears actually pick up. The eventual ground truth is a community voting system (see [docs/tasks.md](tasks.md)) that lets visitors rank model outputs head-to-head; once enough votes accumulate, NAQ weights will be refit to predict the rank and shipped as NAQ v3.

Until then, the 0.5 / 0.5 macro split is unfit but principled: equal weight to the two axes the algorithm is designed around. The threshold table below is calibrated against the cloning ranking in [docs/cloning.md](cloning.md).

---

## How to read the score

| NAQ | Rough interpretation |
|---|---|
| 70-100 | Production-grade naturalness; few audible artifacts; expressive prosody |
| 40-70 | Clearly TTS but usable; common for current open-source models |
| 20-40 | Audibly artifacted or unnaturally flat; distracting in most contexts |
| <20 | Broken or near-unusable |

Re-calibrate after listening to several samples in the [Demos site](https://5uck1ess.github.io/tts-bench/).

---

## Implementation

NAQ lives in `runners/_naq.py`. Each runner imports it and spreads `_naq.score(out_path)` into the success JSON, the same way `_meminfo` plumbs.

The score is only computed for the **cold run** of each cell — warm runs produce identical audio (same generation, same wav file). Warm runs emit null NAQ fields so the CSV column width stays consistent.

CSV columns:
- `naq` — composite, 0-100
- `naq_artifact` — ARTIFACT macro, 0-100
- `naq_naturalness` — NATURALNESS macro, 0-100

Sub-features (HARM/BUZZ/DYN/PROSODY/RHYTHM/PITCH_MVMT) are computed internally but **not written to CSV** — exposing them risks fake precision against signals we haven't calibrated against human ranking. They're surfaced as a single NAQ cell tooltip in the HTML quality report.

---

## Limitations

- **F0-dependent features (HARM, PROSODY, PITCH_MVMT).** Use pitch detection (librosa pyin). Very low or very high pitches may produce unstable readings. Tuned for typical human-speech range (50-500 Hz).
- **Single frequency band for BUZZ.** Only 4-8 kHz is checked. Vocoder artifacts above 8 kHz (codec stair-steps near Nyquist) slip past this check.
- **RHYTHM needs onsets.** Very short clips (<~3 detected onsets) cause RHYTHM to return null. Composite still computes if at least 2 naturalness sub-features succeed.
- **Best-effort.** If librosa or scipy aren't installed in the venv, or the wav fails to load, NAQ columns come back blank. The bench cell does not fail.
- **No subjective-quality fit.** The 50/50 macro weighting is principled but unfit. NAQ v3 will be the weight-fitted version once voting data exists.
- **Macro mean isn't variance-weighted.** A model scoring 90 on PROSODY and 10 on RHYTHM averages to 50, same as a model scoring 50 on each. Acceptable for v2; voting data lets v3 differentiate.

---

## Self-test

Run `python runners/_naq.py` from the repo root. Validates:

- Real speech (`reference/chris_hemsworth_15s.wav`) → NAQ > 10
- White noise → composite low or null (no F0)
- Silence → composite null (no signal)
- Synthetic monotone 220 Hz sine → NATURALNESS < 30 (flat F0)
