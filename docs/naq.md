# NAQ — Naturalness-Artifact Quotient

A 0-100 objective score for the audio output of any TTS model. Higher = more natural; lower = more roboty / vocoder-artifacted.

The point is to make the quality story as concrete and measurable as the speed story. Listening rankings are subjective, error-prone, and not benchmarkable. NAQ is computed automatically over every cold-run wav and lands in the `results.csv` for every bench run.

---

## Formula

```
NAQ = 0.65 × HARM  +  0.35 × BUZZ
```

Both sub-scores are normalized to 0-100 so the composite stays in [0, 100].

| Sub-score | What it captures | How it's computed |
|---|---|---|
| **HARM** | "Real voiced regions vs noisy ones" — phase-noise / GAN buzziness | Harmonic-to-noise ratio (HNR) in voiced regions via librosa pyin + Praat-style normalized autocorrelation at the F0 lag (Boersma 1993). Normalized to a 30 dB cap. |
| **BUZZ** | High-frequency vocoder hash | 1 − spectral flatness in 4-8 kHz (the vocoder buzz band), via `scipy.signal.welch`. Flat in 4-8 kHz = vocoder-style hash; peaked = natural consonant noise. |

The weighting (0.65 / 0.35) puts more trust in the harmonic-structure axis (HARM) because that's what catches both phase noise and GAN-vocoder buzziness — the two most common artifact families in modern TTS. BUZZ adds the specific check for vocoder hash in the 4-8 kHz band that the human ear catches as "roboty static".

---

## Why these two sub-scores

The two sub-scores catch different artifact families:

- **HARM alone** would score white noise low (good) but would also score very compressed/buzzy speech high if the model preserves harmonic structure with phase noise on top. HARM doesn't see *what* the noise is, only that voiced regions have it.
- **BUZZ alone** is narrow — it only looks at one frequency band. Other artifacts (over-smoothing, prosody glitches) fly past it.

Combining the two with weights gives a reading that's more robust to either's blind spot. The per-axis sub-scores are surfaced as a tooltip in the HTML report so you can diagnose *why* a model scored where it did.

---

## Why no learned-MOS predictor

The original design included **UTMOS** as a third sub-score (a pretrained model that predicts subjective MOS from a wav). It was dropped because portable install across the 18+ heterogeneous model venvs wasn't workable:

- UTMOS depends on `fairseq`, which has dataclass-compat breakages on Python 3.11
- UTMOS uses `torchaudio.load`, which on torch 2.9+ routes through `torchcodec` requiring FFmpeg shared DLLs not present on Windows
- Three predefined-voice venvs (Piper, KittenTTS, Supertonic) have no torch at all, so UTMOS can't run there

Five venv-local patches got UTMOS working in one venv, but codifying them across all 18 was brittle and didn't scale. If a portable MOS predictor surfaces later (e.g. DNSMOS via ONNX), it slots back in as a third sub-score with a weight redistribution.

---

## How to read the score

| NAQ | Rough interpretation |
|---|---|
| 70-100 | Production-grade naturalness; few audible artifacts |
| 40-70 | Clear "this is TTS" feel but usable; common for current open-source models |
| 20-40 | Audibly artifacted; distracting in most contexts |
| <20 | Output is broken or near-unusable |

These thresholds are rough — calibrated against the listening notes in [docs/cloning.md](cloning.md). Re-calibrate after listening to several samples in the [Demos site](https://5uck1ess.github.io/tts-bench/).

Reference points from the helper's self-test on a noisy-but-natural recording (Chris Hemsworth, ~15s clip with audible background noise):
- Real (noisy) speech: NAQ ≈ 12
- White noise: NAQ < 5
- Silence: returns null (no harmonic structure, no spectral content)

For clean TTS output without background noise, NAQ in the 30-70 range is typical.

---

## Implementation

NAQ lives in `runners/_naq.py` alongside the memory-sampling helper. Each runner imports it and spreads `_naq.score(out_path)` into the success JSON, the same way `_meminfo` plumbs.

The score is only computed for the **cold run** of each cell — warm runs produce identical audio (same generation, same wav file), so no point re-scoring. Warm runs emit null NAQ fields so the CSV column width stays consistent.

Sub-scores appear as three CSV columns:
- `naq` — composite, 0-100
- `naq_harm` — HARM sub-score, 0-100
- `naq_buzz` — BUZZ sub-score, 0-100

And as a single `NAQ` column in each per-prompt report table, with the two sub-scores in a `title=""` tooltip.

---

## Limitations

- **F0-dependent HARM.** Uses pitch detection (librosa pyin). Very low or very high pitches may produce unstable HNR readings. Tuned for typical human-speech range (50-500 Hz).
- **Single frequency band for BUZZ.** Only 4-8 kHz is checked. Vocoder artifacts above 8 kHz (codec stair-steps near Nyquist) slip past this check.
- **Best-effort.** If librosa or scipy aren't installed in the venv, or the wav fails to load, NAQ columns come back blank. The bench cell does not fail.
- **No prosody axis.** Unnatural pacing, robotic intonation, or unnatural pauses (the "Sesame fake pauses" issue) aren't directly captured. HARM and BUZZ measure acoustic-spectrum properties, not temporal patterns.
- **No subjective-quality axis.** With UTMOS dropped, NAQ captures specific artifact types rather than overall perceived quality. Pair with subjective listening (see [docs/cloning.md](cloning.md)) for the full picture.

---

## Calibration data points

To be filled in after the first post-NAQ cloning bench pass. Will include a small table mapping NAQ buckets to listening-rank position from [docs/cloning.md](cloning.md), so visitors can sanity-check whether NAQ agrees with human ears on this hardware/reference.
