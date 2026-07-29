# F0 temporal-structure report

**Headline: MIXED: English=no, French=mixed for less-negative declination plus weaker absolute terminal movement.**

This is a descriptive corpus result, not a replication of the paper's experiment. “Weaker terminal movement” is operationalized as a smaller absolute median `terminal_slope`; declination is compared as a signed slope (higher means less negative).

**UNDERPOWERED HUMAN BASELINE:** only 14 metric-valid human segments survived. Every human-vs-synthetic statement below is underpowered.

## Human baseline and aggregate synthetic comparison

| language | set | n | declination median | terminal median | |terminal| | paper pattern? |
|---|---|---:|---:|---:|---:|---|
| English (prompts 1-4) | human | 12 | -2.514 | -6.704 | 6.704 | baseline |
| English (prompts 1-4) | synthetic | 784 | -4.478 | -15.853 | 15.853 | no |
| French (prompt 5) | human | 2 | -9.138 | -10.271 | 10.271 | baseline |
| French (prompt 5) | synthetic | 95 | -3.679 | -21.274 | 21.274 | mixed |

English prompts 1-4 and French prompt 5 are never pooled. Human language labels come only from named repository references: `juliette.wav` is French; `chris_hemsworth*`, `dafoe*`, and `jo.wav` are English. Distinct `_reference.wav` content without a named match is labeled unknown and excluded from language comparisons.

## Per-model ranking (primary result)

Terminal rank orders the smallest absolute median movement first (weakest movement). Declination rank orders the highest signed median first (least negative/most positive). `fallback n` counts clips whose keys were parsed from filenames because scores.csv did not join.

### English (prompts 1-4)

0/49 model medians meet both descriptive directions relative to the language-matched human baseline.

| terminal rank | model | n | terminal median | |terminal| | declination median | declination rank | fallback n |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | qwentts_fast | 12 | -0.083 | 0.083 | -5.967 | 43 | 4 |
| 2 | longcat_3p5b | 16 | 1.264 | 1.264 | -5.001 | 32 | 8 |
| 3 | chatterbox_turbo | 16 | -2.478 | 2.478 | -3.792 | 16 | 0 |
| 4 | miso | 11 | -3.060 | 3.060 | -4.317 | 22 | 8 |
| 5 | mars5 | 8 | -3.482 | 3.482 | -3.142 | 9 | 0 |
| 6 | openvoice | 48 | -5.817 | 5.817 | -5.128 | 35 | 32 |
| 7 | voxcpm | 16 | -7.063 | 7.063 | -2.826 | 6 | 0 |
| 8 | moss_tts_nano | 32 | -7.816 | 7.816 | -3.684 | 14 | 16 |
| 9 | outetts | 19 | -8.371 | 8.371 | -3.667 | 13 | 11 |
| 10 | miratts | 9 | -8.464 | 8.464 | -10.168 | 49 | 5 |
| 11 | echo | 17 | -9.245 | 9.245 | -2.339 | 3 | 9 |
| 12 | kokoro | 8 | -11.073 | 11.073 | -3.132 | 8 | 0 |
| 13 | melotts | 16 | -11.744 | 11.744 | -3.914 | 18 | 8 |
| 14 | styletts2 | 32 | -12.016 | 12.016 | -5.586 | 38 | 16 |
| 15 | fish_15 | 32 | -12.181 | 12.181 | -5.014 | 33 | 16 |
| 16 | wavtts | 8 | -12.895 | 12.895 | -4.225 | 20 | 4 |
| 17 | f5tts | 32 | -12.949 | 12.949 | -4.611 | 26 | 16 |
| 18 | moss_tts_v1.0 | 8 | -12.978 | 12.978 | -5.731 | 40 | 8 |
| 19 | zonos | 33 | -13.337 | 13.337 | -2.390 | 4 | 20 |
| 20 | pocket | 8 | -14.370 | 14.370 | -4.863 | 31 | 0 |
| 21 | neutts_air | 19 | -14.651 | 14.651 | -0.769 | 2 | 6 |
| 22 | sesame | 13 | -15.598 | 15.598 | -5.606 | 39 | 0 |
| 23 | neutts_nano | 19 | -15.605 | 15.605 | -3.832 | 17 | 8 |
| 24 | soprano | 8 | -15.682 | 15.682 | -6.419 | 44 | 0 |
| 25 | moss_tts | 39 | -16.165 | 16.165 | -5.823 | 41 | 32 |
| 26 | scyllasband | 8 | -16.779 | 16.779 | -4.319 | 23 | 4 |
| 27 | magpie | 16 | -16.832 | 16.832 | -3.098 | 7 | 8 |
| 28 | vibevoice | 8 | -17.190 | 17.190 | -4.481 | 24 | 0 |
| 29 | piper | 4 | -18.348 | 18.348 | -4.632 | 27 | 0 |
| 30 | dia | 11 | -18.970 | 18.970 | 3.376 | 1 | 5 |
| 31 | coqui | 15 | -19.369 | 19.369 | -3.692 | 15 | 0 |
| 32 | longcat_1b | 16 | -19.568 | 19.568 | -5.179 | 36 | 8 |
| 33 | chatterbox | 16 | -21.817 | 21.817 | -4.284 | 21 | 0 |
| 34 | vibevoice_15b | 8 | -21.829 | 21.829 | -4.015 | 19 | 0 |
| 35 | parler | 15 | -22.663 | 22.663 | -5.057 | 34 | 8 |
| 36 | moss_tts_v1.5 | 8 | -22.917 | 22.917 | -4.787 | 29 | 8 |
| 36 | moss_tts_v15 | 8 | -22.917 | 22.917 | -4.787 | 29 | 0 |
| 38 | maya1 | 14 | -23.202 | 23.202 | -3.616 | 12 | 7 |
| 39 | indextts | 32 | -24.393 | 24.393 | -5.861 | 42 | 16 |
| 40 | lfm2_audio | 18 | -24.503 | 24.503 | -7.762 | 47 | 10 |
| 41 | omnivoice | 16 | -25.943 | 25.943 | -3.484 | 11 | 0 |
| 42 | zipvoice | 12 | -26.314 | 26.314 | -2.788 | 5 | 0 |
| 43 | dramabox | 16 | -28.224 | 28.224 | -4.579 | 25 | 8 |
| 44 | inflect_nano | 16 | -28.929 | 28.929 | -4.724 | 28 | 8 |
| 45 | vibevoice_7b | 16 | -32.258 | 32.258 | -6.722 | 45 | 8 |
| 46 | qwentts | 8 | -32.422 | 32.422 | -8.228 | 48 | 0 |
| 47 | inflect_micro | 16 | -36.006 | 36.006 | -3.194 | 10 | 8 |
| 48 | supertonic | 4 | -40.959 | 40.959 | -6.739 | 46 | 0 |
| 49 | kittentts | 4 | -65.943 | 65.943 | -5.310 | 37 | 0 |

### French (prompt 5)

4/22 model medians meet both descriptive directions relative to the language-matched human baseline.

| terminal rank | model | n | terminal median | |terminal| | declination median | declination rank | fallback n |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | zonos | 9 | -1.299 | 1.299 | -3.130 | 8 | 5 |
| 2 | zipvoice | 4 | -1.554 | 1.554 | -3.798 | 11 | 0 |
| 3 | qwentts_fast | 3 | -4.427 | 4.427 | -13.751 | 22 | 1 |
| 4 | neutts_nano | 5 | 5.604 | 5.604 | 0.154 | 1 | 2 |
| 5 | pocket | 2 | 7.273 | 7.273 | -6.339 | 17 | 0 |
| 6 | omnivoice | 4 | -10.641 | 10.641 | -3.565 | 9 | 0 |
| 7 | moss_tts_nano | 7 | -11.317 | 11.317 | -9.336 | 21 | 4 |
| 8 | moss_tts | 10 | -11.796 | 11.796 | -4.385 | 14 | 8 |
| 9 | fish_15 | 8 | -15.779 | 15.779 | -6.998 | 18 | 4 |
| 10 | piper | 1 | -17.323 | 17.323 | -3.586 | 10 | 0 |
| 11 | outetts | 4 | -19.669 | 19.669 | -6.079 | 16 | 2 |
| 12 | moss_tts_v1.5 | 2 | -19.721 | 19.721 | -7.265 | 19 | 2 |
| 12 | moss_tts_v15 | 2 | -19.721 | 19.721 | -7.265 | 19 | 0 |
| 14 | melotts | 4 | -23.180 | 23.180 | -2.537 | 5 | 2 |
| 15 | openvoice | 12 | -24.933 | 24.933 | -2.983 | 7 | 8 |
| 16 | moss_tts_v1.0 | 2 | -29.135 | 29.135 | -2.786 | 6 | 2 |
| 17 | voxcpm | 4 | -30.088 | 30.088 | -4.254 | 13 | 0 |
| 18 | kokoro | 2 | -32.720 | 32.720 | -3.871 | 12 | 0 |
| 19 | qwentts | 2 | -33.863 | 33.863 | -2.409 | 4 | 0 |
| 20 | coqui | 3 | -35.436 | 35.436 | -1.091 | 2 | 0 |
| 21 | magpie | 4 | -36.050 | 36.050 | -2.143 | 3 | 2 |
| 22 | supertonic | 1 | -59.805 | 59.805 | -5.910 | 15 | 0 |

## Dispersion control

Overlap is defined before inspection as overlap between human and synthetic interquartile intervals. Metrics are computed from cleaned voiced frames before time interpolation.

| language | metric | human n | human median [IQR] | synthetic n | synthetic median [IQR] | IQR overlap? |
|---|---|---:|---:|---:|---:|---|
| English (prompts 1-4) | f0_range_st | 12 | 10.681 [6.340, 13.811] | 784 | 9.720 [7.607, 12.294] | yes |
| English (prompts 1-4) | f0_sd_st | 12 | 3.787 [1.993, 4.056] | 784 | 3.149 [2.464, 3.864] | yes |
| French (prompt 5) | f0_range_st | 2 | 10.018 [9.430, 10.606] | 95 | 9.486 [8.116, 10.813] | yes |
| French (prompt 5) | f0_sd_st | 2 | 3.217 [3.165, 3.268] | 95 | 3.061 [2.492, 3.266] | yes |

Dispersion-control result: English=reproduced; French=reproduced. This criterion requires IQR overlap for both `f0_range_st` and `f0_sd_st`.

## Duration control

Spearman correlations are utterance-level and descriptive. They do not account for repeated prompts, voices, models, or multiple segments from one human source.

| language | set | n | declination vs duration rho | p | terminal vs duration rho | p |
|---|---|---:|---:|---:|---:|---:|
| English (prompts 1-4) | human | 12 | 0.070 | 0.829 | -0.007 | 0.983 |
| English (prompts 1-4) | synthetic | 784 | 0.341 | <0.001 | -0.067 | 0.059 |
| French (prompt 5) | human | 2 | NA | NA | NA | NA |
| French (prompt 5) | synthetic | 95 | 0.124 | 0.230 | 0.085 | 0.415 |

## Counts and exclusions

- Synthetic WAVs discovered: 998.
- Joined to `C:/Users/tymra/LocalDev/tts-bench-codex-f0prosody/scoring/scores.csv` on `(dir, wav)`: 540; filename fallback: 458.
- Score rows with no matching WAV under `C:/Users/tymra/LocalDev/tts-bench/results`: 578 of 1118.
- Failed generations removed before F0 analysis: 36 unique WAVs; health flags=gap=10 (total 10), WER > 0.5=27, both=1.
- Synthetic retained/analyzed: 962; metric-valid=959; too few voiced frames=3; analysis errors=0.
- Human source WAV candidates: 23; unique decoded-audio hashes=6; duplicate copies=17; source load errors=0.
- Human segments >= 1.2s analyzed: 14; metric-valid=14; too few voiced frames=0; analysis errors=0.
- Rows with unknown language (excluded from EN/FR comparisons): synthetic=81, human=0.

## Method

Praat autocorrelation F0 uses a broad 50-800 Hz pass to estimate each clip/source voice, followed by an adaptive median/2.5 to median*2.5 floor/ceiling. Unvoiced frames are dropped. Contiguous voiced runs receive a five-frame median filter; remaining adjacent jumps over 12 semitones reject the member farther from the run median. Each valid contour is converted to semitones relative to its voiced-frame median and linearly interpolated to 100 points from the first through last cleaned voiced frame (internal unvoiced gaps retain their timing; container-edge padding is not extrapolated). OLS slopes use normalized time 0-1 globally and 0.80-1.00 terminally. Dispersion uses the cleaned, uninterpolated frames.

Human sources are deduplicated by SHA-256 over decoded float32 audio content and segmented with 25 ms energy frames/10 ms hops, an adaptive max(-45 dBFS, p90-30 dB) threshold, and 0.35s minimum silence. Segments shorter than 1.2s are ignored.

## Limitations

- The human baseline is a handful of source recordings. Silence-derived segments from the same clip are correlated and do not create independent speakers.
- Human segments are not text- or duration-matched to the five synthetic prompts. The duration correlations diagnose, but do not remove, this confound.
- The `(dir, wav)` join covered only 540/998 discovered synthetic WAVs. Filename-fallback rows have no health/WER gate, and mode is unknown when it is not encoded in the path.
- The recursive synthetic set includes historical runs, smoke/comparison artifacts, and potentially repeated audio. It was not deduplicated because the requested set was every non-reference WAV.
- Per-model medians with small n are unstable. Rows are not statistically independent across devices, modes, prompts, repeated runs, or shared cloning references.
- Terminal slope is especially sensitive to utterance duration, endpoint voicing, and silence segmentation; normalized time does not eliminate those effects.
- Automatic pitch bounds and median filtering reduce octave errors but cannot guarantee that every tracker error is removed, especially for creaky voice, music, noise, or non-speech artifacts.
- The result is descriptive medians without uncertainty intervals, mixed-effects modeling, or a prompt-matched inferential test. It cannot establish that model architecture causes over-smoothing.

---

## Verification note (Claude, 2026-07-29) — the human baseline is not just underpowered, the segmentation is WRONG

Added after running this report's own output back through inspection. **Do not cite the
human-vs-synthetic direction from this run.**

The human set is 14 segments, and inspecting them individually shows the silence-based
segmentation did not split sentences:

| clip | segment | duration | terminal_slope |
|---|---|---:|---:|
| chris_hemsworth.wav | 1 | 18.24 s | -4.95 |
| chris_hemsworth.wav | 2 | 17.15 s | +1.84 |
| chris_hemsworth_15s.wav | 1 | 14.85 s | -27.04 |
| jo.wav | 1 | 13.00 s | -33.55 |
| dafoe_full.wav | 1 | 1.98 s | -41.57 |
| dafoe_full.wav | 2 | 1.74 s | +33.08 |

Three problems, each independently disqualifying:

1. **Multi-sentence blocks.** An 18-second segment spans many sentences. `terminal_slope` over the
   last 20% of such a block averages across sentence boundaries and is flattened by construction —
   it is not the same measurement the synthetic side gets on a single ~3-6 s sentence. This alone
   biases the human baseline toward weaker terminal movement, which is exactly the direction the
   headline reports.
2. **Terminal slopes range from -41.57 to +33.08**, with adjacent segments of the *same clip*
   taking opposite signs. That is noise, not a baseline.
3. **7 of 14 segments come from one speaker** (`chris_hemsworth`), so the median is one voice.

**What survives:** the **dispersion control** (English and French both reproduced — human and
synthetic `f0_range_st` / `f0_sd_st` IQRs overlap). That is a distributional overlap test rather
than a point comparison, so a thin baseline hurts it far less. It supports the paper's central
negative claim: *pitch span is not what separates human from synthetic speech.*

**What does NOT survive:** any directional claim about human vs synthetic declination or terminal
slope, in either language.

**What is real but uninterpretable:** the per-model ranking. The spread is large and reproducible
(`qwentts_fast` -0.08 to `omnivoice` -25.9; note `chatterbox` -21.8 vs `chatterbox_turbo` -2.48
within one family), but with no trustworthy human anchor we cannot say which end of the ranking is
*more human-like* — only that models differ enormously on this axis.

**Fix before this metric can rank models by human-likeness:** record a real human reading the 5
bench prompts, one utterance per file, matching the synthetic side's design exactly. That replaces
segmentation guesswork with the paper's actual paired design.
