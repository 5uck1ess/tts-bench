Goal: Add an F0 temporal-structure metric to tts-bench and use it to test whether modern TTS shows the pitch over-smoothing that Santos & San Segundo (Cadernos de Linguistica) found in CycleGAN/diffusion voice conversion.

Background (the claim under test):
  Human and synthetic speech have BROADLY OVERLAPPING global F0 range and standard deviation --
  pitch span is NOT the difference. What separates them is the TEMPORAL organization of F0:
  synthetic speech shows (a) a less-negative global declination slope across the utterance, and
  (b) weaker terminal movement in the final fifth. Their synthetic side was GAN + diffusion voice
  conversion, NOT modern autoregressive TTS. We are testing whether it transfers to our models.

Success criteria:
  - New module `scoring/prosody.py` computing, per wav, using praat-parselmouth:
      * F0 extracted with Praat autocorrelation; unvoiced frames dropped
      * reference F0 = median of voiced frames; convert F0 to SEMITONES relative to that median
      * time-normalize each utterance to a fixed number of equidistant points (use 100)
      * `declination_slope`  = OLS slope of semitone contour over full normalized time
      * `terminal_slope`     = OLS slope over the 0.80-1.00 normalized-time window
      * `f0_range_st`        = 5th-to-95th percentile spread in semitones (outlier-robust)
      * `f0_sd_st`           = standard deviation in semitones
      * `duration_s`, `voiced_frac`, `n_voiced`
    Guard octave-jump artifacts (Praat pitch floor/ceiling per voice; median-filter or reject
    frames jumping >1 octave between adjacent voiced frames). Return NaN, never a bogus number,
    when an utterance has too few voiced frames (<30) -- and count how many you dropped.
  - New script `scripts/prosody_report.py` that:
      * SYNTHETIC set = all .wav under C:/Users/tymra/LocalDev/tts-bench/results/**  (READ ONLY)
        excluding any file named `_reference.wav`
      * HUMAN set = C:/Users/tymra/LocalDev/tts-bench/reference/*.wav plus the distinct
        `results/*/_reference.wav` files (dedupe by audio content hash -- many are copies)
        Human clips are long voice-cloning samples, so SEGMENT them into utterances on silence
        (Praat/energy-based) and treat each segment >=1.2s as one utterance.
      * joins synthetic wavs to `scoring/scores.csv` on (dir, wav) to recover model/mode/prompt_id,
        and DROPS rows whose scores.csv row indicates a failed/garbage generation (use the health
        column and an obviously-broken WER, e.g. wer > 0.5) -- report how many were dropped and why
      * analyzes prompt_id 5 (French) SEPARATELY from the English prompts 1-4, never pooled
      * writes `scoring/prosody.csv` (one row per utterance, all metrics + join keys)
      * writes `docs/prosody-report.md` containing:
          - per-model medians for declination_slope and terminal_slope, ranked
          - the human baseline for the same measures
          - a DISPERSION CONTROL section showing f0_range_st / f0_sd_st for human vs synthetic
            (the paper's point is these should OVERLAP -- report whether they do here)
          - duration control: report Spearman correlation of each slope measure with duration_s,
            since terminal_slope is known to be duration-sensitive
  - Run it end to end and paste the real summary numbers into your final report.
  - Add a `tests/test_prosody.py` with synthetic-signal unit tests: a synthesized falling-pitch
    tone must yield a negative declination_slope, a flat tone ~0, a rising tone positive.
    Run the tests and make sure they pass.

Constraints:
  - Python interpreter: ./.venv-prosody/Scripts/python.exe  (praat-parselmouth, numpy, scipy,
    pandas already installed -- network is BLOCKED, do not attempt to install anything)
  - Only create/modify: scoring/prosody.py, scripts/prosody_report.py, tests/test_prosody.py,
    docs/prosody-report.md, scoring/prosody.csv
  - The wav corpus outside this worktree is READ-ONLY. Never write, move, rename or delete
    anything under C:/Users/tymra/LocalDev/tts-bench/. Read it in place.
  - Do not modify bench.py, harness.py, scoring/score_all.py or any existing scoring module.
  - No dependency changes, no unrelated refactors.

Output (report back):
  1. The headline answer in one line: DOES modern TTS in this corpus show the paper's pattern
     (less-negative declination slope + weaker terminal slope than human), yes / no / mixed?
  2. The per-model ranking table for terminal_slope and declination_slope.
  3. Whether the dispersion control reproduced (do human and synthetic f0_range/f0_sd overlap here?).
  4. Counts: utterances analyzed per set, dropped for too-few-voiced-frames, dropped as failed gens.
  5. Every place the result is weak or the design is limited -- state it plainly.

Stop rules:
  - The HUMAN baseline is thin (a handful of source clips). Do NOT paper over this. If after
    segmentation you have fewer than ~30 human utterances, say so explicitly and label every
    human-vs-synthetic claim as underpowered. Report the per-model comparison (which is
    well-powered) as the primary result regardless.
  - If scores.csv cannot be joined to the wav paths, do not guess the mapping -- report the exact
    mismatch and proceed with model names parsed from filenames, clearly flagged as a fallback.
  - If a required fact is missing, name it and stop. Do not invent a number.
