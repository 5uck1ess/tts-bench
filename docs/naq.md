# NAQ — Naturalness-Artifact Quotient

An objective 0-100 score for the audio output of any TTS model. Higher = more natural; lower = more roboty / vocoder-artifacted.

The point is to make the quality story as concrete and measurable as the speed story. Listening rankings are subjective, error-prone, and not benchmarkable. NAQ is computed automatically over every cold-run wav and lands in the `results.csv` for every bench run.

---

## What it measures

NAQ combines two factor groups, each surfaced as its own 0-100 macro in the quality report:

- **ARTIFACT** — acoustic cues for the *absence* of vocoder artifacts (noise, buzz, phase issues, codec hash).
- **NATURALNESS** — acoustic cues for the *presence* of expressive speech (dynamics, prosodic variation, rhythm, pitch movement).

The two-macro design forces the composite to reflect both **absence of artifacts** AND **presence of positive naturalness cues**. Vocoded-but-prosodically-rich output and clean-but-monotone output both lose ground.

The exact features inside each macro, their normalizations, and the weights that combine them are intentionally kept opaque in this repo. They will change as community voting data accumulates — see [docs/tasks.md](tasks.md) for the planned voting harness. Treat the published weights as a moving target.

---

## How to read the score

| NAQ | Rough interpretation |
|---|---|
| 70-100 | Production-grade naturalness; few audible artifacts; expressive prosody |
| 40-70 | Clearly TTS but usable; common for current open-source models |
| 20-40 | Audibly artifacted or unnaturally flat; distracting in most contexts |
| <20 | Broken or near-unusable |

Re-calibrate after listening to several samples in the [Demos site](https://5uck1ess.github.io/tts-bench/). The ARTIFACT and NATURALNESS macros are also shown per-cell — a model with high ARTIFACT but low NATURALNESS sounds clean but monotone; the opposite combination sounds alive but vocoded.

---

## CSV columns

The score lands in `results.csv` as three columns:

- `naq` — composite, 0-100
- `naq_artifact` — ARTIFACT macro, 0-100
- `naq_naturalness` — NATURALNESS macro, 0-100

Warm-run rows emit null NAQ fields (warm runs reuse the cold-run wav, so the score would be identical).

---

## Best-effort proxy until votes arrive

NAQ is a best-effort acoustic proxy for what human ears actually pick up. The eventual ground truth is a community voting system (see [docs/tasks.md](tasks.md)) that lets visitors rank model outputs head-to-head; once enough votes accumulate, the weighting is refit to predict the rank.

Until then, the current weighting is principled but unfit. The threshold table above is calibrated against the cloning ranking in [docs/cloning.md](cloning.md).

---

## Limitations

- **Pitch-tracking failure modes.** Several naturalness signals depend on F0 estimation. Very low or very high pitches, or noisy outputs, may produce unstable readings.
- **Short-clip failure modes.** Rhythm signals need several detected onsets. Very short clips fall back gracefully but yield lower naturalness confidence.
- **Best-effort.** If a runner's venv is missing the analysis deps, or the wav fails to load, NAQ columns come back blank. The bench cell does not fail.
- **No subjective-quality fit yet.** The current weighting is principled but unfit; subjective-vote-fit refits will replace it once data exists.
- **Macro mean isn't variance-weighted.** A model scoring high on one naturalness signal and low on another averages out the same as a model scoring middling on each.
