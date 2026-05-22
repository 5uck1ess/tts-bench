---
title: tts-bench — Pocket-TTS vs NeuTTS Air on Windows
created: 2026-05-22
related:
  - homebase/wiki/research/cicero/pocket-tts-vs-neutts-bench-may2026.md
---

# tts-bench — Pocket-TTS vs NeuTTS Air on Windows

## Purpose

Implement the bench described in `homebase/wiki/research/cicero/pocket-tts-vs-neutts-bench-may2026.md`, adapted to run on this Windows box instead of the ryzen-ai (3090) box the homebase spec assumes. Goal: produce the data needed to decide whether Pocket-TTS replaces NeuTTS Air as Cicero's Sovereign-tier default TTS.

The homebase doc is the **source of truth for what to measure and how to decide**. This spec is the source of truth for **how the harness is built and run on Windows**.

## Deviations from the homebase spec

| Aspect | Homebase spec | This bench |
|---|---|---|
| Machine | ryzen-ai (Linux, 3090) | This Windows box |
| Reference voice | `~/.hermes/.../jo.wav` | Two-pass: (1) each model's default voice, (2) user-supplied `reference/reference.wav` |
| Scope | CPU-only both models | CPU-only both models **plus** NeuTTS-on-GPU as a third row |
| Install commands | bash + uv | PowerShell + uv |

Pass 1 (default voice) is naturalness/speed comparison only — voice-similarity scoring is N/A because the two models clone different built-in voices. Pass 2 (user voice) is the full apples-to-apples clone-fidelity comparison.

The decision rubric in the homebase doc (RTF, TTFA-100ms, quality-within-0.5) still applies, evaluated against Pass 2 numbers.

## Architecture

Two-venv harness with an orchestrator. Each TTS model gets its own `uv` venv inside the repo so torch/audio deps don't fight. A top-level `bench.py` orchestrates by spawning per-model runner scripts in their own venvs, feeding them identical prompts via JSON over stdin, and capturing identical metrics.

### Directory layout

```
tts-bench/
  bench.py                  # orchestrator
  prompts.py                # the 5 prompts from the homebase spec (verbatim)
  runners/
    pocket_tts_runner.py    # runs inside pocket venv
    neutts_runner.py        # runs inside neutts venv; takes --device cpu|cuda
  reference/
    .gitkeep                # user drops reference.wav + reference.txt here for Pass 2
  venvs/
    pocket/                 # uv venv: pocket-tts
    neutts/                 # uv venv: neutts-air + torch
  results/
    <ISO-timestamp>/        # one dir per run (e.g., 2026-05-22T1430)
      pocket_cpu/prompt-{1..5}.wav
      neutts_cpu/prompt-{1..4}.wav
      neutts_gpu/prompt-{1..4}.wav    # only if --include-gpu
      results.csv
      quality.csv
      run.json              # run config: reference path, device list, prompt set hash
      README.md
  scripts/
    install.ps1             # creates both venvs, installs both models
  README.md
```

`venvs/`, `results/`, and `reference/*.wav` are gitignored.

### Runner protocol

Each runner is a tiny CLI invoked as:

```
<venv-python> runners/<model>_runner.py --job <path-to-job.json> --out <path-to-output.wav>
```

`job.json`:

```json
{
  "prompt": "Open the browser and read my email.",
  "reference_wav": null,                  // null = use model default voice
  "reference_txt": null,
  "device": "cpu",                        // ignored by pocket runner
  "warmup": false                         // if true, do a tiny warmup gen before timing
}
```

The runner writes the WAV to `--out` and writes a single JSON line to stdout:

```json
{"ttfa_ms": 412, "wall_s": 1.83, "audio_s": 2.1, "peak_rss_mb": 640, "ok": true}
```

TTFA is captured **inside the runner** (timer started right before `generate()`, stopped on first audio sample received). The orchestrator measures wall time externally as a sanity check.

If the runner fails it writes `{"ok": false, "error": "..."}` and exits non-zero. The orchestrator records the row as failed and continues.

### Orchestrator (`bench.py`)

```
python bench.py --reference default --include-gpu
python bench.py --reference reference/reference.wav --include-gpu
```

Flags:
- `--reference {default | <path-to-wav>}` — required
- `--include-gpu` — adds the `neutts_gpu` row; omitted by default so a fresh checkout works without CUDA
- `--prompts {all | 1,2,3}` — defaults to all; useful for iteration
- `--cold-warm-prompt 2 --cold-warm-runs 3` — runs the given prompt N times back-to-back to capture cold-vs-warm TTFA (defaults match the homebase spec)
- `--out-dir <path>` — overrides `results/<timestamp>`

The run has two phases:

**Phase 1 — main sweep.** For each (model_config, prompt) pair the orchestrator:

1. Builds a `job.json` in a tempfile.
2. Spawns the runner as a subprocess.
3. Records subprocess start/end wall time and peak RSS via `psutil.Process(pid).memory_info()` polled every 50ms on a background thread.
4. Parses the runner's stdout JSON for TTFA and audio duration.
5. Appends a row to `results.csv` with `cold_or_warm=cold`.

**Phase 2 — cold/warm sweep.** For each model_config, the orchestrator re-runs the `--cold-warm-prompt` (default: prompt 2) `--cold-warm-runs - 1` additional times back-to-back (default: 2 more, so prompt 2 ends up with 3 total rows per model_config across both phases). These additional rows are written with `cold_or_warm=warm`. This isolates first-call JIT/model-load cost from steady-state latency.

Per-core CPU usage is captured by sampling `psutil.cpu_percent(percpu=True)` on the same 50ms poller. For each row we record the peak count of cores simultaneously above 50% during the run, plus the per-core peak utilizations as a compact string (e.g. `[98,97,12,8,...]`) — enough for the user to verify Pocket-TTS's "2 cores" claim by eye.

### `results.csv` columns

```
prompt_id, model_config, ttfa_ms, wall_s, audio_s, rtf, peak_rss_mb, cores_active_peak, per_core_peaks, cold_or_warm, reference_kind, ok, error
```

`rtf = audio_s / wall_s`. `reference_kind` is `default` or `user`. `cold_or_warm` is `cold` for the first occurrence per (model_config, prompt) in the run and `warm` for subsequent.

### `quality.csv` (blank, user fills in after listening)

```
prompt_id, model_config, naturalness_1_5, similarity_1_5, intelligibility_1_5, artifacts_1_5, notes
```

The run's `README.md` explains the rubric inline so the user doesn't have to flip back to the homebase doc.

## Install (`scripts/install.ps1`)

Creates both venvs via `uv venv`, installs each model, pre-bakes any voice embeddings the models need for their default voice. Idempotent — re-running with existing venvs is a no-op except for upgrades.

Open install questions (resolved during implementation, not in this spec):
- Pocket-TTS Windows install path — the homebase spec uses `git clone + uv pip install -e .`; whether the upstream repo builds clean on Windows will be confirmed in the install step. If it doesn't, the install script surfaces the failure clearly (it does not silently fall back).
- NeuTTS Air install — package name and torch wheel selection (CPU vs CUDA) will be confirmed at install time.

If either install fails, the user sees a clear error and the bench can still run with the other model.

## Error handling and edge cases

- Missing reference files in Pass 2 → orchestrator errors before spawning runners.
- Runner crash on one prompt → row marked failed, bench continues with remaining prompts.
- No CUDA available with `--include-gpu` → orchestrator warns and skips `neutts_gpu` rows (does not fail the whole run).
- Prompt 5 (French) → only attempted for `pocket_cpu`; `neutts_*` rows skip it.

## Testing

This is a benchmark harness, not a library — the tests that matter are:

1. **Smoke test**: `python bench.py --reference default --prompts 1 --cold-warm-runs 1` completes and writes a non-empty WAV + a valid `results.csv` row for at least `pocket_cpu`.
2. **Manual ear test**: open the WAVs in the run dir, listen, fill in `quality.csv`. This is the actual deliverable.

No automated audio-quality tests — the homebase spec is explicit that quality scoring is subjective.

## Out of scope

- Replicating the exact ryzen-ai numbers. This bench produces Windows numbers; if the swap decision needs ryzen-ai validation we run the same harness there later.
- Patching Hermes to swap the TTS provider. The homebase doc says that's a separate spec filed only if the decision is YES.
- Real-time streaming integration. We measure TTFA but don't pipe audio to speakers.

## Deliverable

A populated `results/<timestamp>/` directory with `results.csv`, filled-in `quality.csv`, and WAVs, plus a verdict appended to the homebase doc's "After the bench" section.
