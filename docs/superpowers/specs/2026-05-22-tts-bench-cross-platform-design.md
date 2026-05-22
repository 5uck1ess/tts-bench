---
title: tts-bench — Cross-Platform Speed Bench for Cicero Sovereign TTS
created: 2026-05-22
updated: 2026-05-22
related:
  - homebase/wiki/research/cicero/pocket-tts-vs-neutts-bench-may2026.md
  - homebase/wiki/research/deepfake/voice/new-models-apr2026.md
---

# tts-bench — Cross-Platform Speed Bench for Cicero Sovereign TTS

## Purpose

Pick the fastest viable TTS for Cicero's Sovereign tier across **all three deployment surfaces** (CPU edge box, CUDA workstation, Apple Silicon laptop). Cicero is interactive — speed (TTFA + RTF) is the primary criterion. Quality has to clear a floor, not win.

*Sovereign tier*, per the homebase Cicero positioning doc: 100% local, no cloud API, runs on the user's own hardware. Voice clone is mandatory; license must be permissive; working footprint ≤1GB so it fits alongside an LLM and STT on a single appliance.

Extends the homebase doc `pocket-tts-vs-neutts-bench-may2026.md` from a 2-model CPU-parity bench on the ryzen-ai box to a 4-model cross-platform bench that you can run on Windows, then on Mac, and join the results.

## Candidate set

Four open-weight TTS models that satisfy all four Cicero Sovereign constraints (voice cloning + cross-platform + permissive license + ≤1GB working footprint):

| Model | Params | License | CPU | CUDA | Metal/MPS | Notes |
|---|---|---|---|---|---|---|
| **Pocket-TTS** (Kyutai) | 100M | MIT | ✅ 6× RT M4 | ⚠️ no speedup | ✅ | Multilingual (EN/FR/DE/IT/PT/ES) |
| **NeuTTS Air** (Neuphonic) | 748M | Apache-2.0 | ✅ 2× RT Ryzen 9 | ✅ 320× RT 4090 | ✅ | GGUF Q4/Q8 via llama.cpp |
| **NeuTTS Nano** (Neuphonic) | smaller | Apache-2.0 | ✅ | ✅ | ✅ | Multilingual (EN/ES/FR/DE) |
| **LuxTTS** (k2-fsa) | small | Apache-2.0 | ✅ "faster than RT" | ✅ 150× RT GPU | ✅ | 1GB VRAM, ZipVoice-distilled |

Explicitly excluded (and why, so future-us doesn't re-litigate):

- **Kokoro** — no voice cloning; can't be a Cicero Sovereign pick
- **GPT-SoVITS** — install footprint kills the appliance angle
- **Qwen3-TTS 0.6B** — community says CPU is too slow for real-time
- **Magpie 357M** — NVIDIA Open Model License (restrictive)
- **VoxCPM2, Fish S2 Pro, Qwen3-TTS 1.7B+** — too large for ≤1GB Sovereign footprint
- **Cartesia Sonic 3, Realtime TTS, Hume Octave 2** — closed API, can't be Sovereign

## Device matrix

| Platform | Devices to bench |
|---|---|
| Windows (this box) | `cpu`, `cuda` (if available) |
| Mac (later) | `cpu`, `mps` |
| Linux ryzen-ai (optional, later) | `cpu`, `cuda` |

Each runner declares the devices it supports. The orchestrator skips unsupported combinations with a clear "skipped: pocket-tts has no CUDA path" log line — it doesn't fail.

## Architecture

Per-model `uv` venv + an orchestrator. Each model runs in its own venv so torch/audio/llama-cpp versions don't fight. A top-level `bench.py` orchestrates by spawning per-model runner scripts, feeding identical JSON jobs, and capturing identical metrics.

### Directory layout

```
tts-bench/
  bench.py                  # orchestrator
  prompts.py                # the 5 prompts from the homebase spec
  runners/
    pocket_tts_runner.py    # supports: cpu, mps
    neutts_runner.py        # supports: cpu, cuda, mps (--model {air,nano})
    luxtts_runner.py        # supports: cpu, cuda, mps
  reference/
    .gitkeep                # user drops reference.wav + reference.txt here for Pass 2
  venvs/
    pocket/                 # gitignored
    neutts/                 # gitignored — shared by Air and Nano
    luxtts/                 # gitignored
  results/
    <ISO-timestamp>/
      <model_config>/prompt-{N}.wav        # e.g. pocket_cpu/prompt-1.wav
      results.csv
      quality.csv
      run.json              # platform, devices, reference kind, prompt set hash, commit
      README.md             # rubric inline
  scripts/
    install.ps1             # Windows
    install.sh              # Mac/Linux
    bench.ps1 / bench.sh    # thin wrappers around `python bench.py`
  README.md
```

Install scripts are platform-specific; everything downstream is platform-agnostic Python using `pathlib`. No hardcoded `/` or `\`. No shell calls that aren't wrapped in `shutil.which()`/`sys.platform` checks.

### Model configs

A `model_config` is `<model>_<device>`. The set varies by platform:

- Windows-CPU: `pocket_cpu`, `neutts_air_cpu`, `neutts_nano_cpu`, `luxtts_cpu`
- Windows-CUDA: `neutts_air_cuda`, `neutts_nano_cuda`, `luxtts_cuda` (no `pocket_cuda` — Pocket-TTS has no GPU speedup)
- Mac-CPU: same as Windows-CPU
- Mac-MPS: `pocket_mps`, `neutts_air_mps`, `neutts_nano_mps`, `luxtts_mps`

That's up to **15 distinct model_configs across all platforms** × 5 prompts × 2 reference passes = a lot of WAVs. The orchestrator handles this by being incremental (resume support — re-running skips already-generated rows unless `--force`).

### Runner protocol

Each runner is invoked as:

```
<venv-python> runners/<model>_runner.py --job <path> --out <path>
```

`job.json`:

```json
{
  "prompt": "Open the browser and read my email.",
  "reference_wav": null,
  "reference_txt": null,
  "device": "cpu",
  "model_variant": "air",
  "warmup": false
}
```

`model_variant` is NeuTTS-only (`air` | `nano`); other runners ignore it.

Runner stdout (one JSON line):

```json
{
  "ttfa_ms": 412,
  "wall_s": 1.83,
  "audio_s": 2.1,
  "peak_rss_mb": 640,
  "device_used": "cpu",
  "ok": true
}
```

`device_used` is what the model actually ran on — lets the orchestrator detect silent MPS-to-CPU fallbacks. TTFA is captured inside the runner. Wall time is also measured externally by the orchestrator as a cross-check.

If a runner can't honor a requested device (e.g. CUDA not available), it errors immediately with `{"ok": false, "error": "cuda unavailable"}` rather than falling back silently — the orchestrator records the skip.

### Orchestrator (`bench.py`)

```
python bench.py --reference default
python bench.py --reference reference/reference.wav
python bench.py --reference default --devices cpu
python bench.py --reference default --models pocket
python bench.py --resume
```

Flags:
- `--reference {default | <path>}` — required
- `--devices <list>` — comma-sep `cpu,cuda,mps` (default: auto-detect available)
- `--models <list>` — comma-sep `pocket,neutts_air,neutts_nano,luxtts` (default: all)
- `--prompts <list>` — default: all 5
- `--cold-warm-prompt 2 --cold-warm-runs 3` — see Phase 2 below
- `--resume` — skip rows that already have a non-failed entry in this run dir
- `--out-dir <path>` — overrides `results/<timestamp>`

Device auto-detection: orchestrator runs a tiny probe subprocess inside one of the model venvs (the orchestrator process itself stays venv-free). The probe imports torch, returns `{"cuda": bool, "mps": bool, "platform": "...", "machine": "..."}` as JSON.

### Two-phase run

**Phase 1 — main sweep.** For each (model_config, prompt) pair, generate once. Row is tagged `cold`.

**Phase 2 — cold/warm sweep.** For each model_config, re-run `--cold-warm-prompt` (default 2) `--cold-warm-runs - 1` additional times back-to-back (default 2 more). These rows are tagged `warm`.

For each row the orchestrator:
1. Builds `job.json` in a tempfile.
2. Spawns the runner subprocess.
3. Polls `psutil.Process(pid).memory_info()` and `psutil.cpu_percent(percpu=True)` every 50ms on a background thread.
4. Parses runner stdout JSON for TTFA and audio duration.
5. Appends a row to `results.csv`.

CPU monitoring captures: peak count of cores simultaneously above 50%, plus per-core peak utilizations as a compact string (e.g. `[98,97,12,8,...]`). Enough to verify Pocket-TTS's "2 cores" claim by eye and to spot when a model is using all cores on a low-core box.

### `results.csv` columns

```
platform, device, model_config, prompt_id, ttfa_ms, wall_s, audio_s, rtf,
peak_rss_mb, cores_active_peak, per_core_peaks, cold_or_warm, reference_kind,
ok, error, commit, run_started_at
```

`platform` is `windows | macos | linux`. `commit` is the tts-bench git short SHA. `run_started_at` lets you correlate across platform-specific CSVs when joining the cross-platform rollup.

### `quality.csv` (blank, user fills in after listening)

```
platform, device, model_config, prompt_id, naturalness_1_5, similarity_1_5,
intelligibility_1_5, artifacts_1_5, notes
```

The run's `README.md` includes the rubric inline (naturalness, similarity, intelligibility, artifacts — 1-5 scale, no averaging) so the user can listen + score without flipping back to the homebase doc.

## Reference voice — two passes

Per Cicero's voice-clone requirement and the user's chosen flow:

**Pass 1 — default voices.** Each model uses its built-in default speaker. Voice-similarity scoring is N/A (each model is cloning a different voice) but naturalness/intelligibility/speed are all comparable. Establishes the speed picture.

**Pass 2 — user voice.** User drops `reference/reference.wav` (~5-10s clean speech) and `reference/reference.txt` (transcript). All models clone the same voice. Full apples-to-apples comparison including voice-similarity scoring.

Decision rubric is evaluated against Pass 2.

## Decision rubric (extended from homebase doc)

The homebase rubric was 1-vs-1 RTF/TTFA/quality. For 4 models × 3 device tiers it becomes a ranking. Top-line questions, answered from Pass 2:

1. **Per-platform speed winner.** Which model has the best RTF on each of (Windows-CPU, Windows-CUDA, Mac-CPU, Mac-MPS)? Same winner across all four = clear pick.
2. **TTFA floor.** Which models can hit <250ms TTFA on prompt 1 across all platforms? That's the Cicero interactivity floor.
3. **Quality floor.** Which models score ≥3/5 average on naturalness + similarity across all 5 prompts? Below 3 is disqualified regardless of speed.
4. **Footprint.** Peak RAM ≤1GB on the CPU configs? (Sovereign appliance constraint.)

A model passes Cicero Sovereign if it: wins-or-ties speed on ≥2 of 4 device tiers (within 20% of best RTF counts as "ties"), clears the TTFA floor on all tiers it supports, clears the quality floor, and fits the footprint. We expect one model to pass, possibly two; if zero pass, we re-scope.

Multilingual (prompt 5, French) only runs on Pocket-TTS and NeuTTS Nano — those are the models claiming non-English support. NeuTTS Air is English-only; LuxTTS multilingual status is TBD at install time. If LuxTTS supports French, the harness opts it in; if not, it skips with a logged note.

## Install (`scripts/install.ps1` and `scripts/install.sh`)

Both scripts do the same things, idempotently:

1. Create `venvs/pocket/`, `venvs/neutts/`, `venvs/luxtts/` via `uv venv`.
2. Install each model into its venv. Torch wheel selection is platform/device-driven: CPU wheels on Mac, CPU + CUDA wheels on Windows/Linux with NVIDIA, MPS-capable wheels on Mac.
3. Download / cache default voices for each model.
4. Pre-bake the user reference embedding into each model's expected format if `reference/reference.wav` exists.
5. Print a per-model status table at the end (installed / failed / device support).

Open install questions (resolved during implementation, surfaced loudly if they fail):

- Whether `pocket-tts` builds clean from source on Windows (the upstream repo's tooling is Linux/Mac primary).
- Whether `luxtts`'s ZipVoice base has a Windows wheel or needs a build step.
- Exact NeuTTS install command — there are GGUF variants via `llama-cpp-python` and a HF `transformers` path; we pick whichever Neuphonic recommends in their README.

Failures in any single model's install are non-fatal — the bench runs with the models that did install.

## Cross-platform implementation notes

- All file paths via `pathlib.Path`. No string concatenation of paths.
- All subprocess invocations use list-form args, not shell strings. `shell=False` everywhere.
- Line-ending normalization: `.gitattributes` set to `text eol=lf` for `.py`, `.sh`, `.md`. PowerShell scripts stay `eol=crlf`.
- Audio I/O via `soundfile` (cross-platform, no ffmpeg dep for WAV).
- No `os.system`, no `cmd /c`, no `bash -c`.
- WAV filenames use only `[a-z0-9_-]` — no colons or spaces (Windows path issues).
- `run.json` records full platform info: `sys.platform`, `platform.machine()`, `platform.python_version()`, OS version, CPU model.

## Error handling and edge cases

- Missing reference files in Pass 2 → orchestrator errors before any spawn.
- Runner crash on one prompt → row marked failed, bench continues.
- Device unavailable (e.g. `--devices cuda` on Mac) → silently skipped from auto-detect; if explicitly requested via `--devices` it's a clear error before any spawn.
- Subprocess hang → 60s wall-time timeout per generation; row marked timed-out, bench continues.
- Prompt 5 (French) → only attempted for models with declared multilingual support.
- Resume mid-run → `--resume` keeps existing successful rows, retries failed rows.

## Testing

Benchmark harness — automated correctness tests aren't the deliverable. What we test:

1. **Smoke run**: `python bench.py --reference default --prompts 1 --models pocket --cold-warm-runs 1` completes, writes a non-empty WAV, writes a valid CSV row.
2. **Cross-platform smoke**: same command on Mac (later). If both pass, the harness is portable.
3. **Manual ear test**: open the WAVs in the results dir, listen on headphones, fill in `quality.csv`. This is the actual deliverable.

## Out of scope

- Replicating ryzen-ai numbers from the homebase doc. We *can* run there later with the same harness; not required for v1.
- Patching Hermes to swap TTS provider (separate spec, only if Pocket-TTS or LuxTTS wins).
- Live audio playback / streaming integration. TTFA is measured but audio isn't piped to speakers.
- Automated audio quality scoring (UTMOS, WER, etc). Homebase spec is explicit that quality scoring is subjective ear-test.
- Quantization sweeps. NeuTTS GGUF Q4 vs Q8 is a separate axis; we use Q4 throughout (Neuphonic's recommended default).

## Deliverable

For each platform run (Windows, Mac, etc):
- A populated `results/<timestamp>/` directory with `results.csv`, filled-in `quality.csv`, WAVs for each (model_config, prompt) pair, and a `run.json` recording the platform/device/commit.

After all platforms are benched:
- A short rollup appended to the homebase doc's "After the bench" section with the per-platform speed winner, the cross-platform speed winner (if any), and the Cicero Sovereign decision.
