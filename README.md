# tts-bench

Personal speed bench for small/local TTS models. Cold + warm runs, same prompts, real wall clock — across CPU, CUDA, and Apple Silicon (MPS).

Built to answer one question: *which open TTS model do I plug into an always-on voice agent on the machine I actually have?*

---

## Current results (Windows CPU, x86, 2 cores, May 2026)

Average across 5 prompts (1 cold + 2 warm per prompt per model):

| Model | Variant | TTFA cold | TTFA warm | RTF cold | RTF warm | Notes |
|---|---|---|---|---|---|---|
| **Pocket-TTS** (Kyutai, 100M, MIT) | — | **95-150ms** | **97-150ms** | 2.9× | **2.9-3.0×** | wins on CPU; 26 predefined voices, BYO-voice gated by HF accept-terms |
| NeuTTS Nano (Neuphonic, GGUF Q4) | nano | 1.2s | 0.43-0.51s | 0.82× | 1.3-1.4× | multilingual fallback; separate model per language |
| NeuTTS Air (Neuphonic, GGUF Q4) | air | 1.7-1.9s | 0.67-0.70s | 0.68-0.85× | 0.88-0.90× | **below realtime on CPU** — needs GPU |
| LuxTTS (k2-fsa) | — | — | — | — | — | install blocked on Windows (see [Known issues](#known-issues)) |

**Reading the table:** TTFA = milliseconds until the first audio sample. RTF = `audio_seconds / generation_seconds` (1.0× = realtime, higher = faster than realtime). Pocket-TTS is **5-7× faster TTFA and 2-3× faster RTF** than the runner-up on this hardware.

Raw CSV + WAVs from the run live in `results/2026-05-23_1139/`.

Caveats: one machine, one run, two CPU cores. Re-bench on your own hardware before committing — see [Known issues](#known-issues) for examples of model README claims that didn't survive contact with a real install.

---

## Quick start

Requires [`uv`](https://github.com/astral-sh/uv) and Python 3.11. ~10-15 min total install (NeuTTS builds `llama-cpp-python` from source on first run).

```powershell
# Windows
.\install.ps1
python bench.py
```

```bash
# macOS / Linux
./install.sh
python bench.py
```

Each model gets its own venv under `venvs/` (isolated dependency trees — NeuTTS's torch needs vs Pocket-TTS's MLX path vs Kokoro's misaki tokenizer don't have to coexist).

### Interactive mode

To feel the latency yourself instead of staring at CSV rows:

```powershell
python speak.py pocket
python speak.py neutts_air
python speak.py neutts_nano --language fr
python speak.py neutts_nano --reference reference/myvoice.wav
```

Loads the model once, opens a prompt. Type, hear it, repeat. First turn is cold, subsequent are warm — that's what an always-on agent feels like.

### Voice cloning

NeuTTS Air and NeuTTS Nano do *zero-shot reference cloning*: drop a WAV into `reference/` with a matching `.txt` transcript file (same name, e.g. `myvoice.wav` + `myvoice.txt`), then:

```bash
python bench.py --reference reference/myvoice.wav
python speak.py neutts_air --reference reference/myvoice.wav
```

Pocket-TTS in this bench uses its 26 predefined voices (`anna` for English, `juergen` for German, `estelle` for French, etc.). Its zero-shot cloning path requires accepting the [`kyutai/pocket-tts` model terms on HuggingFace](https://huggingface.co/kyutai/pocket-tts).

---

## What this bench measures

Five prompts, English + one French, mixing conversational and technical content:

1. `"Open the browser and read my email."` — short / conversational
2. `"I'll start a new git branch, push the changes, and open a pull request when the tests pass."` — medium / dev-flavored
3. ~30-word Parakeet-TDT paragraph with embedded numbers — long / technical
4. Shell command read aloud — punctuation / symbol density stress
5. `"Bonjour, je m'appelle Cicero..."` — French / multilingual (skipped on EN-only models)

Per `(model, device, prompt)` cell:
- One subprocess loads the model once
- Generates the prompt **N times** (default 3 = 1 cold + 2 warm)
- Run 1 = cold (no warm cache, no JIT prime)
- Runs 2..N = warm (model resident, codecs primed)

Both numbers matter for different reasons. Cold matters for "user opens the app and says something". Warm matters for "user is in a conversation, every turn after the first".

---

## How it works

Three layers, all stdlib in the orchestrator:

| Layer | What |
|---|---|
| `bench.py` | Orchestrator. Picks (model × device × prompt) cells, spawns one subprocess per cell, parses JSON-line output, writes `results.csv`, prints summary. |
| `runners/*.py` | One runner per model. Loads the model in its own venv, generates audio, writes WAV, emits one JSON line per run to stdout. Also supports `--stdin` mode for `speak.py`. |
| `speak.py` | REPL that keeps a runner subprocess alive across turns. Uses `winsound` / `afplay` / `aplay` for playback. |

Runners communicate via JSON lines so each model can live in its own conflicting dependency tree. The orchestrator never imports any TTS library directly.

---

## Adding a model

1. Make a venv under `venvs/<name>/` (add to `install.ps1` and `install.sh`).
2. Write `runners/<name>_runner.py` matching the existing runner protocol:
   - Single-shot mode: `--text TEXT --out PATH --runs N` → one JSON line per run on stdout: `{"ok": true, "run_index": N, "ttfa_ms": ..., "gen_s": ..., "audio_s": ...}`
   - REPL mode: `--stdin` → on startup emit `{"ready": true}` after model load, then per stdin line: `{"text": "...", "out": "path.wav"}` → respond with one JSON line per generation
3. Add a row to `MODELS` in `bench.py` and `speak.py`.
4. Test: `python bench.py --models <name> --prompts 1 --runs 1`

The two existing runners (`pocket_runner.py`, `neutts_runner.py`) are ~150 lines each — short enough to copy and adapt.

---

## Known issues

- **`piper-phonemize` blocks LuxTTS on Windows.** LuxTTS depends on `piper-phonemize`, which only ships manylinux x86_64/aarch64 and macOS wheels — no Windows wheel exists. `install.ps1` marks LuxTTS as expected-fail; the bench skips it on Windows. Workaround: WSL2 (untested here) or run on Mac/Linux. (Note: `piper-tts` 1.4+ itself removed this dependency, so Piper proper works on Windows — only LuxTTS is affected.)
- **NeuTTS via `pip install neutts` gives the torch backbone, not the production fast path.** Torch backbone on x86 CPU runs at ~0.2× RTF (sub-realtime, unusable). The production path is `llama-cpp-python` + `neuphonic/neutts-*-q4-gguf` models. Post-switch numbers are what's in the table. `install.ps1` and `install.sh` install the GGUF path by default.
- **Pocket-TTS voice cloning is HF accept-terms gated.** Predefined voices (`anna`, `juergen`, `estelle`, etc.) work without auth. Passing a reference WAV triggers a fetch of the gated `kyutai/pocket-tts` model and 401s. Either accept the terms or use NeuTTS for BYO-voice.
- **NeuTTS reference voices need both `.wav` AND `.txt` (transcript)** in the same directory with the same basename. Wav-only fails inside `encode_reference()`.
- **NeuTTS Nano multilingual = separate model file per language.** `neuphonic/neutts-nano-q4-gguf` (English) vs `neuphonic/neutts-nano-french-q4-gguf` etc. The runner picks the right one based on `--language`.
- **`uv venv` doesn't seed pip.** Installs use `uv pip install --python <venv-python>` throughout. The install scripts codify this.
- **PowerShell 5.1 + native exes + `$ErrorActionPreference=Stop` is a trap.** `uv` prints to stderr on success (`"Using CPython 3.11.15"`); PowerShell wraps each stderr line as a `NativeCommandError` and trips Stop even on exit 0. `install.ps1` uses `$LASTEXITCODE` checks instead.

---

## Roadmap

Pending model additions (PRs welcome):

- **Kokoro-82M** — Apache 2.0, 54 preset voices, claimed 90-210× RT on CPU. Venv installed; runner WIP.
- **KittenTTS** — Apache 2.0, <100M, English only, predefined voices. Venv installed; runner pending.
- **Piper** (rhasspy/piper) — MIT, per-language voice models, no longer blocked on Windows in 1.4+. Pending.
- **Mac M4 Pro pass** — `install.sh` + MPS device detection are wired up; bench pending hardware.
- **LuxTTS on macOS** — should install cleanly per upstream (piper-phonemize macOS wheels exist).

These are all non-cloning small models (predefined voices only) — they fill a different niche than NeuTTS, which is the BYO-voice path. See `docs/superpowers/specs/2026-05-22-tts-bench-cross-platform-design.md` for the full bench design.

---

## Layout

```
tts-bench/
├── bench.py              # orchestrator
├── speak.py              # interactive REPL
├── install.ps1           # Windows installer
├── install.sh            # macOS / Linux installer
├── runners/
│   ├── pocket_runner.py
│   ├── neutts_runner.py
│   └── luxtts_runner.py
├── reference/            # voice cloning reference audio (.wav + .txt pairs)
├── venvs/                # one isolated venv per model (gitignored)
├── results/              # bench output WAVs + CSV (gitignored)
└── docs/                 # design specs
```

---

## License

MIT for the bench code in this repo. **Each TTS model has its own license** (Pocket-TTS: MIT, NeuTTS: Apache 2.0, Kokoro: Apache 2.0, LuxTTS: Apache 2.0, etc.) — check before deploying any of them in a product.

---

## Why this exists

The published latency numbers for small open TTS models are usually:
1. Run on different hardware than yours
2. Quote a single TTFA that conflates cold and warm
3. From the model's own README (vendor-favorable)

This bench fixes (1) by running on whatever hardware you put it on, (2) by reporting cold and warm separately, and (3) by being reproducible from a clean machine in <15 minutes. It already disproved one vendor claim during its first pass — NeuTTS Air's "2× realtime on AMD Ryzen 9" turned into 0.9× RTF on x86 Windows CPU.

If you're picking a small TTS model for a real-time agent, run it on the hardware that agent will actually run on. This is the harness for doing that.
