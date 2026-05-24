# tts-bench

Personal speed bench for small/local TTS models. Cold + warm runs, same prompts, real wall clock — across CPU, CUDA, and Apple Silicon (MPS).

Built to answer one question: *which open TTS model do I plug into an always-on voice agent on the machine I actually have?*

---

## ▶ Demos

**[5uck1ess.github.io/tts-bench](https://5uck1ess.github.io/tts-bench/)** — public side-by-side audio.

Every model × prompt × device combination is rendered with an inline `<audio>` player so you can hear the actual output without cloning the repo or running anything locally. Useful for:

- *Picking a model.* Listen to the same prompt across 18 TTS models on the same hardware. Quality, prosody, and artifacts are obvious in 5 seconds; benchmark tables can't show that.
- *Comparing rigs.* Each report is tagged with the rig that produced it (Ryzen 9 9950X3D, Apple M4, etc.) so you can see how the same model sounds on the box you actually own.
- *Comparing devices for one model.* CPU vs CUDA vs MPS rows for the same model, side by side, with their audio.

You can also build and publish your own demos from any bench run — see [Generating local demos](#generating-local-demos) and [Publishing your own to GitHub Pages](#publishing-your-own-to-github-pages) below.

---

## Test hardware

Listed because TTS speed is hardware-dependent — RTF claims that hold on a Ryzen 9 won't necessarily hold on a Raspberry Pi.

| Machine | OS | CPU | RAM | GPU | Used for |
|---|---|---|---|---|---|
| **Windows desktop** | Windows 11 Pro | AMD Ryzen 9 9950X3D (16C / 32T @ 4.3 GHz base) | 128 GB | NVIDIA RTX 5090 | Windows CPU bench rows (GPU runs pending) |
| **Mac** | macOS 26.5 | Apple M4 (10C) | 16 GB | Apple M4 GPU (MPS) | Mac CPU + MPS bench rows |

If you reproduce on different hardware, your numbers will differ — file an issue or PR with your results and we'll add a column.

## Current results (May 2026)

Same five prompts run on both rigs above. Numbers shown are from short prompts; long prompts scale RTF linearly. Warm averages over runs 2-3 across all prompts the model can speak.

### Windows desktop — Ryzen 9 9950X3D CPU

#### Predefined-voice models (pick from a baked-in voice list)

| Model | Size / License | TTFA cold | TTFA warm | RTF warm | Languages | Notes |
|---|---|---|---|---|---|---|
| **[Piper](https://github.com/OHF-voice/piper1-gpl)** (OHF-voice, formerly rhasspy) | per-voice ~25MB / MIT | **72ms** | **39ms** | **47×** | 40+ via separate voice models | leader on this hardware; streaming-native, bundles espeak-ng (no Windows wheel pain) |
| **[Kokoro-82M](https://github.com/hexgrad/kokoro)** (hexgrad) | 82M / Apache 2.0 | 335ms | 245ms | 13× | 9 (a/b/e/f/h/i/j/p/z codes) | 54 voices; misaki tokenizer needs spaCy preinstall (see Known issues) |
| **[KittenTTS](https://github.com/KittenML/KittenTTS)** (KittenML) | <100M / Apache 2.0 | 516ms | 487ms | 6.6× | EN only | 8 voices; non-streaming so TTFA == gen_s |
| **[VibeVoice-Realtime-0.5B](https://github.com/vibevoice-community/VibeVoice)** (Microsoft, community fork) | 0.5B / MIT | ~3.9s | ~3.7s | **~0.5×** | EN only (7 preset voices) | streaming-class but heavy diffusion; DDPM steps tunable (5 default). Predefined `.pt` voice embeddings auto-downloaded |
| [Magpie-TTS Multilingual 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) (NVIDIA, NeMo) | 357M / NVIDIA Open Model License | pending | pending | pending (CUDA: ~1.0× cold smoke) | 9 (en/es/de/it/vi/zh/fr/hi/ja) | fixed speaker embeddings (this checkpoint variant); HF accept-terms gated; install skips `[tts]` extra to avoid `pynini` on Windows — runner forces `apply_TN=False` to compensate |

#### Zero-shot voice cloning models (accept a reference wav at inference time)

| Model | Size / License | TTFA cold | TTFA warm | RTF warm | Cloning ref | Notes |
|---|---|---|---|---|---|---|
| **[Pocket-TTS](https://github.com/kyutai-labs/pocket-tts)** (Kyutai, predefined mode) | 100M / MIT | 95-150ms | 97-150ms | **2.9-3.0×** | wav or voice name | 26 voices unauth; BYO-voice path is HF accept-terms gated on `kyutai/pocket-tts` |
| [NeuTTS Nano](https://github.com/neuphonic/neutts) (GGUF Q4) | 748M / Apache 2.0 | 1.2s | 0.43-0.51s | 1.3-1.4× | wav + transcript | multilingual fallback; separate `.gguf` per language |
| [NeuTTS Air](https://github.com/neuphonic/neutts) (GGUF Q4) | 748M / Apache 2.0 | 1.7-1.9s | 0.67-0.70s | 0.88-0.90× | wav + transcript | below realtime on CPU — needs GPU |
| [ChatterBox-TTS](https://github.com/resemble-ai/chatterbox) (Resemble AI) | ~1.2B / MIT | ~8s | ~8s | **~0.30×** | wav (no transcript) | 1000 diffusion steps — GPU-targeted, community quality leader |
| [F5-TTS](https://github.com/SWivid/F5-TTS) (v1 Base) | ~330M / MIT | ~48s | ~48s | **~0.05×** | wav + transcript | flow matching, very slow on CPU; needs GPU |
| [Coqui XTTS-v2](https://github.com/idiap/coqui-ai-TTS) (idiap fork) | ~750M / CPML 1.0 (non-commercial) | pending | pending | pending | wav (no transcript) | de facto multilingual cloning baseline; ~2GB download on first use; auto-accepts CPML via `COQUI_TOS_AGREED=1` |
| [OmniVoice](https://github.com/k2-fsa/OmniVoice) (k2-fsa) | TBD / see upstream | pending | pending | pending | wav + transcript | 600+ languages; diffusion-LM, vendor-claimed 0.025× RTF (GPU); voice design tags (gender/age/whisper) |
| [VoxCPM2](https://github.com/OpenBMB/VoxCPM) (OpenBMB) | 2B / see upstream | pending | pending | pending | wav (no transcript) | tokenizer-free, 48kHz, 30 langs; in-process via `voxcpm` pip pkg (not the optional Nano-vLLM server path). Earlier 0.5B variant doesn't support cloning — skipped. |
| [Qwen3-TTS-Base 1.7B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) (Alibaba Qwen) | 1.7B / Apache 2.0 | pending | pending | pending | wav + transcript | 10 langs (zh/en/ja/ko/de/fr/ru/pt/es/it); claimed 97ms streaming TTFA; FlashAttention 2 skipped on Windows |
| [IndexTTS-2](https://github.com/index-tts/index-tts) (Bilibili Index) | ~1.5B / Apache 2.0 | pending | pending | pending | wav (no transcript) | zero-shot cloning + optional emotion-reference conditioning; source-clone install (no pip wheel); ~5 GB weights auto-download from HF on first use |
| [Sesame CSM-1B](https://huggingface.co/sesame/csm-1b) (Sesame AI) | 1B / Apache 2.0 | pending | pending | pending | wav + transcript (as prior-turn context) | conversational speech model; in-context cloning via apply_chat_template; native transformers >= 4.52.1; **HF manual-approval gated** — request access on the model page before first run |
| [MARS5-TTS](https://github.com/Camb-ai/MARS5-TTS) (CAMB.AI) | ~1.2B (750M AR + 450M NAR) / **AGPL-3.0** | pending | pending | pending | wav (shallow clone) or wav + transcript (deep clone, higher quality) | English-only; loaded via `torch.hub.load`; reference audio must be 1-12 seconds; **AGPL-3.0 means non-commercial unless you license from CAMB.AI** |
| [LuxTTS](https://github.com/ysharma3501/LuxTTS) (k2-fsa-based) | — | — | — | — | wav | install blocked on Windows (see [Known issues](#known-issues)) |

**Reading the tables:** TTFA = milliseconds until the first audio sample. RTF = `audio_seconds / generation_seconds` (1.0× = realtime, higher = faster than realtime). Non-streaming models (KittenTTS, ChatterBox, F5-TTS) emit full audio in one call so TTFA = gen_s by definition.

**Top-line takeaway:** if you don't need voice cloning, **Piper wins by a huge margin on this CPU** (39ms warm TTFA, 47× RTF). Pocket-TTS is the fastest cloning-capable option (with the HF accept-terms caveat). NeuTTS Air/Nano give clean BYO-voice without auth gates but at lower RTF. ChatterBox + F5-TTS are GPU-class — file them under "bench-cold but not deployable" until 5090 runs land.

### Mac — Apple M4 (10C, 16 GB) CPU + MPS

Same harness, 5 prompts × 3 runs each, warm averages across all runnable prompts. ChatterBox / F5-TTS / Coqui XTTS not run on this rig — already labeled GPU-class for CPU; M4 CPU would be even worse.

#### Predefined-voice models

| Model | Device | TTFA cold | TTFA warm | RTF warm | Notes |
|---|---|---|---|---|---|
| **[Piper](https://github.com/OHF-voice/piper1-gpl)** | cpu | **268ms** | **202ms** | **33.0×** | still the leader. 5× slower TTFA than Ryzen 9 but well above the headroom needed for an always-on agent |
| **[Kokoro-82M](https://github.com/hexgrad/kokoro)** | mps | 2995ms | 486ms | **15.4×** | MPS gives ~50% RTF lift over CPU after warmup; cold-load tax (~3s) hits the first turn |
| **[Kokoro-82M](https://github.com/hexgrad/kokoro)** | cpu | 994ms | 741ms | 10.2× | |
| **[KittenTTS](https://github.com/KittenML/KittenTTS)** | cpu | 929ms | 1031ms | 8.0× | non-streaming so TTFA == gen_s; CPU-only (no MPS path in upstream) |
| **[VibeVoice-Realtime-0.5B](https://github.com/vibevoice-community/VibeVoice)** | mps | 10760ms | 8287ms | **1.1×** | finally at realtime on MPS — CPU can't get there. Still 10s+ first-turn cold load |
| **[VibeVoice-Realtime-0.5B](https://github.com/vibevoice-community/VibeVoice)** | cpu | 30305ms | 25519ms | 0.4× | below realtime on M4 CPU (Ryzen 9 hit ~0.5×; M4's fewer cores hurt diffusion) |
| **[Magpie-TTS Multi 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)** (NVIDIA NeMo) | cpu | 26716ms | 27459ms | 0.4× | 9 langs (en/es/de/it/vi/zh/fr/hi/ja). HF-gated; works once `hf auth login` is done. NeMo CPU is heavy — close to RTX 5090's 0.97× drops to 0.4× on M4 |

#### Zero-shot voice cloning models

| Model | Device | TTFA cold | TTFA warm | RTF warm | Cloning ref | Notes |
|---|---|---|---|---|---|---|
| **[Pocket-TTS](https://github.com/kyutai-labs/pocket-tts)** (predefined mode) | cpu | **77ms** | **42ms** | **7.8×** | wav or voice name | fastest cloning-capable option here too. M4 single-thread perf actually beats Ryzen 9 on TTFA |
| [NeuTTS Nano](https://github.com/neuphonic/neutts) (GGUF Q4) | cpu | 815ms | 270ms | 3.0× | wav + transcript | multilingual via separate `.gguf` per language |
| [NeuTTS Nano](https://github.com/neuphonic/neutts) (GGUF Q4) | mps | 1491ms | 444ms | 2.8× | wav + transcript | MPS gives no win — GGUF inference runs CPU-side via llama-cpp |
| [NeuTTS Air](https://github.com/neuphonic/neutts) (GGUF Q4) | cpu | 1436ms | 364ms | 2.1× | wav + transcript | ~2.4× faster than Windows numbers thanks to M4 single-thread |
| [NeuTTS Air](https://github.com/neuphonic/neutts) (GGUF Q4) | mps | 2399ms | 568ms | 2.1× | wav + transcript | same — MPS doesn't help GGUF path |
| **[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)** (k2-fsa, 600+ langs) | mps | 5802ms | 5064ms | **0.9×** | wav + transcript | nearly realtime on MPS for short/medium prompts. **Long prompts (30+ words) OOM the MPS allocator** on 16 GB at ~3.4 GiB — 1 of 5 prompts failed |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (k2-fsa, 600+ langs) | cpu | 13653ms | 11444ms | 0.6× | wav + transcript | below realtime on CPU, expected for diffusion-LM |
| [VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B) (OpenBMB) | cpu | 10883ms | 9582ms | 0.7× | wav only | no MPS path in harness; reasonably close to RTX 5090's 1.0× — VoxCPM is less GPU-dependent than the others |
| [LuxTTS](https://github.com/ysharma3501/LuxTTS) (zipvoice-based) | — | — | — | — | wav | install blocked on arm64 Mac too (see [Known issues](#known-issues)) |

**Top-line takeaway on Mac:** Piper wins again (33× RTF, 202ms warm TTFA — drop-in for an always-on agent). Among cloning models, **Pocket-TTS is the clear winner on M4** — its 42ms warm TTFA actually beats the Windows number, because Pocket-TTS is single-thread dominated and M4 has strong single-thread perf. VibeVoice/MPS is the only diffusion-class model that reaches realtime on this machine; CPU diffusion isn't viable. NeuTTS gets no MPS benefit because its hot path is GGUF (llama-cpp, CPU-side). The new GPU-class additions (OmniVoice, VoxCPM, Magpie) all land sub-realtime on M4 — useful as "works at all on a Mac" data points, not as deploy candidates.

Raw CSVs from the runs live in `results/2026-05-23_1542/` (predefined pass 1), `results/2026-05-23_1600/` (cloning pass 2), and `results/2026-05-23_2152/` (omnivoice + voxcpm + magpie pass 3).

Caveats: one machine, one run. Re-bench on your own hardware before committing — see [Known issues](#known-issues) for examples of model README claims that didn't survive contact with a real install.

---

## Quick start

Requires [`uv`](https://github.com/astral-sh/uv) and Python 3.11. ~10-15 min total install (NeuTTS builds `llama-cpp-python` from source on first run).

**Disk budget: ~60 GB once every model has loaded at least once.** ~16 GB across the per-model `venvs/`, plus ~40 GB of model weights downloaded to `~/.cache/huggingface/` (and a smaller `~/.cache/modelscope/` for VoxCPM2). `results/` and `_gh-pages/` are negligible by comparison — local benches cost <100 MB unless you keep dozens of historical runs. To prune, just `rm -rf venvs/<model>/` for models you don't want; the install scripts are idempotent so you can re-add later.

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

### Interactive feel-test

To feel the latency yourself instead of staring at CSV rows:

```powershell
python speak.py pocket
python speak.py neutts_air
python speak.py neutts_nano --language fr
python speak.py neutts_nano --reference reference/myvoice.wav
```

Loads the model once, opens a prompt. Type, hear it, repeat. First turn is cold, subsequent are warm — that's what an always-on agent feels like.

### One-shot A/B compare

Run one phrase through every installed model on every available device, hear them back-to-back, see the comparison table:

```powershell
python compare.py "hello, this is a quick test of every model"
python compare.py --file reference/reference.txt          # paragraph
python compare.py "..." --devices cpu                     # CPU only
python compare.py "..." --reference reference/chris.wav   # cloning models only
python compare.py "..." --no-play                         # silent batch run
```

Wavs land in `results/compare/<timestamp>/<model>_<device>.wav` so you can re-listen later.

### Generating local demos

`bench.py` writes a self-contained dark-mode `report.html` next to each run's `results.csv` — sortable-looking table of TTFA cold/warm + RTF cold/warm per (model, device, prompt), with an inline `<audio>` player per cell so you can click-play each wav without leaving the page. A top-level `results/index.html` lists every run with a link to its report. This is what gets published to the [public Demos](#-demos).

```powershell
python report.py results/2026-05-23_1409          # one report
python report.py results/2026-05-23_1409 --open   # ... and open it in a browser
python report.py --all                            # regenerate everything
python report.py --index                          # just rebuild results/index.html
```

No dependencies — pure stdlib + a small HTML/CSS template. The report has a light/dark theme toggle (top-right, persisted to localStorage), sortable columns (click any header), and a live row filter. Reports regenerate from `results.csv`, so you can tweak `report.py` and re-run `--all` to refresh historical runs.

### Publishing your own to GitHub Pages

`publish.py` ships a chosen run to a `gh-pages` branch (managed via a git worktree at `_gh-pages/`, never touches master) so anyone can view it without cloning the repo — same mechanism as the [public Demos](#-demos) at the top of this README.

```powershell
python publish.py results/2026-05-23_2203          # publish + push
python publish.py results/2026-05-23_2203 --no-push # commit to gh-pages, push later
python publish.py --list                            # show what's already published
```

After the first push, enable Pages in GitHub repo settings → Pages → Source: "Deploy from a branch" → Branch: `gh-pages` / root. Reports land at `https://<user>.github.io/<repo>/<run-name>/report.html`; the root index lists all published runs.

The `gh-pages` branch holds report HTML + wavs + CSV — so audio playback works for any visitor (no need to clone the repo to hear the samples). Cross-machine workflow: each machine publishes its own runs (since the wavs stay on the machine that produced them — they're gitignored on master). The `_gh-pages/` worktree is created lazily on first `python publish.py` call; on second machines it tracks the existing remote `gh-pages` branch automatically.

### Voice cloning

Three flavors of "zero-shot cloning" are supported, with slightly different reference-file requirements:

| Models | Reference needed |
|---|---|
| ChatterBox, Coqui XTTS-v2, LuxTTS, VoxCPM | wav only — no transcript |
| NeuTTS Air, NeuTTS Nano, F5-TTS, OmniVoice | wav **+** matching `.txt` transcript (same basename, e.g. `myvoice.wav` + `myvoice.txt`). The transcript MUST be the literal words spoken in the wav. |
| Pocket-TTS (cloning path) | wav, HF accept-terms gated on [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts) + `hf auth login` |

Drop the wav (and optional `.txt`) into `reference/`, then:

```bash
python bench.py --reference reference/myvoice.wav
python compare.py "..." --reference reference/myvoice.wav
python speak.py chatterbox --reference reference/myvoice.wav
```

`--reference` auto-skips models that can only use predefined voices (Kokoro, KittenTTS, Piper, VibeVoice, Magpie).

### Subjective cloning quality notes

Numbers tell you which model is fast; ears tell you which one is *good*. Tested on Windows + RTX 5090 with `chris_hemsworth_15s.wav` reference (the cut version — full 67s wav blew through NeuTTS's 2048-token context):

1. **OmniVoice — best cloning fidelity** (top of the listening test). Surprisingly preserves the **source accent** (Chris Hemsworth's Australian came through), which most zero-shot cloners flatten to neutral American. Caveat: audible artifacts (glitches / noise) layered on top of the cloned voice. If those can be tamed it's the pick.
2. **ChatterBox — second best.** Cleaner output than OmniVoice (no artifacts), but doesn't carry the accent as faithfully. Right answer when you want clean audio and accent isn't critical.
3. **Coqui XTTS-v2** — clone fidelity weaker than ChatterBox. Multilingual baseline is its strongest reason to keep around.
4. **Pocket-TTS** — terrible at cloning (output is mostly artifacts, not usable). Keep it for **predefined voices only** — it's the fastest cloning-capable model in that mode.
5. **NeuTTS Air / Nano** — works, but on long-form inputs both truncated at exactly 1.9s in our compare run; likely a GGUF context/decode cap that needs investigation.

These are first-pass impressions on a single reference; replication with other references (jo, juliette) recommended before treating them as definitive.

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

Four layers, all stdlib in the orchestrator:

| Layer | What |
|---|---|
| `harness.py` | Shared model registry + subprocess plumbing. Defines `MODELS`, builds the list of runnable `(model, device)` cells for the current machine, owns the JSON-line protocol. Imported by everything else. |
| `bench.py` | Formal benchmark. Loops prompt × cell × runs, writes `results.csv`, prints per-prompt summary. |
| `compare.py` | One-shot A/B listening tool. Takes one piece of text, runs it through every installed model on every available device, dumps a wav per cell, plays them out loud as they finish, prints a comparison table. |
| `speak.py` | Interactive REPL. Keeps a single runner subprocess alive across turns so warm-run latency is measurable. Uses `winsound` / `afplay` / `aplay` for playback. |
| `runners/*.py` | One runner per model. Loads the model in its own venv, generates audio, writes WAV, emits one JSON line per run to stdout. Also supports `--stdin` mode for `speak.py`. |

Runners communicate via JSON lines so each model can live in its own conflicting dependency tree. The orchestrator never imports any TTS library directly.

### Three tools, three jobs

- **`bench.py`** — numbers. 5 prompts × every model × cold + warm. CSV output. Reach for this when you want hard data.
- **`compare.py`** — ears. One phrase → every model → audio out loud, side by side. Reach for this when you want to *hear* which model sounds best on your line.
- **`speak.py`** — feel. REPL that holds one model in memory. Type a prompt, hear it, type the next. Reach for this when you want to feel the warm-run latency of one model interactively.

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

Frictions surfaced while building the harness. None are blockers on Mac/Linux — most apply to Windows or to specific models.

**Cross-cutting**

- **`uv venv` doesn't seed pip.** `python -m pip install ...` fails inside a fresh uv venv. Installs use `uv pip install --python <venv-python>` throughout. The install scripts codify this.
- **PowerShell 5.1 + native exes + `$ErrorActionPreference=Stop` is a trap.** `uv` prints to stderr on success (`"Using CPython 3.11.15"`); PowerShell wraps each stderr line as a `NativeCommandError` and trips Stop even on exit 0. `install.ps1` uses `$LASTEXITCODE` checks instead.

**Per-model**

- **OmniVoice MPS — OOM on long prompts on 16 GB Macs.** The 30-word Parakeet paragraph (prompt 3 in the bench) hits "MPS backend out of memory (MPS allocated: 3.38 GiB)" on a 16 GB M4. Short/medium prompts (5-15 words) generate fine at ~0.9× RTF. If you need long-form on MPS, fall back to OmniVoice/cpu (0.6× RTF) or use a 32 GB+ Mac.
- **LuxTTS — blocked on Windows AND Apple Silicon.** Depends on `piper-phonemize` 1.1.0 which only ships wheels for `manylinux_2_28_{x86_64,aarch64}` and `macosx_10_14_x86_64` (Intel Mac only). On arm64 macOS (M-series) the install fails with "no wheels with a matching platform tag (e.g., `macosx_26_0_arm64`)". Workaround: build piper-phonemize from source (untested) or use Linux. The README previously claimed "should install cleanly on macOS — piper-phonemize macOS wheels exist" — that's only true for Intel Macs. Also note the cloned repo's pyproject lists `name = "Zipvoice"` even though the GitHub repo is named LuxTTS, so the runner imports `from zipvoice.luxvoice import LuxTTS`. (Piper proper is unaffected — `piper-tts` 1.4+ bundles espeak-ng and dropped the piper-phonemize dependency.)
- **NeuTTS — `pip install neutts` gives the torch backbone, not the production fast path.** Torch backbone on x86 CPU runs at ~0.2× RTF (unusable). Production path is `llama-cpp-python` + `neuphonic/neutts-*-q4-gguf` models. Post-switch RTF is what's in the table. Install scripts install the GGUF path by default.
- **NeuTTS — reference voices need both `.wav` AND `.txt` (transcript).** Wav-only fails inside `encode_reference()`. Pocket-TTS in contrast takes a voice name string or a single wav (gated).
- **NeuTTS Nano — multilingual = separate model file per language.** `neuphonic/neutts-nano-q4-gguf` (EN), `neuphonic/neutts-nano-french-q4-gguf` (FR), etc. Runner switches based on `--language`.
- **Pocket-TTS — voice cloning is HF accept-terms gated.** Predefined voices work without auth. Reference-wav cloning triggers a fetch of the gated `kyutai/pocket-tts` repo and 401s. Either accept terms or use NeuTTS Air/Nano for BYO-voice without auth.
- **Kokoro — misaki tokenizer calls `spacy.cli.download()` at init.** Tries to install `en_core_web_sm` via pip; fails in uv venvs. Install scripts pre-install the model wheel directly to bypass this.
- **KittenTTS — needs `espeakng-loader` (bundles espeak-ng DLL).** System espeak install also works on Mac/Linux but is more friction. Runner sets `ESPEAK_DATA_PATH` env var because the bundled DLL has a hardcoded CI build path.
- **ChatterBox — needs `setuptools<80`.** The `perth` audio watermarker imports `pkg_resources` (removed in setuptools 80+). Install scripts pin the version.
- **F5-TTS — `torchaudio.load()` routes through `torchcodec` in torch 2.12+**, which needs FFmpeg shared DLLs (libtorchcodec_core4.dll etc., NOT just ffmpeg.exe). On Windows with the typical static FFmpeg build this fails. Runner monkey-patches `torchaudio.load` to use `soundfile` directly. Install scripts also pin `datasets<3.0` to avoid pulling torchcodec into the import chain.
- **VibeVoice — install from `vibevoice-community/VibeVoice`, NOT pypi or `microsoft/VibeVoice`.** pypi `vibevoice==0.0.1` only ships the base architecture (no streaming class). The official Microsoft repo was taken down in September 2025 then partially restored without code. The community fork at github.com/vibevoice-community/VibeVoice keeps the original code and added a working streaming variant on 2025-12-04. Voice presets (`.pt` files, 2-4MB each) are not bundled in the package — runner auto-downloads them from the fork to `~/.cache/vibevoice-voices/` on first use.
- **VibeVoice — the "you should probably TRAIN this model" warning is benign.** The HF checkpoint deliberately omits the `acoustic_tokenizer.encoder` weights (~400 keys load as random). That subnet is unused at inference because the `.pt` voice presets are pre-encoded representations. Audio output is unaffected.

---

## Pending work

- **RTX 5090 full bench pass** — formal `bench.py --device cuda` run for all 18 installed models with the cloning reference. CUDA torch is installed; cold/warm + memory numbers pending.
- **Memory tracking in reports** — CPU RSS + CUDA peak VRAM per (model, device, prompt) cell. Plumbing added to the runner JSON-line protocol; back-fill across all runners pending.
- **ChatterBox / F5-TTS / Coqui XTTS on Mac MPS** — skipped earlier pass (all three are GPU-class on CPU; M4 CPU would be minutes per prompt). Worth re-running on Mac once MPS torch perf improves, or on a dedicated GPU Mac.
- **LuxTTS on Mac arm64 / Windows** — depends on piper-phonemize, which has no wheels for those platforms; needs build-from-source or upstream wheel.

## Considered but skipped

Models that were evaluated for inclusion and intentionally left out, with the reason. The bar is in `harness.py`'s scope: must install cleanly cross-platform in a self-contained venv, run end-to-end on CPU at all (even if slow), and expose a Python API that fits the runner protocol.

- **[Fish Audio S2 / S2-Pro](https://github.com/fishaudio/fish-speech)** (current `main` branch, 4B params). Linux/WSL only per official docs (Windows native unsupported); 24GB VRAM floor; no clean Python API — inference is a 3-stage CLI pipeline (DAC encode → text2semantic → DAC decode) with intermediate `.npy` files; research-license non-commercial. Doesn't fit the harness pattern. Run via the upstream SGLang/vLLM serving setup if you want it.
- **[Fish Audio S1-mini](https://huggingface.co/fishaudio/s1-mini)** (0.5B distilled S1). Small enough to fit in principle, but the S1 inference code lives at a specific mid-2025 *commit* (no tag) — pre-S2 branch's `v1.5.1` doesn't pair (different `firefly-gan-vq-*` filenames), and `v2.0.0-beta` is S2-only. Pinning to commit `781bf1cd` works, but pulls a heavy dep tree (lightning, wandb, gradio, faster_whisper, modelscope, funasr, `pyaudio`) and still needs the same 3-stage CLI subprocess wrapper. CC-BY-NC-SA license. Worth revisiting if the user wants the model specifically — the install path just doesn't pay for itself in a "fast comprehensive bench" context.
- **[Orpheus TTS](https://github.com/canopyai/Orpheus-TTS)** (Canopy Labs, 3B Llama-based). High-quality emotional/empathetic speech with ~200ms streaming latency, but the inference path is `pip install orpheus-speech` which depends on **vllm**. vllm has no first-party Windows wheel; community Windows wheels exist (SystemPanic, devnen) but the Blackwell-compatible one (5090 / sm_120) is pinned to vllm 0.20.0 + cu132 while Orpheus pins vllm 0.7.3 — version mismatch is unrecoverable without rebuilding from source. Linux/WSL or Mac (via LM Studio + GGUF) work, but this is a Windows-primary bench so Orpheus is out of scope here. Revisit on the Mac M4 Pro pass.
- **[CosyVoice 3](https://github.com/FunAudioLLM/CosyVoice)** (FunAudioLLM, Fun-CosyVoice3-0.5B-2512). High-quality multilingual cloning + 18 Chinese dialects, but installs as a source clone (no pip wheel) with a researcher-pinned `requirements.txt`: `torch==2.3.1` from a cu121 extra-index (incompatible with Blackwell sm_120 / 5090 — would need a post-install swap), `onnxruntime==1.18.0` pinned in a way uv resolves against a Linux-only CUDA wheel feed first, plus 40+ other exact-version pins (`gradio==5.4.0`, `lightning==2.2.4`, `openai-whisper==20231117`, `transformers==4.51.3`, `tensorrt-cu12` Linux-only). Same family of issue as fish-speech and Orpheus: not a clean cross-platform install. Doable with significant install-archaeology effort — revisit if/when the upstream ships a Windows-friendly requirements set.
- **[Higgs-Audio v2](https://github.com/boson-ai/higgs-audio)** (Boson AI, 3B LLM + 2.2B audio adapter). Source-clone install + cu128 torch worked cleanly, but the v2 model checkpoint on HF (`bosonai/higgs-audio-v2-generation-3B-base`) declares `model_type: higgs_audio_v2` while the upstream `boson_multimodal` package only registers `higgs_audio` (v1). Aliasing the config gets past the AutoConfig lookup but fails a deeper `model_type` consistency check in transformers — and even if patched, the v2 architecture (`HiggsAudioV2ForConditionalGeneration`) likely has different layer shapes than v1's `HiggsAudioModel`. Real fix needs upstream to ship the v2 class registration; revisit when boson-ai/higgs-audio gets a commit that adds it.
- **[Voxtral 4B TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)** (Mistral AI, 4B, 9 langs, 20 preset voices). Same vLLM-on-Windows-Blackwell wall as Orpheus: the HF model card declares `library_name: "vllm"` and the file layout is Mistral's custom format (`consolidated.safetensors`, `params.json`, `tekken.json`) — no `config.json` / `tokenizer.json`, so `transformers.AutoModel` can't load it at all. Inference is only documented via vLLM-Omni (vllm >= 0.18.0). Linux/WSL would work, and there's a community MLX port for Mac (`jason1966/Voxtral-TTS-MLX`), but the Windows-primary bench is out of scope. Revisit on the Mac M4 Pro pass via the MLX port, or when a Linux/WSL pass is added.
- **[Step-Audio-TTS-3B](https://huggingface.co/stepfun-ai/Step-Audio-TTS-3B)** (StepFun AI, 3.5B). Two blockers stacked: (1) bundles Linux-only compiled C++ extensions (`lib/liboptimus_ths-torch2.X-cu12X.cpython-310-x86_64-linux-gnu.so`) that `modeling_step1.py` imports at runtime via `trust_remote_code` — no Windows or Mac equivalents shipped. (2) The upstream github repo linked from the model card (`stepfun-ai/Step-Audio`) returns 404 — appears abandoned/renamed. The successor (`stepfun-ai/Step-Audio2`) is a different project (audio understanding LLM, not TTS). With no current install docs and Linux-only native deps, this is effectively a Linux-only model. Revisit on a Linux/WSL pass, or if StepFun publishes a cross-platform inference path.

---

## Layout

```
tts-bench/
├── harness.py            # shared model registry + subprocess plumbing (imported by all 3 tools)
├── bench.py              # formal benchmark — CSV + per-prompt summary + auto-generates report.html
├── compare.py            # one-shot A/B listening tool — every model × every device, plays out loud
├── speak.py              # interactive REPL — feel warm-run latency for one model
├── report.py             # build HTML report (table + inline audio players, light/dark, sort, filter) from a results/ dir
├── publish.py            # ship a chosen run to the gh-pages branch for GitHub Pages hosting (managed via a git worktree at _gh-pages/)
├── install.ps1           # Windows installer
├── install.sh            # macOS / Linux installer
├── runners/
│   ├── pocket_runner.py
│   ├── neutts_runner.py
│   ├── luxtts_runner.py
│   ├── kokoro_runner.py
│   ├── kittentts_runner.py
│   ├── piper_runner.py
│   ├── chatterbox_runner.py
│   ├── f5tts_runner.py
│   ├── coqui_runner.py
│   ├── vibevoice_runner.py
│   ├── omnivoice_runner.py
│   ├── voxcpm_runner.py
│   ├── magpie_runner.py
│   ├── qwentts_runner.py
│   ├── indextts_runner.py
│   ├── sesame_runner.py
│   └── mars5_runner.py
├── reference/            # voice cloning reference audio (.wav + .txt pairs)
├── venvs/                # one isolated venv per model (gitignored)
└── results/              # bench output WAVs + CSV (gitignored)
```

---

## License

MIT for the bench code in this repo. **Each TTS model has its own license** — check the linked upstream repos above before deploying any of them in a product. Quick reference:

- MIT: [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts), [Piper](https://github.com/OHF-voice/piper1-gpl), [ChatterBox](https://github.com/resemble-ai/chatterbox), [F5-TTS](https://github.com/SWivid/F5-TTS), [VibeVoice (community fork)](https://github.com/vibevoice-community/VibeVoice)
- Apache 2.0: [NeuTTS](https://github.com/neuphonic/neutts), [Kokoro](https://github.com/hexgrad/kokoro), [KittenTTS](https://github.com/KittenML/KittenTTS), [LuxTTS](https://github.com/ysharma3501/LuxTTS)
- **CPML 1.0 (non-commercial):** [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2) — research / personal use only. The harness auto-accepts via `COQUI_TOS_AGREED=1`.
- **NVIDIA Open Model License:** [Magpie-TTS Multilingual 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) — commercial use permitted with terms; HF accept-terms gated.
- **Check upstream — see model repo:** [OmniVoice](https://github.com/k2-fsa/OmniVoice), [VoxCPM](https://github.com/OpenBMB/VoxCPM). License field is not stated in the standard MIT/Apache form in the upstream READMEs — verify before deploying anywhere production-adjacent.

For the models in the [Considered but skipped](#considered-but-skipped) section: Fish Audio S2 is research-license non-commercial, Fish Audio S1-mini is CC-BY-NC-SA-4.0, Orpheus TTS is Apache 2.0, CosyVoice 3 see upstream — all explicitly outside this bench but listed there so the reasoning is documented.

---

## Why this exists

The published latency numbers for small open TTS models are usually:
1. Run on different hardware than yours
2. Quote a single TTFA that conflates cold and warm
3. From the model's own README (vendor-favorable)

This bench fixes (1) by running on whatever hardware you put it on, (2) by reporting cold and warm separately, and (3) by being reproducible from a clean machine in <15 minutes. It already disproved one vendor claim during its first pass — NeuTTS Air's "2× realtime on AMD Ryzen 9" turned into 0.9× RTF on x86 Windows CPU.

If you're picking a small TTS model for a real-time agent, run it on the hardware that agent will actually run on. This is the harness for doing that.
