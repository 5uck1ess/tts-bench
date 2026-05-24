# tts-bench

Personal speed bench for small/local TTS models. Cold + warm runs, same prompts, real wall clock — across CPU, CUDA, and Apple Silicon (MPS).

Built to answer one question: *which open TTS model do I plug into an always-on voice agent on the machine I actually have?*

---

## Test hardware

Listed because TTS speed is hardware-dependent — RTF claims that hold on a Ryzen 9 won't necessarily hold on a Raspberry Pi.

| Machine | OS | CPU | RAM | GPU | Used for |
|---|---|---|---|---|---|
| **Windows desktop** | Windows 11 Pro | AMD Ryzen 9 9950X3D (16C / 32T @ 4.3 GHz base) | 128 GB | NVIDIA RTX 5090 | All current bench rows below (CPU mode; GPU runs pending) |
| **Mac (pending)** | macOS — | Apple M4 Pro | — | Apple M4 Pro GPU (MPS) | Future MPS bench rows |

If you reproduce on different hardware, your numbers will differ — file an issue or PR with your results and we'll add a column.

## Current results (Windows CPU, May 2026)

Two tiers measured on the Ryzen 9 9950X3D above. Numbers shown are from short prompts; long prompts scale RTF linearly.

### Predefined-voice models (pick from a baked-in voice list)

| Model | Size / License | TTFA cold | TTFA warm | RTF warm | Languages | Notes |
|---|---|---|---|---|---|---|
| **[Piper](https://github.com/OHF-voice/piper1-gpl)** (OHF-voice, formerly rhasspy) | per-voice ~25MB / MIT | **72ms** | **39ms** | **47×** | 40+ via separate voice models | leader on this hardware; streaming-native, bundles espeak-ng (no Windows wheel pain) |
| **[Kokoro-82M](https://github.com/hexgrad/kokoro)** (hexgrad) | 82M / Apache 2.0 | 335ms | 245ms | 13× | 9 (a/b/e/f/h/i/j/p/z codes) | 54 voices; misaki tokenizer needs spaCy preinstall (see Known issues) |
| **[KittenTTS](https://github.com/KittenML/KittenTTS)** (KittenML) | <100M / Apache 2.0 | 516ms | 487ms | 6.6× | EN only | 8 voices; non-streaming so TTFA == gen_s |
| **[VibeVoice-Realtime-0.5B](https://github.com/vibevoice-community/VibeVoice)** (Microsoft, community fork) | 0.5B / MIT | ~3.9s | ~3.7s | **~0.5×** | EN only (7 preset voices) | streaming-class but heavy diffusion; DDPM steps tunable (5 default). Predefined `.pt` voice embeddings auto-downloaded |
| [Magpie-TTS Multilingual 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) (NVIDIA, NeMo) | 357M / NVIDIA Open Model License | pending | pending | pending (CUDA: ~1.0× cold smoke) | 9 (en/es/de/it/vi/zh/fr/hi/ja) | fixed speaker embeddings (this checkpoint variant); HF accept-terms gated; install skips `[tts]` extra to avoid `pynini` on Windows — runner forces `apply_TN=False` to compensate |

### Zero-shot voice cloning models (accept a reference wav at inference time)

| Model | Size / License | TTFA cold | TTFA warm | RTF warm | Cloning ref | Notes |
|---|---|---|---|---|---|---|
| **[Pocket-TTS](https://github.com/kyutai-labs/pocket-tts)** (Kyutai, predefined mode) | 100M / MIT | 95-150ms | 97-150ms | **2.9-3.0×** | wav or voice name | 26 voices unauth; BYO-voice path is HF accept-terms gated on `kyutai/pocket-tts` |
| [NeuTTS Nano](https://github.com/neuphonic/neutts) (GGUF Q4) | 748M / Apache 2.0 | 1.2s | 0.43-0.51s | 1.3-1.4× | wav + transcript | multilingual fallback; separate `.gguf` per language |
| [NeuTTS Air](https://github.com/neuphonic/neutts) (GGUF Q4) | 748M / Apache 2.0 | 1.7-1.9s | 0.67-0.70s | 0.88-0.90× | wav + transcript | below realtime on CPU — needs GPU |
| [ChatterBox-TTS](https://github.com/resemble-ai/chatterbox) (Resemble AI) | ~1.2B / MIT | ~8s | ~8s | **~0.30×** | wav (no transcript) | 1000 diffusion steps — GPU-targeted, community quality leader |
| [F5-TTS](https://github.com/SWivid/F5-TTS) (v1 Base) | ~330M / MIT | ~48s | ~48s | **~0.05×** | wav + transcript | flow matching, very slow on CPU; needs GPU |
| [Coqui XTTS-v2](https://github.com/idiap/coqui-ai-TTS) (idiap fork) | ~750M / CPML 1.0 (non-commercial) | pending | pending | pending | wav (no transcript) | de facto multilingual cloning baseline; ~2GB download on first use; auto-accepts CPML via `COQUI_TOS_AGREED=1` |
| [OmniVoice](https://github.com/k2-fsa/OmniVoice) (k2-fsa) | TBD / see upstream | pending | pending | pending | wav + transcript | 600+ languages; diffusion-LM, vendor-claimed 0.025× RTF (GPU); voice design tags (gender/age/whisper) |
| [VoxCPM-0.5B](https://github.com/OpenBMB/VoxCPM) (OpenBMB) | 0.5B / see upstream | pending | pending | pending | wav (no transcript) | tokenizer-free TTS; multilingual; 0.5B variant in this bench (the larger 2B `VoxCPM2` uses a server API and is skipped) |
| [LuxTTS](https://github.com/ysharma3501/LuxTTS) (k2-fsa-based) | — | — | — | — | wav | install blocked on Windows (see [Known issues](#known-issues)) |

**Reading the tables:** TTFA = milliseconds until the first audio sample. RTF = `audio_seconds / generation_seconds` (1.0× = realtime, higher = faster than realtime). Non-streaming models (KittenTTS, ChatterBox, F5-TTS) emit full audio in one call so TTFA = gen_s by definition.

**Top-line takeaway:** if you don't need voice cloning, **Piper wins by a huge margin on this CPU** (39ms warm TTFA, 47× RTF). Pocket-TTS is the fastest cloning-capable option (with the HF accept-terms caveat). NeuTTS Air/Nano give clean BYO-voice without auth gates but at lower RTF. ChatterBox + F5-TTS are GPU-class — file them under "bench-cold but not deployable" until 5090 runs land.

Raw CSV + WAVs from the latest run live in `results/`.

Caveats: one machine, one run. Re-bench on your own hardware before committing — see [Known issues](#known-issues) for examples of model README claims that didn't survive contact with a real install.

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

- **LuxTTS — blocked on Windows.** Depends on `piper-phonemize` which only ships manylinux x86_64/aarch64 and macOS wheels. Workaround: WSL2 (untested) or run on Mac/Linux. (Note: `piper-tts` 1.4+ replaced this dependency with bundled espeak-ng, so Piper proper works on Windows.)
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

## Roadmap

Done in this round (May 23, 2026):
- ✓ Kokoro, KittenTTS, Piper (predefined-voice tier)
- ✓ ChatterBox, F5-TTS (extra cloning models)
- ✓ VibeVoice-Realtime-0.5B (predefined-voice tier, via the community fork)
- ✓ Coqui XTTS-v2 (idiap fork) — multilingual cloning baseline
- ✓ OmniVoice (k2-fsa) — 600+ languages, diffusion-LM cloning
- ✓ VoxCPM-0.5B (OpenBMB) — tokenizer-free multilingual cloning
- ✓ Magpie-TTS Multilingual 357M (NVIDIA NeMo) — 9-lang predefined-voice; installed without the `[tts]` extra to sidestep `pynini` on Windows (runner forces `apply_TN=False`)
- ✓ `can_clone` column in `results.csv` so cloning vs predefined is one-dimensional
- ✓ `harness.py` extracted — shared model registry + subprocess plumbing
- ✓ `compare.py` added — one-shot A/B listening tool across all installed models × devices, with audio playback
- ✓ CUDA 12.8 torch wheels installed in GPU-targeted venvs (Blackwell sm_120 floor)

Pending:

- **Mac M4 Pro pass** — `install.sh` + MPS device detection are wired up; bench pending hardware.
- **RTX 5090 pass** — formal `bench.py --device cuda` run for all GPU-capable models. CUDA torch is now installed; results pending.
- **Coqui XTTS-v2 numbers** — venv install in `install.ps1` / `install.sh`; bench numbers pending first run.
- **OmniVoice / VoxCPM / Magpie numbers** — venvs added, runner wiring in place, bench numbers pending first run.
- **LuxTTS on macOS** — should install cleanly per upstream (piper-phonemize macOS wheels exist).

## Considered but skipped

Models that were evaluated for inclusion and intentionally left out, with the reason. The bar is in `harness.py`'s scope: must install cleanly cross-platform in a self-contained venv, run end-to-end on CPU at all (even if slow), and expose a Python API that fits the runner protocol.

- **[Fish Audio S2 / S2-Pro](https://github.com/fishaudio/fish-speech)** (current `main` branch, 4B params). Linux/WSL only per official docs (Windows native unsupported); 24GB VRAM floor; no clean Python API — inference is a 3-stage CLI pipeline (DAC encode → text2semantic → DAC decode) with intermediate `.npy` files; research-license non-commercial. Doesn't fit the harness pattern. Run via the upstream SGLang/vLLM serving setup if you want it.
- **[Fish Audio S1-mini](https://huggingface.co/fishaudio/s1-mini)** (0.5B distilled S1). Small enough to fit in principle, but the S1 inference code lives at a specific mid-2025 *commit* (no tag) — pre-S2 branch's `v1.5.1` doesn't pair (different `firefly-gan-vq-*` filenames), and `v2.0.0-beta` is S2-only. Pinning to commit `781bf1cd` works, but pulls a heavy dep tree (lightning, wandb, gradio, faster_whisper, modelscope, funasr, `pyaudio`) and still needs the same 3-stage CLI subprocess wrapper. CC-BY-NC-SA license. Worth revisiting if the user wants the model specifically — the install path just doesn't pay for itself in a "fast comprehensive bench" context.
- **[Orpheus TTS](https://github.com/canopyai/Orpheus-TTS)** (Canopy Labs, 3B Llama-based). High-quality emotional/empathetic speech with ~200ms streaming latency, but the inference path is `pip install orpheus-speech` which depends on **vllm**. vllm has no first-party Windows wheel; community Windows wheels exist (SystemPanic, devnen) but the Blackwell-compatible one (5090 / sm_120) is pinned to vllm 0.20.0 + cu132 while Orpheus pins vllm 0.7.3 — version mismatch is unrecoverable without rebuilding from source. Linux/WSL or Mac (via LM Studio + GGUF) work, but this is a Windows-primary bench so Orpheus is out of scope here. Revisit on the Mac M4 Pro pass.
- **[CosyVoice 3](https://github.com/FunAudioLLM/CosyVoice)** (FunAudioLLM, Fun-CosyVoice3-0.5B-2512). High-quality multilingual cloning + 18 Chinese dialects, but installs as a source clone (no pip wheel) with a researcher-pinned `requirements.txt`: `torch==2.3.1` from a cu121 extra-index (incompatible with Blackwell sm_120 / 5090 — would need a post-install swap), `onnxruntime==1.18.0` pinned in a way uv resolves against a Linux-only CUDA wheel feed first, plus 40+ other exact-version pins (`gradio==5.4.0`, `lightning==2.2.4`, `openai-whisper==20231117`, `transformers==4.51.3`, `tensorrt-cu12` Linux-only). Same family of issue as fish-speech and Orpheus: not a clean cross-platform install. Doable with significant install-archaeology effort — revisit if/when the upstream ships a Windows-friendly requirements set.

---

## Layout

```
tts-bench/
├── harness.py            # shared model registry + subprocess plumbing (imported by all 3 tools)
├── bench.py              # formal benchmark — CSV + per-prompt summary
├── compare.py            # one-shot A/B listening tool — every model × every device, plays out loud
├── speak.py              # interactive REPL — feel warm-run latency for one model
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
│   └── magpie_runner.py
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
