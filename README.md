<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-flat-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-flat-light.svg">
    <img alt="tts-bench" src="assets/logo-flat-dark.svg" width="520">
  </picture>
</p>

Bench for local TTS models. Two lenses, on whatever hardware you put it on:

- **Speed** — cold + warm **TTFA** (time to first audio), **RTF** (real-time factor; higher = faster than realtime), memory, on CPU / CUDA / Apple Silicon
- **Listen** — every model on every prompt, default voice + voice cloning, with inline audio players, so you can pick a model by ear

An objective quality score (NAQ) was prototyped and is currently paused for redesign — the v2 features didn't track subjective ranking closely enough to publish. The bench still computes it into the CSV; the HTML report omits it until a refit lands.

---

## ▶ Demos

**[5uck1ess.github.io/tts-bench](https://5uck1ess.github.io/tts-bench/)** — listen to every model, no install. Two lenses:

- **Listen** — one consolidated gallery with an inline `<audio>` player for every model on every prompt, in **default voice** and **voice cloning** (each clone sits next to the reference it's imitating). Browse **by prompt** (compare all models on one sentence) or **by model** (audition one model across prompts); only one clip plays at a time. Audio is rig-independent, so each sample is sourced once from the highest-fidelity rig and tagged with where it came from. Quality, prosody, and artifacts are obvious in 5 seconds — benchmark tables can't show that.
- **Speed** — per-rig leaderboards (Ryzen 9 9950X3D + RTX 5090, Apple M4, Ryzen + RTX 3090) with cold/warm TTFA, RTF, and memory, sortable. Pick the box you actually own.

Full per-rig reports (every model × prompt × device, plus by-prompt samples) are linked from the **Archive**.

---

## Quick start

Requires [`uv`](https://github.com/astral-sh/uv) and Python 3.11. ~10-15 min install. Disk for the full set is large: **~39 GB of per-model venvs** in the repo, plus **~125 GB of model weights** downloaded to your Hugging Face cache (`~/.cache/huggingface`, **not** the repo) — **~165 GB all-in**. Individual models are far smaller, so installing a subset costs a fraction of that.

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

Interactive feel-test: `python speak.py kokoro`. One-shot A/B comparison: `python compare.py "your phrase"`. See [docs/architecture.md](docs/architecture.md) for the runner protocol and how to add a model.

---

## TLDR (May 2026)

**Fastest:**
- CPU (Ryzen 9 9950X3D, Windows): **Piper** — 39ms warm TTFA, 47× RTF
- CUDA (RTX 5090): **Kokoro** — 69ms warm TTFA, 101× RTF
- CPU + MPS (Apple M4, 16 GB): **Piper** — 62ms warm TTFA, 33× RTF

**Best sounding:** *No objective ranking right now — the NAQ score is paused pending redesign. Open the [Demos site](https://5uck1ess.github.io/tts-bench/) and use the Listen lens.*

**Best cloning (subjective rank):**
- 1. **OmniVoice** — accent preserved, top of listening test
- 2. **ChatterBox** — strong second, clean output
- 3. **IndexTTS-2** — also good, accent preserved

[→ full per-rig results](docs/results.md) · [→ full cloning ranking](docs/cloning.md)

---

## Models tracked (40)

#### Predefined voices

| Model | Params | Predefined | Cloning | Multilingual | SR | License |
|---|---|---|---|---|---|---|
| KittenTTS | <100M | ✓ | — | — | 24k | Apache 2.0 |
| Kokoro | 82M | ✓ | — | ✓ | 24k | Apache 2.0 |
| LuxTTS | 123M | ✓ | — | — | 22.05k | MIT |
| Magpie-TTS | 357M | ✓ | — | ✓ (9) | 22.05k | NVIDIA OML |
| Maya1 | 3B | ✓ (voice desc) | — | — | 24k | Apache 2.0 |
| OuteTTS 1.0 1B | ~1B | ✓ | ✓ | ✓ (12) | 44.1k | CC-BY-NC-SA 4.0 + Llama 3.2 |
| Piper | ~15M | ✓ | — | ✓ | 22.05k | MIT |
| Soprano 80M | 80M | ✓ | — | — | 32k | Apache 2.0 |
| Supertonic | 99M | ✓ | — | ✓ (31) | 24k | MIT + OpenRAIL-M |
| VibeVoice Realtime 0.5B | 0.5B | ✓ | — | — | 24k | MIT |
| Voxtral 4B TTS | 4B | ✓ (20) | ✓ | ✓ | 24k | CC-BY-NC 4.0 |

#### Zero-shot cloning

| Model | Params | Predefined | Cloning | Multilingual | SR | License |
|---|---|---|---|---|---|---|
| ChatterBox | 1.2B | — | ✓ | — | 24k | MIT |
| ChatterBox Turbo | 744M | — | ✓ | — | 24k | MIT |
| Coqui XTTS-v2 | 750M | — | ✓ | ✓ (17) | 24k | CPML (non-commercial) |
| Dia 1.6B | 1.6B | — | ✓ | — | 44.1k | Apache 2.0 |
| Echo-TTS | ~2.8B | — | ✓ | — | **44.1k** | CC-BY-NC-SA 4.0 |
| F5-TTS | 330M | — | ✓ | ✓ | 24k | CC-BY-NC |
| Fish Speech 1.5 | ~500M | — | ✓ | ✓ | **44.1k** | CC-BY-NC-SA 4.0 |
| Fish Speech S2-Pro | 4B | — | ✓ | — | **44.1k** | Research (non-commercial) |
| IndexTTS-2 | 1.5B | — | ✓ | ✓ | 24k | Apache 2.0 |
| Mars5-TTS | 1.2B | — | ✓ | — | 24k | AGPL-3.0 |
| MetaVoice-1B | 1.2B | — | ✓ | — | **48k** | Apache 2.0 |
| MiraTTS | 0.5B | — | ✓ | — | **48k** | MIT |
| MOSS-TTS | 8B (Qwen3) | — | ✓ | ✓ (20) | 24k | Apache 2.0 |
| MOSS-TTS-Nano | 100M | — | ✓ | ✓ (zh+en) | **48k** | Apache 2.0 |
| NeuTTS Air | 748M | — | ✓ | — | 24k | Apache 2.0 |
| NeuTTS Nano | 229M | — | ✓ | — | 24k | Apache 2.0 |
| OmniVoice | ~1B | — | ✓ | ✓ (600+) | 24k | Apache 2.0 |
| OpenVoice v2 | ~100M | — | ✓ | ✓ | 22.05k | MIT |
| Pocket-TTS | 100M | — | ✓ | — | 24k | Apache 2.0 |
| Qwen3-TTS Base | 1.7B | — | ✓ | ✓ | 24k | Apache 2.0 |
| Qwen3-TTS 1.7B (CUDA-graph) | 1.7B | — | ✓ | ✓ | 24k | MIT |
| Sesame CSM-1B | 1B | — | ✓ | — | 24k | Apache 2.0 |
| Step-Audio-EditX | 3B | — | ✓ | — | 24k | Apache 2.0 |
| StyleTTS 2 | ~148M | — | ✓ | — | 24k | MIT |
| VibeVoice 1.5B | 1.5B | — | ✓ | — | 24k | MIT |
| VibeVoice 7B | 7B | — | ✓ | — | 24k | MIT |
| VoxCPM2 | 2B | — | ✓ | ✓ (30) | **48k** | Apache 2.0 |
| ZipVoice | 123M | — | ✓ | ✓ (zh+en) | 24k | Apache 2.0 |
| Zonos v0.1 | 1.6B | — | ✓ | ✓ | **44.1k** | Apache 2.0 |

Full per-model gotchas + license details: **[docs/known-issues.md](docs/known-issues.md)**. Models considered but excluded: **[docs/considered.md](docs/considered.md)**.

> **Predefined vs Cloning.** *Predefined* models have fixed/selectable speaker voices baked into the weights — they speak with no reference needed. *Cloning* (zero-shot) models have **no voice of their own**: they synthesize whatever voice you hand them as a reference clip at inference. Given no reference, a pure zero-shot model falls back to a bundled sample (this bench uses `chris_hemsworth_15s.wav`), so its "default voice" is just a clone of that clip. A few models do both (e.g. Voxtral has 20 presets *and* cloning).

> Rig availability: Voxtral is Mac (MLX, preset-voice only) + Linux (vLLM, cloning); Fish S2-Pro / MetaVoice / Step-Audio-EditX are Linux-only (CUDA); Echo-TTS is Windows + Linux (CUDA-only, no CPU/MPS). The rest run on Windows + Linux CUDA, most on CPU/MPS too. Per-rig speed + samples on the [Demos site](https://5uck1ess.github.io/tts-bench/).

---

## Voice cloning

**30 of the 40 tracked models can clone** a voice from a reference clip. Three reference formats supported (wav only / wav + transcript / HF-gated wav). Drop a reference into `reference/`, then `python bench.py --reference reference/myvoice.wav`.

Reference-format docs + the subjective cloning ranking (10 models ranked by ear so far): **[docs/cloning.md](docs/cloning.md)**.

---

## Test hardware

| Machine | Used for |
|---|---|
| Windows desktop (Ryzen 9 9950X3D / 128 GB / RTX 5090 32 GB) | Windows CPU + CUDA bench rows |
| Linux workstation (Ryzen 9 5900XT / 64 GB / RTX 3090 24 GB, Ubuntu Server 24.04) | Linux CPU + CUDA; the only rig that runs Fish-Speech S2 natively |
| Mac (Apple M4 / 16 GB / M4 GPU) | Mac CPU + MPS bench rows |

If you reproduce on different hardware, file an issue or PR with your results and we'll add a column.

---

## Docs

- [Full results tables](docs/results.md) — per-rig, per-prompt, per-model
- [Cloning ranking](docs/cloning.md) — reference formats + subjective ranking (10 of 30 cloning models ranked so far)
- [Architecture](docs/architecture.md) — bench design, runner protocol, adding a model
- [Known issues](docs/known-issues.md) — per-model gotchas + per-license table
- [Considered but skipped](docs/considered.md) — models evaluated and excluded
- [Tasks & pending work](docs/tasks.md) — open issues, planned features
- [Methodology](docs/methodology.md) — what's measured, why cold + warm, why reproducible

---

## License

MIT for the bench code in this repo. **Each TTS model has its own license** — see [docs/known-issues.md](docs/known-issues.md) for the full per-model table.

---

## Support

If this bench saved you a weekend of writing your own:

<a href="https://ko-fi.com/5uck1ess" target="_blank"><img src="https://storage.ko-fi.com/cdn/kofi2.png?v=3" alt="Buy me a coffee at ko-fi.com" height="50" /></a>
