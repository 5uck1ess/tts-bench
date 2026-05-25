<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-flat-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-flat-light.svg">
    <img alt="tts-bench" src="assets/logo-flat-dark.svg" width="520">
  </picture>
</p>

Bench for local TTS models. Two lenses, on whatever hardware you put it on:

- **Speed** — cold + warm **TTFA** (time to first audio), **RTF** (real-time factor; higher = faster than realtime), memory, on CPU / CUDA / Apple Silicon
- **Samples** — every model × prompt × rig with inline audio players, so you can pick a model by listening

An objective quality score (NAQ) was prototyped and is currently paused for redesign — the v2 features didn't track subjective ranking closely enough to publish. The bench still computes it into the CSV; the HTML report omits it until a refit lands.

---

## ▶ Demos

**[5uck1ess.github.io/tts-bench](https://5uck1ess.github.io/tts-bench/)** — public side-by-side audio.

Every model × prompt × device combination is rendered with an inline `<audio>` player so you can hear the actual output without cloning the repo or running anything locally. Useful for:

- *Picking a model.* Listen to the same prompt across 18 TTS models on the same hardware. Quality, prosody, and artifacts are obvious in 5 seconds; benchmark tables can't show that.
- *Comparing rigs.* Each report is tagged with the rig (Ryzen 9 9950X3D, Apple M4, etc.) and labeled (default voice vs cloning) so you can see how the same model sounds on the box you actually own.
- *Comparing devices for one model.* CPU vs CUDA vs MPS rows for the same model, side by side, with their audio.

---

## Quick start

Requires [`uv`](https://github.com/astral-sh/uv) and Python 3.11. ~10-15 min install; ~100 GB disk for the full set (~56 GB model weights + ~47 GB of per-model venvs).

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
- CPU + MPS (Apple M4, 16 GB): **Piper** — 202ms warm TTFA, 33× RTF

**Best sounding:** *No objective ranking right now — the NAQ score is paused pending redesign. Open any [Demos report](https://5uck1ess.github.io/tts-bench/) and listen via the Samples lens.*

**Best cloning (subjective rank):**
- 1. **OmniVoice** — accent preserved, top of listening test
- 2. **ChatterBox** — strong second, clean output
- 3. **IndexTTS-2** — also good, accent preserved

[→ full per-rig results](docs/results.md) · [→ full cloning ranking](docs/cloning.md)

---

## Models tracked (27)

| Model | Params | Predefined | Cloning | Multilingual | SR | License |
|---|---|---|---|---|---|---|
| Piper | ~15M | ✓ | — | ✓ | 22.05k | MIT |
| Kokoro | 82M | ✓ | — | ✓ | 24k | Apache 2.0 |
| KittenTTS | <100M | ✓ | — | — | 24k | Apache 2.0 |
| Magpie-TTS | 357M | ✓ | — | ✓ (9) | 22.05k | NVIDIA OML |
| VibeVoice Realtime 0.5B | 0.5B | ✓ | — | — | 24k | MIT |
| VibeVoice 1.5B | 1.5B | ✓ | — | — | 24k | MIT |
| Supertonic | 99M | ✓ | — | ✓ (31) | 24k | MIT + OpenRAIL-M |
| LuxTTS | ~123M | ✓ | — | — | 22.05k | MIT |
| Soprano 80M | 80M | ✓ | — | — | 32k | Apache 2.0 |
| Pocket-TTS | 100M | — | ✓ | — | 24k | Apache 2.0 |
| ChatterBox | 1.2B | — | ✓ | — | 24k | MIT |
| ChatterBox Turbo | 744M | — | ✓ | — | 24k | MIT |
| F5-TTS | 330M | — | ✓ | ✓ | 24k | CC-BY-NC |
| IndexTTS-2 | 1.5B | — | ✓ | ✓ | 24k | Apache 2.0 |
| OmniVoice | ~1B | — | ✓ | ✓ (600+) | 24k | Apache 2.0 |
| ZipVoice | 123M | — | ✓ | ✓ (zh+en) | 24k | Apache 2.0 |
| VoxCPM2 | 2B | — | ✓ | ✓ (30) | **48k** | Apache 2.0 |
| Sesame CSM-1B | 1B | — | ✓ | — | 24k | Apache 2.0 |
| Coqui XTTS-v2 | 750M | — | ✓ | ✓ (17) | 24k | CPML (non-commercial) |
| Qwen3-TTS Base | 1.7B | — | ✓ | ✓ | 24k | Apache 2.0 |
| Qwen3-TTS 1.7B (CUDA-graph) | 1.7B | — | ✓ | ✓ | 24k | MIT |
| Mars5-TTS | 1.2B | — | ✓ | — | 24k | AGPL-3.0 |
| NeuTTS Air | 748M | — | ✓ | — | 24k | Apache 2.0 |
| NeuTTS Nano | 748M | — | ✓ | — | 24k | Apache 2.0 |
| Dia 1.6B | 1.6B | — | ✓ | — | 44.1k | Apache 2.0 |
| MOSS-TTS-Nano | 100M | — | ✓ | ✓ (zh+en) | **48k** | Apache 2.0 |
| MOSS-TTS | 8B (Qwen3) | — | ✓ | ✓ (20) | 24k | Apache 2.0 |

Full per-model gotchas + license details: **[docs/known-issues.md](docs/known-issues.md)**. Models considered but excluded: **[docs/considered.md](docs/considered.md)**.

---

## Voice cloning

Three reference formats supported (wav only / wav + transcript / HF-gated wav). Drop a reference into `reference/`, then `python bench.py --reference reference/myvoice.wav`.

Full 10-model cloning ranking + ref format docs: **[docs/cloning.md](docs/cloning.md)**.

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
- [Cloning ranking](docs/cloning.md) — 10-model subjective ranking, reference format docs
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
