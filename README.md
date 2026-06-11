<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-flat-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-flat-light.svg">
    <img alt="tts-bench" src="assets/logo-flat-dark.svg" width="520">
  </picture>
</p>

Bench for local text-to-speech (TTS) models. Three lenses, on whatever hardware you put it on:

- **Speed** — cold + warm **TTFA** (time to first audio), **RTF** (real-time factor; higher = faster than realtime), memory, on CPU / CUDA / Apple Silicon
- **Listen** — every model on every prompt, default voice + voice cloning, with inline audio players, so you can pick a model by ear
- **Scores** — objective metrics per model: UTMOS (naturalness), WER (intelligibility), SIM (cloning fidelity), scored over the bench prompts via [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval)-style ASR + speaker-verification. Sortable, with a Default/Cloning toggle.

An objective quality score (NAQ) was prototyped but isn't part of the bench — the v2 features didn't track subjective ranking closely enough to publish, so it was pulled and is being redesigned separately. The bench measures speed; quality is by-ear via the Listen lens.

---

## ▶ Demos

**[5uck1ess.github.io/tts-bench](https://5uck1ess.github.io/tts-bench/)** — listen to every model, no install. Three lenses:

- **Listen** — one consolidated gallery with an inline `<audio>` player for every model on every prompt, in **default voice** and **voice cloning** (each clone sits next to the reference it's imitating). Browse **by prompt** (compare all models on one sentence) or **by model** (audition one model across prompts); only one clip plays at a time. Audio is rig-independent, so each sample is sourced once from the highest-fidelity rig and tagged with where it came from. Quality, prosody, and artifacts are obvious in 5 seconds — benchmark tables can't show that.
- **Speed** — per-rig leaderboards (Ryzen 9 9950X3D + RTX 5090, Apple M4, Ryzen + RTX 3090) with cold/warm TTFA, RTF, and memory, sortable. Pick the box you actually own.
- **Scores** — objective metrics per model (UTMOS naturalness, WER intelligibility, SIM cloning fidelity), sortable, with a Default/Cloning toggle. Human votes remain the preference ground truth; these are objective backstops.

Full per-rig reports (every model × prompt × device, plus by-prompt samples) are linked from the **Archive**.

---

## 🗳 Vote

Quality is subjective, so the ground truth is **your ears**. The companion **[TTS Voting Arena](https://5uck1ess-tts-arena.hf.space)** is a public, blind A/B listening test — two clips, no model names shown, pick the one that sounds better. No login, ~5 seconds a vote.

- **Default voice** — which model sounds more natural?
- **Cloning** — which clone better matches the reference voice?

Votes feed a **live human-preference Elo leaderboard** right there on the arena. This is where the "best sounding" and cloning calls above come from — every vote sharpens the ranking.

**[→ Vote now at the TTS Arena](https://5uck1ess-tts-arena.hf.space)**

---

## Quick start

Requires [`uv`](https://github.com/astral-sh/uv) and Python 3.11. ~10-15 min install. Disk for the full set is large: **~39 GB of per-model venvs** in the repo, plus **~125 GB of model weights** downloaded to your Hugging Face cache (`~/.cache/huggingface`, **not** the repo) — **~165 GB all-in**. Individual models are far smaller, so installing a subset costs a fraction of that.

```powershell
# Windows — everything, or just the models you want
.\install.ps1
.\install.ps1 kokoro,piper,miso
python bench.py
```

```bash
# macOS / Linux — everything, or just the models you want
./install.sh
./install.sh kokoro piper miso
python bench.py
```

Pass model names to install only those (names = the `venvs/<name>` slugs, which match the tables below — lowercase, e.g. `kokoro`, `f5tts`, `chatterbox`, `miso`). A few share one install: `neutts` covers NeuTTS Air + Nano, `chatterbox` both ChatterBox variants, `vibevoice` the 0.5B/1.5B, `moss_tts` both MOSS checkpoints, `fish` is Fish Speech 1.5. Add `scoring` (plus `scoring_sim` on Linux) for the objective-metrics venv. `bench.py` only runs models whose venv exists, so a partial install benches cleanly — install more models later by re-running with new names.

Interactive feel-test: `python speak.py kokoro`. One-shot A/B comparison: `python compare.py "your phrase"`. See [docs/architecture.md](docs/architecture.md) for the runner protocol and how to add a model.

---

## TLDR (June 2026)

**Fastest:**
- CPU (Ryzen 9 9950X3D, Windows): **Piper** — 107ms warm TTFA, 59× RTF
- CUDA (RTX 5090): **Kokoro** — 67ms warm TTFA, 104× RTF
- CPU + MPS (Apple M4, 16 GB): **Piper** — 208ms warm TTFA, 32× RTF

**Best sounding:** *No objective ranking right now — the NAQ score is paused pending redesign. Open the [Demos site](https://5uck1ess.github.io/tts-bench/) and use the Listen lens.*

**Best cloning — blind A/B votes (these measure voice-match *preference*, not intelligibility):**
- 1. **OmniVoice** — top on voice/accent match (24-1-3), **but it can garble or drop words**; a timbre-focused A/B vote doesn't penalize that, so read this as "best voice match," not "best overall clone." Audition it first — objective **WER** (the new Scores lens) is meant to catch exactly this gap.
- 2. **Echo-TTS** — near-tied #1 (21-1-6), clean 44.1 kHz
- 3. **IndexTTS-2** — third (16-2-5), accent held

[→ full per-rig results](docs/results.md) · [→ full cloning ranking](docs/cloning.md)

---

## Models tracked (47)

#### Predefined voices

| Model | Params | Predefined | Cloning | Multilingual | SR | Expressive | License |
|---|---|---|---|---|---|---|---|
| [KittenTTS Nano 0.1](https://huggingface.co/KittenML/kitten-tts-nano-0.1) | <100M | ✓ | — | — | 24k | — | Apache 2.0 |
| [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) | 82M | ✓ | — | ✓ | 24k | — | Apache 2.0 |
| [LuxTTS](https://github.com/ysharma3501/LuxTTS) | 123M | ✓ | — | — | 22.05k | — | MIT |
| [Magpie-TTS](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) | 357M | ✓ | — | ✓ (9) | 22.05k | emotion voices\* | NVIDIA OML |
| [Maya1](https://huggingface.co/maya-research/maya1) | 3B | ✓ (voice desc) | — | — | 24k | tags + desc | Apache 2.0 |
| [MeloTTS](https://huggingface.co/myshell-ai/MeloTTS-English) | ~52M | ✓ | — | — (en) | **44.1k** | — | MIT |
| [OuteTTS 1.0 1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B) | ~1B | ✓ | ✓ | ✓ (12) | 44.1k | — | CC-BY-NC-SA 4.0 + Llama 3.2 |
| [Parler-TTS Mini v1](https://huggingface.co/parler-tts/parler-tts-mini-v1) | 878M | ✓ (voice desc) | — | — | **44.1k** | desc\* | Apache 2.0 |
| [Piper](https://github.com/OHF-Voice/piper1-gpl) | ~15M | ✓ | — | ✓ | 22.05k | — | GPL-3.0 |
| [Soprano 1.1 80M](https://huggingface.co/ekwek/Soprano-1.1-80M) | 80M | ✓ | — | — | 32k | — | Apache 2.0 |
| [Supertonic 3](https://huggingface.co/Supertone/supertonic-3) | 99M | ✓ | — | ✓ (31) | 24k | tags | MIT + OpenRAIL-M |
| [VibeVoice Realtime 0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) | 0.5B | ✓ | — | — | 24k | — | MIT |
| [Voxtral 4B TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) | 4B | ✓ (20) | ✓ | ✓ | 24k | — | CC-BY-NC 4.0 |

#### Zero-shot cloning

| Model | Params | Predefined | Cloning | Multilingual | SR | Expressive | License |
|---|---|---|---|---|---|---|---|
| [ChatterBox](https://huggingface.co/ResembleAI/chatterbox) | 1.2B | — | ✓ | — | 24k | knob | MIT |
| [ChatterBox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) | 744M | — | ✓ | — | 24k | —\* | MIT |
| [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2) | 750M | — | ✓ | ✓ (17) | 24k | — | CPML (non-commercial) |
| [Dia 1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626) | 1.6B | — | ✓ | — | 44.1k | tags | Apache 2.0 |
| [dots.tts (soar)](https://huggingface.co/rednote-hilab/dots.tts-soar) | 2B | — | ✓ | ✓ (24) | **48k** | — | Apache 2.0 |
| [DramaBox](https://github.com/resemble-ai/DramaBox) | 3.3B | — | ✓ | — (en) | **48k** | desc | LTX-2 Community (NC) |
| [Echo-TTS](https://huggingface.co/jordand/echo-tts-base) | ~2.8B | — | ✓ | — | **44.1k** | tags | CC-BY-NC-SA 4.0 |
| [F5-TTS v1](https://huggingface.co/SWivid/F5-TTS) | 330M | — | ✓ | ✓ | 24k | — | CC-BY-NC |
| [Fish Speech 1.5](https://huggingface.co/fishaudio/fish-speech-1.5) | ~500M | — | ✓ | ✓ | **44.1k** | — | CC-BY-NC-SA 4.0 |
| [Fish Speech S2-Pro](https://huggingface.co/fishaudio/s2-pro) | 4B | — | ✓ | — | **44.1k** | tags | Research (non-commercial) |
| [Higgs Audio v3 TTS](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) | 4B | — | ✓ | ✓ (100) | 24k | tags | Research (NC) |
| [IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) | 1.5B | — | ✓ | ✓ | 24k | emo-ref + desc + knob | Apache 2.0 |
| [Mars5-TTS](https://huggingface.co/Camb-ai/mars5-tts) | 1.2B | — | ✓ | — | 24k | — | AGPL-3.0 |
| [MetaVoice-1B](https://huggingface.co/metavoiceio/metavoice-1B-v0.1) | 1.2B | — | ✓ | — | **48k** | — | Apache 2.0 |
| [MiraTTS](https://huggingface.co/YatharthS/MiraTTS) | 0.5B | — | ✓ | — | **48k** | knob | MIT |
| [Miso TTS 8B](https://huggingface.co/MisoLabs/MisoTTS) | 8.2B | — | ✓ | — (en) | 24k | — | Modified MIT |
| [MOSS-TTS v1.0](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) | 8B (Qwen3) | — | ✓ | ✓ (20) | 24k | — | Apache 2.0 |
| [MOSS-TTS v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) | 8B (Qwen3) | — | ✓ | ✓ (31) | 24k | tags (pause) | Apache 2.0 |
| [MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano) | 100M | — | ✓ | ✓ (zh+en) | **48k** | — | Apache 2.0 |
| [NeuTTS Air](https://huggingface.co/neuphonic/neutts-air) | 748M | — | ✓ | — | 24k | — | Apache 2.0 |
| [NeuTTS Nano](https://huggingface.co/neuphonic/neutts-nano-q4-gguf) | 229M | — | ✓ | — | 24k | — | Apache 2.0 |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | ~1B | — | ✓ | ✓ (600+) | 24k | tags\* | Apache 2.0 |
| [OpenVoice v2](https://huggingface.co/myshell-ai/OpenVoiceV2) | ~100M | — | ✓ | ✓ | 22.05k | knob | MIT |
| [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts) | 100M | — | ✓ | — | 24k | — | Apache 2.0 |
| [Qwen3-TTS 1.7B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | 1.7B | — | ✓ | ✓ | 24k | — | Apache 2.0 |
| [Qwen3-TTS 1.7B (CUDA-graph)](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | 1.7B | — | ✓ | ✓ | 24k | — | MIT |
| [Sesame CSM-1B](https://huggingface.co/sesame/csm-1b) | 1B | — | ✓ | — | 24k | — | Apache 2.0 |
| [Step-Audio-EditX](https://huggingface.co/stepfun-ai/Step-Audio-EditX) | 3B | — | ✓ | — | 24k | tags + desc | Apache 2.0 |
| [StyleTTS 2](https://github.com/yl4579/StyleTTS2) | ~148M | — | ✓ | — | 24k | knob | MIT |
| [VibeVoice 1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) | 1.5B | — | ✓ | — | 24k | — | MIT |
| [VibeVoice 7B](https://huggingface.co/vibevoice/VibeVoice-7B) | 7B | — | ✓ | — | 24k | — | MIT |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | 2B | — | ✓ | ✓ (30) | **48k** | desc | Apache 2.0 |
| [ZipVoice](https://huggingface.co/k2-fsa/ZipVoice) | 123M | — | ✓ | ✓ (zh+en) | 24k | — | Apache 2.0 |
| [Zonos v0.1](https://huggingface.co/Zyphra/Zonos-v0.1-transformer) | 1.6B | — | ✓ | ✓ | **44.1k** | emo-ref + knob | Apache 2.0 |

> **Expressive column** — what explicit emotion/delivery control the model offers: `tags` = inline cues in the text itself (`(laughs)`, `[sigh]`, `<laugh>`); `desc` = natural-language style/emotion instructions; `knob` = numeric or preset parameter (exaggeration, style enum, pitch/speed); `emo-ref` = emotion conditioned on a separate reference clip or emotion vector; `—` = none (for cloning models, expression simply follows the reference clip). `*` = caveat applies. Exact syntax, sources, and caveats per model: **[docs/expressive-control.md](docs/expressive-control.md)**. Note the bench feeds every model the same plain prompts for fairness, so these features are not exercised in any score.

Full per-model gotchas + license details: **[docs/known-issues.md](docs/known-issues.md)**. Models considered but excluded: **[docs/considered.md](docs/considered.md)**.

> **Predefined vs Cloning.** *Predefined* models have fixed/selectable speaker voices baked into the weights — they speak with no reference needed. *Cloning* (zero-shot) models have **no voice of their own**: they synthesize whatever voice you hand them as a reference clip at inference. Given no reference, a pure zero-shot model falls back to a bundled sample (this bench uses `chris_hemsworth_15s.wav`), so its "default voice" is just a clone of that clip. A few models do both (e.g. Voxtral has 20 presets *and* cloning).

> Rig availability: Voxtral is Mac (MLX, preset-voice only) + Linux (vLLM, cloning); Fish S2-Pro / MetaVoice / Step-Audio-EditX / Higgs Audio v3 / dots.tts are Linux-only (CUDA) — Higgs v3 is the one **server-backed** model (it runs via a Docker `sgl-omni` HTTP server, not an in-process load), and dots.tts is Linux-only because its `WeTextProcessing`→`pynini` dependency won't build under Windows MSVC; Echo-TTS and DramaBox are Windows + Linux (CUDA-only, no CPU/MPS; DramaBox needs ~18 GB VRAM). The rest run on Windows + Linux CUDA, most on CPU/MPS too. Per-rig speed + samples on the [Demos site](https://5uck1ess.github.io/tts-bench/).

---

## Voice cloning

**35 of the 47 tracked models can clone** a voice from a reference clip. Three reference formats supported (wav only / wav + transcript / HF-gated wav). Drop a reference into `reference/`, then `python bench.py --reference reference/myvoice.wav`.

Reference-format docs + the blind-vote cloning ranking (28 of 32 cloning models, 397 votes, human-preference A/B): **[docs/cloning.md](docs/cloning.md)**.

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
- [Cloning ranking](docs/cloning.md) — reference formats + blind-vote ranking (28 of 32 cloning models, human-preference A/B)
- [Architecture](docs/architecture.md) — bench design, runner protocol, adding a model
- [Expressive control](docs/expressive-control.md) — which models take emotion tags / style prompts / knobs, with exact syntax
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
