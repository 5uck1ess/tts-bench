# tts-bench

Bench for local TTS models. Cold + warm runs, same prompts, real wall clock — across CPU, CUDA, and Apple Silicon (MPS). Speed + objective quality (via NAQ) + memory + multilingual coverage, all measured the same way on every machine.

Built to answer one question: *which open TTS model do I plug into an always-on voice agent on the machine I actually have?*

---

## ▶ Demos

**[5uck1ess.github.io/tts-bench](https://5uck1ess.github.io/tts-bench/)** — public side-by-side audio.

Every model × prompt × device combination is rendered with an inline `<audio>` player so you can hear the actual output without cloning the repo or running anything locally. Useful for:

- *Picking a model.* Listen to the same prompt across 18 TTS models on the same hardware. Quality, prosody, and artifacts are obvious in 5 seconds; benchmark tables can't show that.
- *Comparing rigs.* Each report is tagged with the rig (Ryzen 9 9950X3D, Apple M4, etc.) and labeled (default voice vs cloning) so you can see how the same model sounds on the box you actually own.
- *Comparing devices for one model.* CPU vs CUDA vs MPS rows for the same model, side by side, with their audio.

---

## Quick start

Requires [`uv`](https://github.com/astral-sh/uv) and Python 3.11. ~10-15 min install, ~60 GB disk for all models.

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

## TLDR results (May 2026)

**CPU — Ryzen 9 9950X3D (Windows):**
- **Piper** wins predefined voices — 39ms warm TTFA, 47× RTF. Drop-in for an always-on agent.
- **Pocket-TTS** wins cloning-capable on default voices — 100ms warm TTFA.

**CPU + MPS — Apple M4 (Mac, 16 GB):**
- **Piper** wins again — 202ms warm TTFA, 33× RTF.
- **Pocket-TTS** best cloning-capable — 42ms warm TTFA (M4 single-thread beats Ryzen 9).

**CUDA — RTX 5090 (Windows):**
- **Kokoro** wins predefined — 69ms warm TTFA, **101× RTF**, 0.7 GB VRAM.
- **OmniVoice** wins cloning — 869ms warm TTFA, **9.2× warm RTF**, 2.4 GB VRAM; scales to 20× RTF on long prompts.

Full per-model, per-prompt tables: **[docs/results.md](docs/results.md)**.

---

## Quality scoring: NAQ

**Naturalness-Artifact Quotient** — a 0-100 objective per-wav score combining HARM (harmonic-to-noise ratio, weighted 0.65) and BUZZ (inverse 4-8 kHz spectral flatness, weighted 0.35). Higher = more natural; lower = more roboty / vocoder-artifacted. Computed automatically for the cold run of every bench cell and shown next to the speed numbers in every report.

Sub-scores (`naq_harm`, `naq_buzz`) are exposed in the CSV and as a tooltip in the HTML report so you can see *why* a model scored where it did.

Full spec, formula, and calibration: **[docs/naq.md](docs/naq.md)**.

---

## Voice cloning

Three reference formats supported (wav only / wav + transcript / HF-gated wav). Drop a reference into `reference/`, then `python bench.py --reference reference/myvoice.wav`.

**Cloning quality ranking (May 2026):**

1. **OmniVoice** — accent preserved, top of listening test
2. **ChatterBox** — strong second, clean output
3. **IndexTTS-2** — also good, accent preserved

Full 10-model ranking + ref format docs: **[docs/cloning.md](docs/cloning.md)**.

---

## Test hardware

| Machine | Used for |
|---|---|
| Windows desktop (Ryzen 9 9950X3D / 128 GB / RTX 5090 32 GB) | Windows CPU + CUDA bench rows |
| Mac (Apple M4 / 16 GB / M4 GPU) | Mac CPU + MPS bench rows |

If you reproduce on different hardware, file an issue or PR with your results and we'll add a column.

---

## Docs

- [Full results tables](docs/results.md) — per-rig, per-prompt, per-model
- [NAQ score spec](docs/naq.md) — what the quality score is and how it's computed
- [Cloning ranking](docs/cloning.md) — 10-model subjective ranking, reference format docs
- [Architecture](docs/architecture.md) — bench design, runner protocol, adding a model
- [Known issues](docs/known-issues.md) — per-model gotchas + per-license table
- [Considered but skipped](docs/considered.md) — models evaluated and excluded

---

## Pending work

- **Subjective listening pass on the predefined-voice tier (CUDA).** Cloning got the full ranking; predefined didn't.
- **MARS5 CUDA investigation** — 0.1× RTF + cloning that doesn't match the reference. Both unusable. Needs deeper look or "skipped after investigation" doc.
- **Qwen3-TTS Base cloning timeout on long prompts** — at the 15s Chris Hemsworth ref, prompts 2-5 hit the 10-min per-cell wall.
- **ChatterBox / F5-TTS / Coqui XTTS on Mac MPS** — skipped earlier (all GPU-class). Worth re-running once MPS torch perf improves.
- **LuxTTS on Mac arm64 / Windows** — depends on piper-phonemize, no wheels for those platforms.

---

## License

MIT for the bench code in this repo. **Each TTS model has its own license** — see [docs/known-issues.md](docs/known-issues.md) for the full per-model table.

---

## Why this exists

The published latency numbers for small open TTS models are usually:
1. Run on different hardware than yours
2. Quote a single TTFA that conflates cold and warm
3. From the model's own README (vendor-favorable)

This bench fixes (1) by running on whatever hardware you put it on, (2) by reporting cold and warm separately, and (3) by being reproducible from a clean machine in <15 minutes. It already disproved one vendor claim during its first pass — NeuTTS Air's "2× realtime on AMD Ryzen 9" turned into 0.9× RTF on x86 Windows CPU.

If you're picking a small TTS model for a real-time agent, run it on the hardware that agent will actually run on. This is the harness for doing that.
