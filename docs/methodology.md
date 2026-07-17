# Methodology

## Why these axes

Open TTS models advertise speed numbers from cherry-picked hardware, often confusing cold and warm starts, and never tell you what the output sounds like. tts-bench measures, on whatever hardware you point it at:

- **Speed** — cold + warm TTFA, RTFx, peak RAM, peak VRAM. Cold = first run after process start. Warm = subsequent runs.
- **Samples** — every model × prompt rendered with an inline audio player so you can pick by ear. Sourced from a single rig per model (see "Speed is per-rig; samples are single-source" below), not duplicated across rigs.
- **Voice cloning** — same prompts, same reference voice, ranked subjectively. See [cloning.md](cloning.md).

An objective quality score (NAQ) was prototyped but is **not** part of the bench — it didn't track subjective ranking closely enough to publish, and is now redesigned separately (offline, over the saved wavs) rather than computed during a run. The bench measures speed only.

## Speed is per-rig; samples are single-source

Speed and samples depend on the hardware very differently, so they're collected and published differently:

- **Speed (TTFA, RTFx, peak RAM/VRAM) is genuinely hardware-dependent.** A 5090, a 3090, and an M4 give different numbers, and that difference is the whole point. So the full bench runs on **every rig**, and the leaderboard shows all of them side by side.
- **Samples (the audio itself) are essentially hardware-independent.** The same model, prompt, and voice produce acoustically the same clip regardless of which GPU runs it — the output is a function of weights + inference code, not the device. The *only* cross-rig difference is **quantization/precision** (fp16 vs Q8 vs MLX-4bit), and even that is only sometimes audible. Publishing the same clip once per rig is therefore just duplication that makes the samples page harder to navigate, not richer.

So we publish **one sample per (model, voice mode)** — default voice and cloning are different voices, so a cloneable model has up to two — chosen by a single sourcing rule:

> **Windows-first, then Linux, then Mac.** Use the Windows-5090 (fp16) clip for every model Windows can run; fall back to Linux-3090, then Mac (Apple-Silicon/MLX), only for models Windows **can't** run.

Windows-first is, in practice, *highest-fidelity-first* — the 5090 runs most models at full precision. The only models that fall through to another rig are the ones Windows genuinely can't run:

- **Voxtral** → Mac (MLX-4bit; Windows has no vLLM wheel).
- **fish_s2 / metavoice / step_editx** → Linux (Linux-only runners).

Two rules keep the single source honest:

1. **"Can run it" means "can produce audio at all" — speed is irrelevant for sample sourcing.** A model that's slow on Windows is still a valid Windows sample; we're choosing an already-generated clip, not re-running anything for the sake of samples.
2. **Cloning samples must use the same reference voice across rigs** (`reference/chris_hemsworth_15s.wav`) so the fallback clips stay comparable to the Windows ones.

This is why a full re-bench on each rig is still expected for **speed**, while **samples** never need to be re-run anywhere they already exist on a higher-priority rig.

## Why cold vs warm

Cold TTFA is what an always-on agent experiences on its first request after a deploy or restart. Warm TTFA is what every subsequent request experiences. Vendor numbers usually quote warm — that's not what users feel during cold start.

## Why reproducible

Every install is `uv` + Python 3.11 + a per-model venv. From a clean machine to a populated `results/` directory is under 15 minutes. The bench has already disproved one vendor claim — NeuTTS Air's "2× realtime on AMD Ryzen 9" turned into 0.9× RTFx on x86 Windows CPU.

## What gets reported

Every per-rig run produces a results directory with `results.csv`, `meta.json`, the generated wavs, and three HTML pages: a landing card (`index.html`), a speed view (`speed.html`), and a by-prompt audio gallery (`samples.html`). The gh-pages site mirrors these so you can compare without running anything locally.
