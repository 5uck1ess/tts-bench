# Methodology

## Why these axes

Open TTS models advertise speed numbers from cherry-picked hardware, often confusing cold and warm starts, and never tell you what the output sounds like. tts-bench measures, on whatever hardware you point it at:

- **Speed** — cold + warm TTFA, RTF, peak RAM, peak VRAM. Cold = first run after process start. Warm = subsequent runs.
- **Samples** — every model × prompt × device combination rendered with an inline audio player so you can pick by ear.
- **Voice cloning** — same prompts, same reference voice, ranked subjectively. See [cloning.md](cloning.md).

An objective quality score (NAQ) is computed into the CSV and paused from publication while the algorithm is redesigned — the current features didn't track subjective ranking closely enough to surface.

## Why cold vs warm

Cold TTFA is what an always-on agent experiences on its first request after a deploy or restart. Warm TTFA is what every subsequent request experiences. Vendor numbers usually quote warm — that's not what users feel during cold start.

## Why reproducible

Every install is `uv` + Python 3.11 + a per-model venv. From a clean machine to a populated `results/` directory is under 15 minutes. The bench has already disproved one vendor claim — NeuTTS Air's "2× realtime on AMD Ryzen 9" turned into 0.9× RTF on x86 Windows CPU.

## What gets reported

Every per-rig run produces a results directory with `results.csv`, `meta.json`, the generated wavs, and four HTML pages: a landing card (`index.html`), a speed view (`speed.html`), a quality view (`quality.html`), and a by-prompt audio gallery (`samples.html`). The gh-pages site mirrors these so you can compare without running anything locally.
