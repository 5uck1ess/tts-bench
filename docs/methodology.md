# Methodology

## Why three axes

Open TTS models advertise speed numbers from cherry-picked hardware, often confusing cold and warm starts, and never tell you what the output sounds like. tts-bench measures three things, on whatever hardware you point it at:

- **Speed** — cold + warm TTFA, RTF, peak RAM, peak VRAM. Cold = first run after process start. Warm = subsequent runs.
- **Quality (NAQ)** — objective 0-100 per-wav score combining two factor groups: ARTIFACT (cues for absence of vocoder artifacts) and NATURALNESS (cues for expressive prosody). Captures both axes, not just one. Best-effort acoustic proxy until voting-system ground truth ships. See [naq.md](naq.md).
- **Voice cloning** — same prompts, same reference voice, ranked subjectively. See [cloning.md](cloning.md).

## Why cold vs warm

Cold TTFA is what an always-on agent experiences on its first request after a deploy or restart. Warm TTFA is what every subsequent request experiences. Vendor numbers usually quote warm — that's not what users feel during cold start.

## Why reproducible

Every install is `uv` + Python 3.11 + a per-model venv. From a clean machine to a populated `results/` directory is under 15 minutes. The bench has already disproved one vendor claim — NeuTTS Air's "2× realtime on AMD Ryzen 9" turned into 0.9× RTF on x86 Windows CPU.

## What gets reported

Every per-rig run produces a results directory with `results.csv`, `meta.json`, the generated wavs, and four HTML pages: a landing card (`index.html`), a speed view (`speed.html`), a quality view (`quality.html`), and a by-prompt audio gallery (`samples.html`). The gh-pages site mirrors these so you can compare without running anything locally.
