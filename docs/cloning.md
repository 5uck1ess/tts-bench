# Voice cloning

Three flavors of "zero-shot cloning" are supported, with slightly different reference-file requirements:

| Models | Reference needed |
|---|---|
| ChatterBox, Coqui XTTS-v2, LuxTTS, VoxCPM, Echo-TTS | wav only — no transcript |
| NeuTTS Air, NeuTTS Nano, F5-TTS, OmniVoice | wav **+** matching `.txt` transcript (same basename, e.g. `myvoice.wav` + `myvoice.txt`). The transcript MUST be the literal words spoken in the wav. |
| Pocket-TTS (cloning path) | wav, HF accept-terms gated on [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts) + `hf auth login` |

Drop the wav (and optional `.txt`) into `reference/`, then:

```bash
python bench.py --reference reference/myvoice.wav
python compare.py "..." --reference reference/myvoice.wav
python speak.py chatterbox --reference reference/myvoice.wav
```

`--reference` auto-skips models that can only use predefined voices (Kokoro, KittenTTS, Piper, VibeVoice, Magpie, Supertonic, Maya1).

---

## Cloning quality ranking

Ranked by a **blind pairwise voting study** — human-preference 2AFC over the cloning-capable models, fit to Bradley-Terry strengths. Each listener hears two anonymized clones of the same reference (`chris_hemsworth_15s.wav`, 5 prompts) and picks the better one; `tie` and `bad` (both-unusable) votes are recorded too. **397 votes** as of June 2026. W-L-T = wins / losses / ties.

> **What this measures (and doesn't).** Listeners are picking the better **voice match** — timbre/accent fidelity and overall preference. It does **not** measure intelligibility: a clone can top this study while still garbling or dropping words, because a short timbre-focused A/B doesn't penalize that. Pair these standings with the objective **WER** column on the [Scores lens](https://5uck1ess.github.io/tts-bench/scores.html) — see the OmniVoice caveat below.

| # | Model | W-L-T | Games |
|---|---|---|---|
| 1 | **OmniVoice** | 24-1-3 | 28 |
| 2 | **Echo-TTS** | 21-1-6 | 28 |
| 3 | **IndexTTS-2** | 16-2-5 | 23 |
| 4 | F5-TTS | 19-5-2 | 26 |
| 5 | MOSS-TTS | 18-4-2 | 24 |
| 6 | Qwen3-TTS 1.7B (CUDA-graph) | 13-4-1 | 18 |
| 7 | Fish Speech S2-Pro | 14-4-7 | 25 |
| 8 | Pocket-TTS | 13-9-4 | 26 |
| 9 | Zonos v0.1 | 12-8-1 | 21 |
| 10 | VibeVoice 1.5B | 12-7-1 | 20 |
| 11 | VoxCPM2 | 11-6-1 | 18 |
| 12 | ChatterBox Turbo | 12-4-4 | 20 |
| 13 | MOSS-TTS-Nano | 14-9-2 | 25 |
| 14 | ChatterBox | 12-9-5 | 26 |
| 15 | VibeVoice 7B | 7-7-2 | 16 |
| 16 | Sesame CSM-1B | 9-9-2 | 20 |
| 17 | Dia 1.6B | 4-15-0 | 19 |
| 18 | Fish Speech 1.5 | 7-13-0 | 20 |
| 19 | Step-Audio-EditX | 3-13-1 | 17 |
| 20 | NeuTTS Nano | 3-13-1 | 17 |
| 21 | ZipVoice | 3-14-0 | 17 |
| 22 | StyleTTS 2 | 2-15-0 | 17 |
| 23 | MetaVoice-1B | 1-10-0 | 11 † |
| 24 | LuxTTS | 1-16-0 | 17 |
| 25 | Coqui XTTS-v2 | 0-12-0 | 12 † |
| 26 | Mars5-TTS | 0-19-0 | 19 |
| 27 | NeuTTS Air | 0-12-0 | 12 † |
| 28 | OpenVoice v2 | 0-10-0 | 10 † |

† fewer than ~12 games — that row's position is still noisy. This ranks **28 of the 30 cloning-capable models**: base Qwen3-TTS isn't here (its cloning is disabled in-harness — the autoregressive sampler blows the 600s timeout on long prompts; the CUDA-graph variant `qwentts_fast` is the benched cloning path), and Voxtral's cloning is Linux-only (vLLM), so neither is in this Windows cloning set.

**What the votes say:**

- **The top is robust on voice match — with one caveat.** OmniVoice and Echo-TTS each lost exactly one game out of 28 — effectively tied for first on **voice match**, and clear of the field. **But OmniVoice's intelligibility is the weak spot:** it can drop or garble words, which this timbre-focused 2AFC doesn't penalize — so treat its #1 as "best voice match," not "best overall clone," and check objective **WER** (or just audition it) before relying on it. Echo-TTS doesn't have that problem. IndexTTS-2, F5-TTS, and MOSS-TTS form a solid second tier (F5-TTS now edges MOSS-TTS for 4th).
- **The May ear-test was wrong about the middle.** A first-pass single-reference listen had ranked ChatterBox #2; the blind votes drop it to 14th (12-9-5). It's fine, not special. Trust the votes over that early impression.
- **F5-TTS** rides on timbre but its **prosody is the weak point** — phrasing/pauses feel off — which is why it sits behind the leaders despite a strong 19-5 record.
- **VoxCPM2 and Sesame CSM-1B** clone passably but drift from the reference — VoxCPM's timbre wanders, Sesame inserts **fake pauses** mid-sentence (a conversational-chunk training artifact). Mid-pack.
- **Dia 1.6B and Fish Speech 1.5 fell below water** (negative BT, 4-15 and 7-13). Both render clean audio but the cloned timbre drifts off the target — Dia in particular wanders toward a generic voice rather than holding the reference.
- **The zero-win tail doesn't clone.** Coqui XTTS-v2, Mars5-TTS, NeuTTS Air, and OpenVoice v2 went winless — their output doesn't match the reference voice in this study. Mars5 is unusable regardless (0.1× warm RTF). NeuTTS Air/Nano also hit long-form truncation. Keep the NeuTTS/Coqui models for **default-voice** use, not cloning.

These standings are on a **single reference** (`chris_hemsworth_15s.wav`). Replication with another reference (e.g. a clean female voice) is recommended before treating the mid-pack as definitive — the top and bottom tiers are robust, the middle is where another reference could reshuffle.
