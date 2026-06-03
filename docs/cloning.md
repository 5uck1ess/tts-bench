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

Ranked by a **blind pairwise voting study** — human-preference 2AFC over the cloning-capable models, fit to Bradley-Terry strengths. Each listener hears two anonymized clones of the same reference (`chris_hemsworth_15s.wav`, 5 prompts) and picks the better one; `tie` and `bad` (both-unusable) votes are recorded too. **325 votes** as of June 2026. W-L-T = wins / losses / ties.

| # | Model | W-L-T | Games |
|---|---|---|---|
| 1 | **OmniVoice** | 20-1-2 | 23 |
| 2 | **Echo-TTS** | 18-1-4 | 23 |
| 3 | **IndexTTS-2** | 14-2-3 | 19 |
| 4 | MOSS-TTS | 14-3-2 | 19 |
| 5 | F5-TTS | 19-5-2 | 26 |
| 6 | Qwen3-TTS 1.7B (CUDA-graph) | 10-4-1 | 15 |
| 7 | Pocket-TTS | 12-7-4 | 23 |
| 8 | Fish Speech S2-Pro | 11-4-7 | 22 |
| 9 | VoxCPM2 | 9-6-1 | 16 |
| 10 | Zonos v0.1 | 11-8-0 | 19 |
| 11 | VibeVoice 1.5B | 9-7-1 | 17 |
| 12 | ChatterBox Turbo | 11-4-4 | 19 |
| 13 | ChatterBox | 10-6-5 | 21 |
| 14 | MOSS-TTS-Nano | 12-9-2 | 23 |
| 15 | Sesame CSM-1B | 8-8-2 | 18 |
| 16 | VibeVoice 7B | 3-5-2 | 10 † |
| 17 | Fish Speech 1.5 | 7-7-0 | 14 |
| 18 | Dia 1.6B | 4-11-0 | 15 |
| 19 | Qwen3-TTS Base | 3-4-0 | 7 † |
| 20 | Step-Audio-EditX | 3-12-1 | 16 |
| 21 | NeuTTS Nano | 3-12-1 | 16 |
| 22 | ZipVoice | 2-12-0 | 14 |
| 23 | StyleTTS 2 | 2-14-0 | 16 |
| 24 | MetaVoice-1B | 1-7-0 | 8 † |
| 25 | LuxTTS | 1-14-0 | 15 |
| 26 | Coqui XTTS-v2 | 0-10-0 | 10 † |
| 27 | Mars5-TTS | 0-17-0 | 17 |
| 28 | NeuTTS Air | 0-8-0 | 8 † |
| 29 | OpenVoice v2 | 0-9-0 | 9 † |

† fewer than ~12 games — that row's position is still noisy. Voxtral isn't ranked here: its cloning path is Linux-only (vLLM) and so isn't in this Windows cloning set, which is 29 of the 30 cloning-capable models.

**What the votes say:**

- **The top is robust.** OmniVoice and Echo-TTS each lost exactly one game out of 23 — effectively tied for first, and clear of the field. IndexTTS-2, MOSS-TTS, and F5-TTS form a solid second tier.
- **The May ear-test was wrong about the middle.** A first-pass single-reference listen had ranked ChatterBox #2; the blind votes drop it to 13th (10-6-5). It's fine, not special. Trust the votes over that early impression.
- **F5-TTS** rides on timbre but its **prosody is the weak point** — phrasing/pauses feel off — which is why it sits behind the leaders despite a strong 19-5 record.
- **VoxCPM2 and Sesame CSM-1B** clone passably but drift from the reference — VoxCPM's timbre wanders, Sesame inserts **fake pauses** mid-sentence (a conversational-chunk training artifact). Mid-pack.
- **The zero-win tail doesn't clone.** Coqui XTTS-v2, Mars5-TTS, NeuTTS Air, and OpenVoice v2 went winless — their output doesn't match the reference voice in this study. Mars5 is unusable regardless (0.1× warm RTF). NeuTTS Air/Nano also hit long-form truncation. Keep the NeuTTS/Coqui models for **default-voice** use, not cloning.

These standings are on a **single reference** (`chris_hemsworth_15s.wav`). Replication with another reference (e.g. a clean female voice) is recommended before treating the mid-pack as definitive — the top and bottom tiers are robust, the middle is where another reference could reshuffle.
