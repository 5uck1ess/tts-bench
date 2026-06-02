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

> **Update (June 2026):** the list below is a *first-pass subjective* impression from a single May-2026 listening pass (10 models, one reference). It's being superseded by a **blind pairwise voting study** (human-preference Elo over all 29 cloning-capable models). Early standings agree at the top — **Echo-TTS** and **OmniVoice** lead — but treat the ordering below as preliminary until the vote-based ranking is published.

Numbers tell you which model is fast; ears tell you which one is *good*. After a full listening pass on the RTX 5090 cloning bench ([`2026-05-24_0154`](https://5uck1ess.github.io/tts-bench/2026-05-24_0154/report.html), `chris_hemsworth_15s.wav` reference, 5 prompts × every cloning-capable model), ranked best → worst by accent preservation and naturalness:

1. **OmniVoice — #1.** Keeps Chris Hemsworth's Australian accent better than any other model, prosody natural. Some artifacts in the audio but the cloned voice itself is the closest match to the reference.
2. **ChatterBox** — strong second. Keeps the accent well, clean output. Trade-off vs OmniVoice is mostly RTF (2× warm vs 9.2×).
3. **IndexTTS-2** — also good, accent preserved. Slower than the top two (1.1× warm RTF) but the quality holds.
4. **F5-TTS** — decent on timbre, **prosody is the weak point** — phrasing feels off / unnatural pauses. Best cloning RTF after OmniVoice though (5.3×).
5. **VoxCPM2** — sounds okay as a general TTS but **cloning isn't its strong suit**; the cloned voice drifts from the reference. Use it for default-voice multilingual instead.
6. **Sesame CSM-1B** — accent comes through but the voice doesn't come out as **deep as the reference**, and it inserts **fake pauses** mid-sentence (likely an artifact of being trained on conversational turn chunks).
7. **Coqui XTTS-v2** — multilingual baseline. Cloning behind ChatterBox/IndexTTS/F5.
8. **Qwen3-TTS (`qwentts_fast`)** — mid. Competent TTS but **cloning fidelity is weak**; timbre wanders from the reference. (The base `qwentts` no longer exposes a cloning path — predefined-only now; the runaway/600s-timeout that plagued the old base cloning attempt was fixed in `qwentts_fast` via `non_streaming_mode`.)
9. **NeuTTS Air / Nano** — **doesn't work for cloning** in this test (output unrelated to reference voice, plus the long-form truncation issue from the compare pass). Keep for default-voice usage.
10. **MARS5-TTS** — **doesn't work** — output didn't match the reference, plus 0.1× warm RTF makes it unusable regardless. Bottom of the list.

Pocket-TTS isn't ranked here — it's CPU-only and its gated cloning path produced unusable artifacts in the earlier compare pass.

These are first-pass impressions on a single reference; replication with another reference (e.g. a clean female voice) recommended before treating the ordering as definitive — especially the middle of the pack.

> A data-driven ranking from the blind pairwise voting study (human-preference Elo) will replace these subjective impressions once enough votes are in across raters.
