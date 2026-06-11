# Expressive control per model

Which benched models accept explicit emotion/delivery control — inline tags, style instructions,
emotion references, or numeric knobs — with the exact syntax, verified against each model's official
HF card / GitHub README / docs (June 2026). Claims are checkpoint-specific: a feature in a newer or
sibling variant of a family does **not** count unless the exact checkpoint we bench has it.

> **The bench never uses any of this.** Every model gets the same plain prompts so timing and
> quality scores stay comparable. This page is "what you *could* do with the model," not what the
> bench measured.

Legend (matches the **Expressive** column in the README tables):

| Code | Meaning |
|---|---|
| `tags` | inline cues in the spoken text itself — `(laughs)`, `[sigh]`, `<laugh>`, emotion tokens |
| `desc` | natural-language style/emotion instructions (separate field or in-text wrapper) |
| `emo-ref` | emotion conditioned on a *separate* reference clip or emotion vector (distinct from the voice-cloning reference) |
| `knob` | numeric or preset parameter — exaggeration, style enum, pitch/speed levels |
| `—` | none; for cloning models, expression simply follows whatever the reference clip does |

## Inline tags

| Model | Syntax | Notes |
|---|---|---|
| Dia 1.6B-0626 | `(laughs) (clears throat) (sighs) (gasps) (coughs) (singing) (sings) (mumbles) (beep) (groans) (sniffs) (claps) (screams) (inhales) (exhales) (applause) (burps) (humming) (sneezes) (chuckle) (whistles)` | Complete documented list — tags outside it can produce unexpected output. |
| Fish Speech S2-Pro | `[laughing] [sigh] [whisper] [angry] [excited] [pause] …` | Free-form natural language in `[brackets]`, 15k+ tags, sub-word placement. S2-Pro native — distinct from OpenAudio S1's `(parenthesis)` markers, and **not** present in Fish Speech 1.5. |
| Higgs Audio v3 | `<\|emotion:elation\|>` (21 emotions), `<\|style:singing\|>` / `whispering` / `shouting`, `<\|prosody:speed_*/pitch_*/pause\|>`, `<\|sfx:laughter/cough/sigh/sneeze\|>` | Emotion/style/speed tokens at turn start; pause/sfx inline. 21 emotions documented in the SGLang cookbook: elation, amusement, enthusiasm, determination, pride, contentment, affection, relief, contemplation, confusion, surprise, awe, longing, arousal, anger, fear, disgust, bitterness, sadness, shame, helplessness. |
| Maya1 | `<laugh> <sigh> <whisper> <angry> <giggle> <chuckle> <gasp> <cry> <laugh_harder> <excited> <disappointed> <sarcastic> <sing>` (20+) plus voice design `<description="40-year-old, warm baritone, conversational">` | Both mechanisms on the official card. |
| Supertonic 3 | `<laugh> <breath> <sigh>` | Deliberately small set (on-device ONNX budget). |
| Echo-TTS | `(laughs) (angry) (whispering)` + `[S1]`/`[S2]` speaker turns | Documented as examples; exhaustive list unpublished. Commas act as pauses; `!` raises expressiveness but can cost quality. |
| Step-Audio-EditX | 10 paralinguistic tokens (Breathing, Laughter, Sigh, Uhm, Surprise-oh/ah/wa, Confirmation-en, Question-ei, Dissatisfaction-hnn) + edit instructions (see desc section) | Editing model — tags apply in its iterative edit passes. |
| MOSS-TTS v1.5 | `[pause 3.2s]` inline | Pause control only (added over v1.0) — no emotion tags. |
| OmniVoice | `[laughter] [sigh] [confirmation-en] [question-*] [surprise-*] [dissatisfaction-hnn]` | **Caveat:** documented, but GitHub issue #28 reports most tags are not reliably processed in practice. Ear-test before relying on them. |
| Higgs Audio v2 | `[laugh] [music]` seen in official demos | **Caveat:** the v2 base model is not fine-tuned — tag behavior is emergent and undocumented (an issue asking for the list was never answered). |

## Style / emotion instructions (`desc`)

| Model | Mechanism |
|---|---|
| DramaBox | The prompt *is* the control surface — screenplay style: `A woman speaks warmly, "Hello?" She laughs, "Hahaha!"` Stage directions (sighs, gulps, pauses) go **outside** quotes; laughter as phonetics ("Hahaha") **inside** quotes. The optional ~10 s reference clip clones timbre only. |
| VoxCPM2 | Parenthetical in-text prompt: `(A young woman, gentle and sweet voice)Hello…` for voice design, or `(slightly faster, cheerful tone)` + reference wav for controllable cloning. Both paths in the official README with code examples. |
| Parler-TTS Mini v1 | Free-text description: speaker (34 named voices), gender, speaking rate, pitch, reverb, background noise, recording quality. **Caveat:** the card itself says Mini v1 "is not very good at conveying emotions" — expressive emotion lives in the separate Expresso fine-tune, which we don't bench. |
| Step-Audio-EditX | `--edit-type emotion --edit-info happy` — emotions: Angry, Happy, Sad, Excited, Fearful, Surprised, Disgusted; styles: Whisper, Serious, Generous, Child, Older, Act_coy, Exaggerated. Plain TTS mode also accepts a style-instruction prompt. |
| Maya1 | `<description="…">` voice-design wrapper (see tags section). |
| IndexTTS-2 | `use_emo_text=True` infers emotion guidance from text via a fine-tuned Qwen3 (see emo-ref section). |

## Emotion reference / vector (`emo-ref`) and knobs

| Model | Mechanism |
|---|---|
| IndexTTS-2 | Three independent mechanisms: `emo_audio_prompt` + `emo_alpha=0.6` (emotion ref can be a **different speaker** than the timbre ref); `emo_vector=[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]` (8 floats 0–1); `use_emo_text=True`. Our runner exposes none of these (plain cloning only, for parity). |
| Zonos v0.1 | `emotion=[happiness, sadness, disgust, fear, surprise, anger, other, neutral]` — 8 weights summing to ~1 — plus `speaking_rate` and `pitch_std` floats. Angry/happy steer more reliably than sad (entangled with text sentiment). |
| ChatterBox | `exaggeration=0.5` (0.25–2.0) + `cfg_weight` — the "emotion exaggeration" knob it's known for. **ChatterBox Turbo silently ignores both** (confirmed by Resemble in HF discussion #22) — use the base model if you need the knob. |
| OpenVoice v2 | `style=` enum: `default friendly cheerful excited sad angry terrified shouting whispering`. The base speaker renders the style; the tone-color converter then grafts the target timbre onto it. |
| StyleTTS 2 | Style vector from reference audio (`compute_style()`), `embedding_scale` (classifier-free guidance — higher = more emotional), `alpha`/`beta` timbre-vs-prosody blend. Numeric API only. |
| MiraTTS | Spark-TTS-inherited discrete pitch/speed levels (`very_low → very_high`) + reference clip. Inferred from the Spark-TTS base — the MiraTTS card is sparse. |
| Magpie-TTS | Emotion-variant voice names (`…Aria.Happy`, `…Pascal.Calm`; Angry/Calm/Disgusted/Fearful/Happy/Neutral/Sad). **Caveat:** documented in NVIDIA's NIM docs; unverified whether the raw HF checkpoint exposes them outside the NIM API. |

## No explicit control

Expression follows the reference clip (cloning models) or is fixed/emergent (predefined models):

- **Reference-clip-driven:** Coqui XTTS-v2, F5-TTS, Mars5 (deep clone), MetaVoice, Sesame CSM (context segments), Pocket-TTS (its bundled voice library includes named emotional variants, e.g. Expresso "confused"), OuteTTS (speaker profile), NeuTTS Air/Nano, MOSS-TTS v1.0 / Nano (the description-driven MOSS-VoiceGenerator is a *sibling* checkpoint we don't bench), Voxtral (Mistral's "voice-as-an-instruction": pick a reference that embodies the affect), VibeVoice 1.5B/7B (spontaneous emotion/singing is explicitly non-controllable; the only lever is choosing a voice prompt with the register you want), dots.tts.
- **Nothing:** Kokoro (officially: "cannot laugh… cannot sound extremely angry"), KittenTTS 0.1 (expressive voices arrived in 0.2), Piper (only espeak phoneme injection `[[ … ]]`, which is pronunciation, not affect), LuxTTS, ZipVoice, Soprano, Fish Speech 1.5 (markers arrived with OpenAudio S1), Qwen3-TTS 1.7B **Base** (the card's "Instruction Control" column is explicitly blank for Base — instructions live in the CustomVoice/VoiceDesign variants), VibeVoice Realtime 0.5B (card: "No need for manual emotion tags"), MeloTTS (speed knob only), Miso TTS 8B (emotive but emergent — no tags/instructions documented).
