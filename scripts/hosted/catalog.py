"""Hosted-board data both pages need: display names + prices, and score aggregates.

Dependency-free on purpose: publish.py's Scores page imports it on any rig (the Linux box has
no hosted venv), so it must not pull in the provider adapters or websockets.
"""

import csv
from collections import defaultdict

# slug -> (display name, $/1M chars as shown, price note). Source + read dates: docs/hosted.md.
CATALOG = {
    "eleven_v4": ("ElevenLabs Eleven v4", "$80", ""),
    "eleven_v4_turbo": ("ElevenLabs Eleven v4 Turbo", "$40", ""),
    "rime_coda": ("Rime Coda", "$50", ""),
    "giggy_tts": ("Giggy TTS", "$10", "streaming tier"),
    "cartesia_sonic36": ("Cartesia Sonic 3.6", "$38–65", "by plan"),
    "gemini38_flash_tts": ("Gemini 3.8 Flash TTS", "≈$15", "token-priced; doubles 2027-01-01"),
    "inworld_tts2": ("Inworld TTS-2", "$25", "on-demand"),
    "speechify_simba35": ("Speechify SIMBA 3.5", "$12–20", "by plan; launch half price"),
    "fish_drama3": ("Fish Audio Drama 3", "$10", "per 1M bytes; $50 after 2026-11-12"),
    "voxtral3_tts": ("Mistral Voxtral TTS 3", "$16", ""),
    "xai_tts": ("xAI TTS", "$15", ""),
    "assemblyai_tts1": ("AssemblyAI TTS-1", "$15", ""),
    "soniox_tts2": ("Soniox TTS v2", "≈$14.1", "$0.70 per hour of speech"),
    "stepaudio3_tts": ("StepFun Step-Audio 3 TTS", "$36", "≈$18 for English"),
    "smallest_lightning31pro": ("Smallest.ai Lightning v3.1 Pro", "$19.5", ""),
    "deepgram_flux_tts": ("Deepgram Flux TTS", "$45", ""),
    "mai_voice21": ("Microsoft MAI-Voice-2.1", "$22", ""),
    "vui_luna": ("VUI Labs Luna", "$15", ""),
    "openai_gpt4o_mini_tts": ("OpenAI gpt-4o-mini-tts", "≈$17", "token-priced"),
    "openai_realtime21": ("OpenAI gpt-realtime-2.1", "token", "$64 per 1M audio-out tokens"),
    "murf_falcon21": ("Murf Falcon 2.1", "$10", ""),
    "heygen_voice1": ("HeyGen Voice 1", "$30", "$15 via OpenRouter"),
    "minimax_speech28hd": ("MiniMax speech-2.8-hd", "$100", "via OpenRouter"),
    "bland_speech3": ("Bland Speech v3", "$40", ""),
    "gradium_tts": ("Gradium", "$35.9–57.8", "by plan"),
    "async_flash15": ("Async Flash v1.5", "≈$9.3", "$0.50 per hour of audio"),
    "palabra_tts": ("Palabra TTS", "$12", ""),
    "resemble_ultra": ("Resemble Ultra", "≈$33", "$0.0005 per second of audio (Flex)"),
}


# slug -> the exact model id our adapter sends for English (Adapter.model_sent("en")). Several display
# names are only the company ("Gradium", "Palabra TTS"), so every hosted row also shows this.
# Kept here, dependency-free, for the Scores page; a test keeps it equal to the adapters.
MODEL_ID = {
    "eleven_v4": "eleven_v4",
    "eleven_v4_turbo": "eleven_v4_turbo",
    "rime_coda": "coda",
    "giggy_tts": "giggyspeech",
    "cartesia_sonic36": "sonic-3.6",
    "gemini38_flash_tts": "gemini-3.8-flash-tts",
    "inworld_tts2": "inworld-tts-2",
    "speechify_simba35": "simba-3.5",
    "fish_drama3": "drama-3",
    "voxtral3_tts": "voxtral-mini-tts-3-0",
    "xai_tts": "xai-tts",
    "assemblyai_tts1": "assemblyai-tts-1",
    "soniox_tts2": "tts-rt-v2",
    "stepaudio3_tts": "stepaudio-3-tts",
    "smallest_lightning31pro": "lightning_v3.1_pro",
    "deepgram_flux_tts": "flux-haley-en",
    "mai_voice21": "MAI-Voice-2.1",
    "vui_luna": "luna-tts",
    "openai_gpt4o_mini_tts": "gpt-4o-mini-tts-2025-12-15",
    "openai_realtime21": "gpt-realtime-2.1",
    "murf_falcon21": "falcon-2.1",
    "heygen_voice1": "heygen-voice-1",
    "minimax_speech28hd": "minimax/speech-2.8-hd",
    "bland_speech3": "btts-3",
    "gradium_tts": "default",
    "async_flash15": "async_flash_v1.5",
    "palabra_tts": "auto",
    "resemble_ultra": "resemble-ultra",
}


# Vendors whose API takes no pinned model: the id we send is an alias the vendor resolves to
# whatever it serves now, so a later re-run may hit a different model under the same label.
# Checked against each vendor's docs 2026-11-02.
MODEL_NOTE = {
    "gradium_tts": "Gradium's production alias (docs: model_name defaults to \"default\"); "
                   "no pinned version is offered, so Gradium picks the model",
    "palabra_tts": "Palabra's documented selector (\"use auto to select the best model for the "
                   "language\"); no other model id is documented, so Palabra picks the model",
}


# Vendors that send a different model id per language (Deepgram encodes voice + language in
# it). slug -> {lang: id sent}; MODEL_ID holds the English one.
MODEL_BY_LANG = {
    "deepgram_flux_tts": {"en": "flux-haley-en", "es": "flux-aina-es", "fr": "flux-arnaud-fr"},
}


def model_shown(slug):
    """The model id as shown on the boards: the English id sent, flagged when it varies by
    language, and an unpinned alias says the vendor picks."""
    return (MODEL_ID[slug] + (" · per language" if slug in MODEL_BY_LANG else "")
            + (" · vendor picks" if slug in MODEL_NOTE else ""))


def model_title(slug):
    """Hover text for the model id: every id actually sent."""
    if slug in MODEL_BY_LANG:
        sent = ", ".join(f"{lang}: {m}" for lang, m in sorted(MODEL_BY_LANG[slug].items()))
    else:
        sent = MODEL_ID[slug]
    return f"model id sent: {sent}" + (f" — {MODEL_NOTE[slug]}" if slug in MODEL_NOTE else "")


def label(slug):
    """Display name plus the model id actually called, e.g. "Rime Coda · coda"."""
    return f"{CATALOG[slug][0]} · {model_shown(slug)}"


def score_aggregates(path, mode):
    """{slug: {"utmos", "wer", "sim": mean or None, "flags": [(pid, flag)], "n": clips}} over
    every scored clip of one lens (all prompts, all takes). Same definition as the Scores page:
    the mean over the clips, blanks skipped per metric."""
    acc = defaultdict(lambda: {"utmos": [], "wer": [], "sim": [], "flags": [], "n": 0})
    try:
        f = open(path, encoding="utf-8", newline="")
    except FileNotFoundError:
        return {}
    with f:
        for r in csv.DictReader(f):
            if r["mode"] != mode:
                continue
            a = acc[r["model"]]
            a["n"] += 1
            for k in ("utmos", "wer", "sim"):
                if r.get(k):
                    a[k].append(float(r[k]))
            a["flags"] += [(r["prompt_id"], x) for x in (r.get("health") or "").split(";") if x]
    return {s: {**{k: (sum(a[k]) / len(a[k]) if a[k] else None) for k in ("utmos", "wer", "sim")},
                # one flag per (prompt, flag): 3 takes with the same defect count once, as
                # the local board's one clip per prompt would
                "flags": sorted(set(a["flags"])), "n": a["n"]} for s, a in acc.items()}
