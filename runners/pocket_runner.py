"""Pocket-TTS runner.

API discovered by inspection (2026-05-22):
    from pocket_tts import TTSModel
    from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES
    model = TTSModel.load_model(language="english_2026-04")
    state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/...")
    for chunk in model.generate_audio_stream(state, text):
        ...

Default voices come from _ORIGINS_OF_PREDEFINED_VOICES — we use "anna" for
English and "estelle" for French (both clean VCTK/Unmute references). User
can pass --reference path/to/wav to override.
"""

import argparse
import json
import sys
import time


LANGUAGE_CONFIG = {
    "en": "english_2026-04",
    "fr": "french_24l",
    "de": "german_24l",
    "it": "italian_24l",
    "pt": "portuguese_24l",
    "es": "spanish_24l",
}


DEFAULT_VOICE = {
    "en": "anna",       # VCTK p228
    "fr": "estelle",    # Unmute prod website
    "de": "juergen",
    "it": "giovanni",
    "pt": "anna",       # no pt-specific default in catalog; reuse en
    "es": "lola",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")        # pocket-tts is CPU-only
    p.add_argument("--reference", default=None)
    p.add_argument("--variant", default=None)        # unused
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    args = p.parse_args()

    try:
        from pocket_tts import TTSModel
        import numpy as np
        import soundfile as sf

        if args.language == "en":
            model = TTSModel.load_model()
        else:
            lang_cfg = LANGUAGE_CONFIG.get(args.language, LANGUAGE_CONFIG["en"])
            model = TTSModel.load_model(language=lang_cfg)
        samplerate = int(model.sample_rate)

        # Pocket-TTS accepts either a predefined voice name ("anna") or a path/url to
        # a wav for cloning. Cloning requires HF auth on the gated kyutai/pocket-tts
        # repo; named voices work without auth.
        voice = args.reference or DEFAULT_VOICE.get(args.language, "anna")
        state = model.get_state_for_audio_prompt(voice)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    for i in range(args.runs):
        try:
            t0 = time.perf_counter()
            first = None
            chunks = []
            for chunk in model.generate_audio_stream(state, args.text):
                if first is None:
                    first = time.perf_counter()
                arr = chunk.numpy() if hasattr(chunk, "numpy") else np.asarray(chunk)
                chunks.append(arr)
            t_end = time.perf_counter()

            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
            audio_s = float(len(audio) / samplerate)

            if i == 0:
                sf.write(args.out, audio, samplerate)

            print(json.dumps({
                "ok": True,
                "run_index": i,
                "ttfa_ms": (first - t0) * 1000 if first else None,
                "gen_s": t_end - t0,
                "audio_s": audio_s,
            }), flush=True)
        except Exception as e:
            print(json.dumps({
                "ok": False, "run_index": i,
                "error": f"{type(e).__name__}: {e}",
            }), flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
