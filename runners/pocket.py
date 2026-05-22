"""Pocket-TTS runner.

API is a best guess from the homebase notes — adjust here if first run errors.
"""

import argparse
import json
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")        # pocket-tts is CPU-only
    p.add_argument("--reference", default=None)
    p.add_argument("--variant", default=None)        # unused
    args = p.parse_args()

    try:
        from pocket_tts import PocketTTS  # type: ignore
        import numpy as np
        import soundfile as sf

        init_kwargs = {}
        if args.reference:
            init_kwargs["voice"] = args.reference
        tts = PocketTTS(**init_kwargs)

        samplerate = getattr(tts, "sample_rate", getattr(tts, "samplerate", 24000))

        t0 = time.perf_counter()
        first = None
        chunks = []
        for chunk in tts.generate(args.text):
            if first is None:
                first = time.perf_counter()
            chunks.append(np.asarray(chunk))

        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
        sf.write(args.out, audio, samplerate)

        print(json.dumps({
            "ok": True,
            "ttfa_ms": (first - t0) * 1000 if first else None,
            "audio_s": float(len(audio) / samplerate),
        }))
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
