"""LuxTTS runner.

API is a best guess from the homebase notes — adjust after first run if wrong.
"""

import argparse
import json
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None)
    p.add_argument("--variant", default=None)
    args = p.parse_args()

    try:
        from luxtts import LuxTTS  # type: ignore
        import numpy as np
        import soundfile as sf

        tts = LuxTTS(device=args.device)
        samplerate = 48000  # homebase note: LuxTTS outputs 48 kHz

        t0 = time.perf_counter()
        first = None
        chunks = []

        result = tts.tts(args.text, ref_audio=args.reference) if args.reference else tts.tts(args.text)
        if hasattr(result, "__iter__") and not isinstance(result, (bytes, bytearray, str)):
            for chunk in result:
                if first is None:
                    first = time.perf_counter()
                chunks.append(np.asarray(chunk))
        else:
            first = time.perf_counter()
            chunks.append(np.asarray(result))

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
