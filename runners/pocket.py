"""Pocket-TTS runner.

Loads the model once, then does --runs generations back-to-back. Prints one JSON
line per run on stdout. Writes the WAV from run 0 only (others reuse the prompt).

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
    p.add_argument("--runs", type=int, default=1)
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
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    for i in range(args.runs):
        try:
            t0 = time.perf_counter()
            first = None
            chunks = []
            for chunk in tts.generate(args.text):
                if first is None:
                    first = time.perf_counter()
                chunks.append(np.asarray(chunk))
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
