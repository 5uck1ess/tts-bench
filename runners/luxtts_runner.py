"""LuxTTS runner.

Loads the model once, then does --runs generations back-to-back. Prints one JSON
line per run on stdout. Writes the WAV from run 0 only.

API is a best guess from the homebase notes — adjust after first run if wrong.
"""

import argparse
import json
import sys
import time

import _meminfo


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")  # accepted for harness uniformity
    args = p.parse_args()

    try:
        from luxtts import LuxTTS  # type: ignore
        import numpy as np
        import soundfile as sf

        tts = LuxTTS(device=args.device)
        samplerate = 48000  # homebase note: LuxTTS outputs 48 kHz
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    for i in range(args.runs):
        try:
            _meminfo.reset_peak(args.device)
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
                **_meminfo.sample(args.device),
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
