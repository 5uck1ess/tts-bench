"""NeuTTS Air + Nano runner.

API is a best guess from the homebase notes — adjust MODEL_IDS and the call
shape after first run if they're wrong.
"""

import argparse
import json
import sys
import time
from pathlib import Path


MODEL_IDS = {
    "air":  "neuphonic/neutts-air-q4-gguf",
    "nano": "neuphonic/neutts-air-q4-gguf",  # placeholder — update with real Nano id
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None)
    p.add_argument("--variant", default="air")
    args = p.parse_args()

    try:
        from neutts import NeuTTS  # type: ignore
        import numpy as np
        import soundfile as sf

        model_id = MODEL_IDS.get(args.variant, MODEL_IDS["air"])
        tts = NeuTTS.from_pretrained(model_id, device=args.device)
        samplerate = getattr(tts, "sample_rate", getattr(tts, "samplerate", 24000))

        ref_text = None
        if args.reference:
            txt_path = Path(args.reference).with_suffix(".txt")
            if txt_path.exists():
                ref_text = txt_path.read_text(encoding="utf-8").strip()

        t0 = time.perf_counter()
        first = None
        chunks = []
        for chunk in tts.synthesize_stream(args.text, ref_audio=args.reference, ref_text=ref_text):
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
