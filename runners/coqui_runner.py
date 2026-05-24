"""Coqui XTTS-v2 runner (CPML 1.0 license, zero-shot voice cloning, 17 langs).

The original Coqui org shut down end-of-2023; this uses the actively-maintained
fork at github.com/idiap/coqui-ai-TTS (PyPI: `coqui-tts`). The model itself
is unchanged from the original 2023 XTTS-v2 release — it remains the de facto
multilingual cloning baseline.

API (coqui-tts):
    from TTS.api import TTS
    m = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    # cloning:        m.tts(text=..., speaker_wav=ref_wav, language="en")
    # built-in voice: m.tts(text=..., speaker="Claribel Dervla", language="en")
    # returns: list of float samples at 24kHz

Languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi.

Non-streaming through this API. TTFA == gen_s. (XTTS supports streaming via
the lower-level Xtts.inference_stream(), but that requires the Xtts/XttsConfig
classes directly — out of scope for this runner.)

License gotcha:
- First load auto-accepts the Coqui Public Model License via the
  COQUI_TOS_AGREED=1 env var (set in this runner). The license is non-commercial.
"""

import argparse
import json
import os
import sys
import time

import _meminfo
import _naq


# Auto-accept Coqui Public Model License (non-commercial). Must be set before
# importing TTS — otherwise the lib prompts interactively and the subprocess
# stalls on stdin waiting for "y".
os.environ.setdefault("COQUI_TOS_AGREED", "1")


XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# A reasonable English default speaker name (XTTS ships ~58 built-in voices).
DEFAULT_SPEAKER = "Claribel Dervla"

# XTTS language codes match ISO 639-1 except zh-cn. Our --language uses ISO 639-1.
LANG_MAP = {"zh": "zh-cn"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot voice cloning (no transcript needed).")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    language = LANG_MAP.get(args.language, args.language)

    try:
        from TTS.api import TTS
        import numpy as np
        import soundfile as sf

        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
        m = TTS(XTTS_MODEL).to(device)
        samplerate = 24000  # XTTS-v2 output rate
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            if args.reference:
                wav = m.tts(text=text, speaker_wav=args.reference, language=language)
            else:
                wav = m.tts(text=text, speaker=DEFAULT_SPEAKER, language=language)
            t_end = time.perf_counter()

            arr = np.asarray(wav, dtype=np.float32)
            audio_s = float(len(arr) / samplerate)
            if write_wav:
                sf.write(out_path, arr, samplerate)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
                **_meminfo.sample(args.device),
                **(_naq.score(out_path) if write_wav else {"naq": None, "naq_harm": None, "naq_buzz": None}),
            }), flush=True)
            return True
        except Exception as e:
            print(json.dumps({
                "ok": False, "run_index": run_index,
                "error": f"{type(e).__name__}: {e}",
            }), flush=True)
            return False

    if args.stdin:
        idx = 0
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError as e:
                print(json.dumps({"ok": False, "run_index": idx,
                                  "error": f"json parse: {e}"}), flush=True)
                idx += 1
                continue
            _one(job["text"], job["out"], idx, write_wav=True)
            idx += 1
        return 0

    for i in range(args.runs):
        if not _one(args.text, args.out, i, write_wav=(i == 0)):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
