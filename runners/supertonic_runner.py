"""Supertonic runner (~99M ONNX, MIT code + OpenRAIL-M weights, 31 langs).

Open-weight release is predefined-voice only: zero-shot cloning lives in
the commercial Voice Builder / Supertone Play API. Default-voice path
covers the 31 supported language codes via the lang= parameter.

API (supertonic==1.3.x):
    from supertonic import TTS
    tts = TTS(auto_download=True)
    style = tts.get_voice_style(voice_name="M1")
    wav, dur = tts.synthesize(text, lang="en", voice_style=style,
                              total_steps=8, speed=1.05)
    # wav: numpy (1, N), float32, 44100 Hz

Languages supported (ISO codes, 31): ar, bg, hr, cs, da, nl, en, et, fi,
fr, de, el, hi, hu, id, it, ja, ko, lv, lt, pl, pt, ro, ru, sk, sl, es,
sv, tr, uk, vi. Plus "na" for language-agnostic input.

Runtime is ONNX (CPU-optimized; GPU mode not tested per upstream HF doc).
Runner accepts --device cuda but Supertonic falls back to CPU internally.
"""

import argparse
import json
import sys
import time


SAMPLE_RATE = 44100
SUPPORTED_LANGS = {
    "ar", "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "hi", "hu", "id", "it", "ja", "ko", "lv", "lt", "pl", "pt",
    "ro", "ru", "sk", "sl", "es", "sv", "tr", "uk", "vi", "na",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Voice NAME string (e.g. 'M1') — Supertonic open release doesn't support zero-shot wav cloning.")
    p.add_argument("--variant", default=None)        # unused
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    try:
        from supertonic import TTS
        import soundfile as sf

        lang = args.language if args.language in SUPPORTED_LANGS else "na"
        voice_name = args.reference or "M1"

        tts = TTS(auto_download=True)
        style = tts.get_voice_style(voice_name=voice_name)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            t0 = time.perf_counter()
            wav, _dur = tts.synthesize(
                text=text, lang=lang, voice_style=style,
                total_steps=8, speed=1.0,
            )
            t_end = time.perf_counter()

            arr = wav.squeeze()  # (1, N) -> (N,)
            audio_s = float(len(arr) / SAMPLE_RATE)
            if write_wav:
                sf.write(out_path, arr, SAMPLE_RATE)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
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
