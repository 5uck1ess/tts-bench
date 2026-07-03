"""Kokoro-82M MLX runner — Apple-Silicon-native backend via mlx-audio.

Same model + voices as runners/kokoro_runner.py (Apache 2.0, predefined voices
only, no zero-shot cloning), but generation runs on Apple's MLX framework
(mlx-audio's Kokoro port) instead of PyTorch-MPS. Registered as the separate
slug `kokoro_mlx` so its speed sits next to PyTorch Kokoro on the Mac rig — see
docs/known-issues.md for why MPS still wins.

API discovered by inspection (2026-06-30, mlx-audio):
    from mlx_audio.tts.utils import load
    model = load("prince-canuma/Kokoro-82M")          # MLX weights repo
    for res in model.generate(text=..., voice="af_heart", lang_code="a"):
        res.audio        # mlx.core array (mono, 24kHz); lazy — must mx.eval()

Apple-Silicon only: mlx has no CUDA/CPU-rig wheel, so install.sh builds the
kokoro_mlx venv on Darwin only and the MODELS entry lists devices=["mps"]
(detect_mps falls back to MLX's Metal probe for this torch-free venv).

Install gotcha (same as kokoro): mlx-audio imports misaki.en for G2P, which
needs num2words + spaCy en_core_web_sm. install.sh pre-installs `misaki[en]`
plus the en_core_web_sm wheel so misaki doesn't shell out to pip in the uv venv.
"""

import argparse
import json
import sys
import time

import _meminfo


# Single-letter language codes, matching kokoro_runner. mlx-audio's KokoroPipeline
# uses the same misaki G2P, so the codes carry over (French uses the espeak fallback).
LANG_CODE = {
    "en": "a",   # American English
    "fr": "f",
    "es": "e",
    "it": "i",
    "pt": "p",
    "de": "a",   # no German in Kokoro — fall back to American EN
    "ja": "j",
    "zh": "z",
    "hi": "h",
}


DEFAULT_VOICE = {
    "en": "af_heart",
    "fr": "ff_siwis",
    "es": "ef_dora",
    "it": "if_sara",
    "pt": "pf_dora",
    "ja": "jf_alpha",
    "zh": "zf_xiaobei",
    "hi": "hf_alpha",
}

REPO_ID = "prince-canuma/Kokoro-82M"  # MLX-format weights (distinct from hexgrad's PyTorch repo)
SAMPLERATE = 24000  # Kokoro fixed output rate


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="mps")
    p.add_argument("--reference", default=None,
                   help="Voice NAME (e.g. 'af_heart') — Kokoro doesn't support zero-shot wav cloning.")
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
        import mlx.core as mx
        import numpy as np
        import soundfile as sf
        from mlx_audio.tts.utils import load

        lang_code = LANG_CODE.get(args.language, "a")
        model = load(REPO_ID)
        voice = args.reference or DEFAULT_VOICE.get(args.language, "af_heart")
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            first = None
            chunks = []
            for res in model.generate(text=text, voice=voice, speed=1.0, lang_code=lang_code):
                audio = res.audio
                if audio is None:
                    continue
                # MLX is lazy: force the segment to materialize so the time we
                # measure is real compute, not deferred graph-building. This is the
                # MLX equivalent of kokoro_runner's torch `.cpu().numpy()`.
                mx.eval(audio)
                arr = np.array(audio)
                if first is None:
                    first = time.perf_counter()
                chunks.append(arr)
            t_end = time.perf_counter()

            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
            audio_s = float(len(audio) / SAMPLERATE)
            if write_wav:
                sf.write(out_path, audio, SAMPLERATE)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (first - t0) * 1000 if first else None,
                "gen_s": t_end - t0, "audio_s": audio_s,
                **_meminfo.sample(args.device),
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
