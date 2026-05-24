"""Piper runner (MIT, per-language voice models, predefined-only, streaming).

API (piper-tts==1.4+):
    from piper import PiperVoice
    v = PiperVoice.load(voice_path)   # .onnx + .onnx.json next to it
    for chunk in v.synthesize(text):
        chunk.audio_int16_array   # numpy int16, mono, at chunk.sample_rate

Streaming-native. Sample rate varies per voice (typically 22050 for medium).

Voice naming: <lang>_<COUNTRY>-<voice>-<quality>
    en_US-lessac-medium, fr_FR-siwis-medium, de_DE-thorsten-medium, etc.

Auto-downloads requested voice on first use via piper.download_voices to
~/.cache/piper-voices/ (or PIPER_VOICE_DIR if set).

Install gotcha — historic: Piper used to depend on piper-phonemize (no
Windows wheels). piper-tts 1.4+ replaced this with bundled espeak-ng,
so Windows works natively now. The wheel-gap warning in older docs no
longer applies to Piper itself (still applies to LuxTTS).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import _meminfo
import _naq


DEFAULT_VOICE = {
    "en": "en_US-lessac-medium",
    "fr": "fr_FR-siwis-medium",
    "de": "de_DE-thorsten-medium",
    "es": "es_ES-mls_9972-low",
    "it": "it_IT-paola-medium",
    "pt": "pt_PT-tugão-medium",
}


def _resolve_voice(voice_code: str, voice_dir: Path) -> Path:
    voice_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = voice_dir / f"{voice_code}.onnx"
    if not onnx_path.exists():
        from piper.download_voices import download_voice
        download_voice(voice_code, voice_dir)
    return onnx_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Voice CODE (e.g. 'en_US-amy-medium') — Piper doesn't support wav cloning.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    try:
        from piper import PiperVoice
        import numpy as np
        import soundfile as sf

        voice_code = args.reference or DEFAULT_VOICE.get(args.language)
        if not voice_code:
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": f"no default Piper voice for language={args.language}; pass --reference <voice_code>"}))
            return 1

        voice_dir = Path(os.environ.get("PIPER_VOICE_DIR", Path.home() / ".cache" / "piper-voices"))
        onnx_path = _resolve_voice(voice_code, voice_dir)
        use_cuda = args.device == "cuda"
        v = PiperVoice.load(onnx_path, use_cuda=use_cuda)
        samplerate = int(v.config.sample_rate)
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
            for c in v.synthesize(text):
                if first is None:
                    first = time.perf_counter()
                # AudioChunk → numpy int16 → convert to float32 for soundfile
                arr = c.audio_int16_array.astype(np.float32) / 32768.0
                chunks.append(arr)
            t_end = time.perf_counter()

            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
            audio_s = float(len(audio) / samplerate)
            if write_wav:
                sf.write(out_path, audio, samplerate)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (first - t0) * 1000 if first else None,
                "gen_s": t_end - t0, "audio_s": audio_s,
                **_meminfo.sample(args.device),
                **(_naq.score(out_path) if write_wav else {"naq": None, "naq_artifact": None, "naq_naturalness": None}),
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
