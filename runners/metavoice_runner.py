"""MetaVoice-1B v0.1 runner (metavoiceio/metavoice-src, Apache-2.0, 48 kHz).

Source-clone install (no pip wheel) — see install.sh. Weights
(`metavoiceio/metavoice-1B-v0.1`) download via huggingface_hub on first
TTS() init; cached after. Abandoned repo — torch is pinned to 2.2.1 (the code
calls torch._inductor.config.fx_graph_cache, added in torch 2.2; the repo's own
requirements.txt pin of 2.1.0 is stale) and numpy<2.

API (from fam.llm.fast_inference):
    from fam.llm.fast_inference import TTS
    tts = TTS()                                  # compile=True hardcoded (~slow first call)
    wav_path = tts.synthesise(text, spk_ref_path)  # returns path to a 48 kHz .wav

Cloning flavor: wav only (no transcript), but the speaker reference must be
>= 30 s of audio. The bundled jo/juliette/chris_hemsworth_15s clips are all
< 30 s, so the default-voice path uses reference/chris_hemsworth.wav (67 s);
a user --reference must likewise be >= 30 s.

ffmpeg@6 note: audiocraft -> av==11.0.0 dynamically links libav*.so.60 (FFmpeg
6). On Linux those come from `brew install ffmpeg@6`; we preload them with
ctypes(RTLD_GLOBAL) so the runner works under bench.py without LD_LIBRARY_PATH.

License: Apache-2.0 (code + model).
"""

import argparse
import ctypes
import glob
import json
import os
import sys
import time
from pathlib import Path

import _meminfo
import _naq

# Silence posthog telemetry (DISABLE_TELEMETRY alone is not enough).
os.environ.setdefault("DISABLE_TELEMETRY", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# Preload the brew ffmpeg@6 shared libs so `av` resolves libav*.so.60 at import
# without a process-level LD_LIBRARY_PATH (bench.py spawns us with a bare env).
_FFMPEG6_LIB = "/home/linuxbrew/.linuxbrew/opt/ffmpeg@6/lib"
if sys.platform.startswith("linux") and os.path.isdir(_FFMPEG6_LIB):
    for _stem in ("libavutil", "libswresample", "libswscale", "libavcodec",
                  "libavformat", "libavfilter", "libavdevice"):
        for _so in sorted(glob.glob(f"{_FFMPEG6_LIB}/{_stem}.so.*")):
            try:
                ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot cloning. MUST be >= 30 s of audio.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    # Default-voice path: MetaVoice needs >= 30 s ref, so the short bundled
    # clips (jo 13 s, juliette 8 s, chris_hemsworth_15s) won't do — use the
    # 67 s chris_hemsworth.wav.
    repo = Path(__file__).resolve().parent.parent
    ref_wav = Path(args.reference) if args.reference else repo / "reference" / "chris_hemsworth.wav"
    if not ref_wav.exists():
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference wav not found (need >=30s): {ref_wav}"}))
        return 1

    try:
        import soundfile as sf
        from fam.llm.fast_inference import TTS

        tts = TTS(telemetry_origin=None)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            wav_path = tts.synthesise(text=text, spk_ref_path=str(ref_wav))
            t_end = time.perf_counter()

            data, sr = sf.read(wav_path)
            audio_s = float(len(data) / sr)
            if write_wav:
                sf.write(out_path, data, sr)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
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
