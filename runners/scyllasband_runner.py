"""Scylla's Band runner (ONNX Runtime, fixed voices, predefined-only).

Ten managed voices support en_us, en_gb, es, and it. The harness language
codes map to en_us, es, and it; French is not supported. Output is 24 kHz.

Uses synthesize_stream(), NOT synthesize(). The bundle ships fixed-size graph
exports capped at 640 latent frames, and the single-shot synthesize() path
raises on anything longer:

    ValueError: Predicted 670 latent frames, but this bundle supports at most 640.

Bench prompt 3 (the ~24s Parakeet sentence) trips that on every voice.
synthesize_stream() is the long-form path — it plans the text into chunks,
renders each within the frame cap, and emits `audio_chunk` events we
concatenate. It also yields a real TTFA (time to the first chunk) rather than
the whole-utterance time.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import _meminfo


LANGUAGE_MAP = {
    "en": "en_us",
    "es": "es",
    "it": "it",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Voice ID (e.g. 'gwen') — Scylla's Band doesn't support wav cloning.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    language = LANGUAGE_MAP.get(args.language)
    if not language:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"unsupported Scylla's Band language={args.language}; supported harness languages: en, es, it"}))
        return 1

    try:
        from scyllasband import ScyllasBandRuntime
        import numpy as np
        import soundfile as sf

        repo_root = Path(__file__).resolve().parents[1]
        bundle_dir = Path(os.environ.get(
            "SCYLLASBAND_BUNDLE",
            repo_root / "venvs" / "scyllasband" / "src" / "scyllasband" / "models" / "onnx",
        )).expanduser().resolve()
        voice_id = args.reference or "scylla"
        rt = ScyllasBandRuntime.from_bundle(bundle_dir, backends=["onnx"])
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
            samplerate = None
            for event in rt.synthesize_stream(
                text=text, voice_id=voice_id, language=language,
            ):
                if event.type != "audio_chunk" or event.audio is None:
                    continue
                if first is None:
                    first = time.perf_counter()
                chunks.append(np.asarray(event.audio, dtype=np.float32))
                samplerate = int(event.sample_rate)
            t_end = time.perf_counter()

            if not chunks:
                raise RuntimeError("no audio_chunk events emitted")
            audio = np.concatenate(chunks)
            audio_s = float(len(audio) / samplerate)
            if write_wav:
                sf.write(out_path, audio, samplerate)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (first - t0) * 1000,
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
