"""PiVo-1.4M runner (Apache-2.0 code + acoustic weights, BSD vocoder; English; no cloning).

PiVo (Empero Research) is a complete TTS in 1,401,562 parameters: a 584,965-param
acoustic model predicting FARGAN features plus an 816,597-param FARGAN-derived vocoder,
16 kHz mono, NumPy + ONNX Runtime only (CPUExecutionProvider hard-coded upstream, so
CPU-only). Two synthetic voices, Pi and Vo; the runner uses Pi, the upstream default.

There is no pip package: inference code (`pivo/`) ships inside the HF repo. The runner
snapshot-downloads the repo at a pinned revision into venvs/pivo/model-<rev>/ (a plain
directory — see the local_dir note in main()) and imports from it.
`PiVo(...)` verifies every file's SHA256 against model_manifest.json at load.

    revision 515d84fc3420da6ae138f1b81017fcbc24d622a8 (main on 2026-11-10)

OUT-OF-VOCABULARY WORDS FAIL BY DESIGN. The frontend is the CMU dictionary plus number
expansion; any word not in it raises ValueError instead of guessing. Canonical prompt 3
fails on "librispeech" / "vthree" (its normaliser's spelling of "V3") and prompt 4 on
"pytest" / "py" / "verbose". Upstream's remedy is per-call ARPAbet `overrides=`.
Deliberately NOT supplied here: hand-writing pronunciations for the bench prompts would
tune the row to the test set. Those cells publish as failures (Tym, 2026-11-10).

Stock settings: threads=1 (upstream default), speed 1.0. Deterministic.

Streaming: `iter_audio` runs the encoder (all durations) up front, decodes features in
16-frame chunks (upstream default chunk_frames), and yields one 10 ms PCM frame per
vocoder step. TTFA is measured at the first yielded 10 ms frame, so it is a real
time-to-first-audio (~2 ms warm on the 9950X3D), not gen_s.
``--reference`` is accepted and ignored: PiVo cannot clone.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


REPO_ID = "empero-ai/PiVo-1.4M"
REVISION = "515d84fc3420da6ae138f1b81017fcbc24d622a8"
VOICE = "Pi"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Ignored — PiVo preset voices cannot clone.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    if args.language != "en":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"PiVo is English-only, got language={args.language}"}))
        return 1
    if args.device != "cpu":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"PiVo is CPU-only upstream, got device={args.device}"}))
        return 1

    try:
        import numpy as np
        import soundfile as sf
        from huggingface_hub import snapshot_download

        # local_dir, not the HF cache: the cache stores files as symlinks into blobs/,
        # and PiVo's load-time manifest check rejects symlinked assets as "unsafe".
        local = Path(__file__).resolve().parents[1] / "venvs" / "pivo" / f"model-{REVISION[:7]}"
        model_dir = snapshot_download(REPO_ID, revision=REVISION, local_dir=str(local))
        sys.path.insert(0, model_dir)
        from pivo import PiVo

        model = PiVo(model_dir, threads=1)
        sr = model.sample_rate
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            ttfa = None
            chunks = []
            for chunk in model.iter_audio(text, voice=VOICE):
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                chunks.append(chunk)
            t_end = time.perf_counter()
            audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

            audio_s = float(len(audio) / sr)
            if write_wav:
                sf.write(out_path, audio, sr)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (ttfa if ttfa is not None else t_end - t0) * 1000,
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
