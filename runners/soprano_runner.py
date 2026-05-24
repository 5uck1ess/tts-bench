"""Soprano 80M runner (Apache 2.0, predefined voice only, 32 kHz).

API discovered by inspection (soprano-tts==0.2.0, 2026-05-24):
    from soprano import SopranoTTS
    tts = SopranoTTS(device='cuda')   # or 'cpu', 'mps'
    # __init__ runs a warmup infer("Hello world!") automatically
    audio = tts.infer(text)           # returns torch.Tensor, 1-D, CPU, float32, 32 kHz

Model loaded from HF: ekwek/Soprano-1.1-80M (decoder.pth + transformers LLM backbone)
Output sample rate: 32000 Hz (hard-coded in Soprano internals)
Single predefined voice — no voice selection or cloning supported.

Windows/Blackwell note: soprano pip install pulls CPU torch 2.12; reinstall
cu128 torch after (see install.ps1). Runner detects device via torch.cuda / mps.
"""

import argparse
import json
import sys
import time
import warnings

import _meminfo
import _naq

SAMPLE_RATE = 32000


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Ignored — Soprano is predefined-voice only (cloning on roadmap).")
    p.add_argument("--variant", default=None)       # unused
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")      # unused — English only
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    if args.reference:
        print(
            "soprano_runner: --reference ignored (Soprano is predefined-voice only; "
            "zero-shot cloning is on the roadmap but not yet released)",
            file=sys.stderr,
        )

    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    try:
        from soprano import SopranoTTS
        import numpy as np
        import soundfile as sf

        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
        # SopranoTTS.__init__ auto-runs a warmup infer("Hello world!") on load —
        # suppress any stdout noise from that by capturing it via warnings filter.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tts = SopranoTTS(device=device)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            audio_tensor = tts.infer(text)
            t_end = time.perf_counter()

            # infer() returns a 1-D CPU torch.Tensor at 32 kHz
            arr = audio_tensor.cpu().numpy() if hasattr(audio_tensor, "cpu") else audio_tensor
            import numpy as np
            if not hasattr(arr, "__len__"):
                arr = np.asarray(arr, dtype="float32")
            arr = arr.squeeze()

            audio_s = float(len(arr) / SAMPLE_RATE)
            if write_wav:
                sf.write(out_path, arr, SAMPLE_RATE)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,   # non-streaming: TTFA == gen_s
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
