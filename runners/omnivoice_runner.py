"""OmniVoice runner (k2-fsa, zero-shot voice cloning, 600+ languages).

Diffusion LM-style TTS with very high RTF on GPU (~0.025x reported). Loads
weights from HuggingFace `k2-fsa/OmniVoice` on first run (auto-download).

API (omnivoice==latest):
    from omnivoice import OmniVoice
    import torch
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda:0",   # or "mps" / "cpu" / "xpu"
        dtype=torch.float16,   # float32 on CPU/MPS for safety
    )
    audio = model.generate(text="hello world")
    # Cloning: pass ref_audio + ref_text (transcript REQUIRED).
    audio = model.generate(text="...", ref_audio="ref.wav", ref_text="transcript of ref.wav")
    # audio: list[np.ndarray (T,)] at 24 kHz mono.

Cloning flavor: wav + transcript (matches NeuTTS Air / F5-TTS).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import _meminfo


SAMPLE_RATE = 24000


def _device_map_from(device: str) -> str:
    return {"cuda": "cuda:0", "mps": "mps", "cpu": "cpu"}.get(device, "cpu")


def _read_ref_transcript(ref_wav: str | None) -> str | None:
    if not ref_wav:
        return None
    txt_path = Path(ref_wav).with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot voice cloning. Needs sibling .txt transcript.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    if args.device == "mps":
        # MPS's default high-watermark ceiling spuriously aborts with "MPS backend
        # out of memory" on the long-form prompt even though real usage is ~1 GB
        # (it over-counts unified-memory "other allocations"). Disabling the
        # watermark lets the allocation proceed and the cell passes. Must be set
        # before torch is imported. NOTE: cloning on mps genuinely exceeds 16 GB
        # and still gets OS-killed — use cpu for omnivoice cloning on a 16 GB Mac.
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    ref_text = _read_ref_transcript(args.reference)
    if args.reference and not ref_text:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference {args.reference} provided but sibling .txt transcript missing"}))
        return 1

    try:
        import numpy as np
        import soundfile as sf
        import torch
        from omnivoice import OmniVoice

        device_map = _device_map_from(args.device)
        dtype = torch.float16 if args.device == "cuda" else torch.float32
        model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice", device_map=device_map, dtype=dtype,
        )
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            if args.reference:
                out = model.generate(text=text, ref_audio=args.reference, ref_text=ref_text)
            else:
                out = model.generate(text=text)
            t_end = time.perf_counter()

            # API returns list[np.ndarray]; concatenate to a single mono track.
            arr = np.concatenate([np.asarray(x).reshape(-1) for x in out]) if isinstance(out, list) else np.asarray(out).reshape(-1)
            arr = arr.astype(np.float32)
            audio_s = float(len(arr) / SAMPLE_RATE)
            if write_wav:
                sf.write(out_path, arr, SAMPLE_RATE)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
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
