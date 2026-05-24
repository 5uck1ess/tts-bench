"""Dia runner (Nari Labs, Apache 2.0, 1.6B params, dialogue-focused TTS).

Dia is a text-to-dialogue model that excels at multi-speaker dialogue with
non-verbal cues ((laughs), (coughs), etc.). Uses [S1]/[S2] speaker tags.

API:
    from dia.model import Dia, DEFAULT_SAMPLE_RATE
    model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")
    output = model.generate(
        text,                        # must include [S1]/[S2] tags
        audio_prompt=ref_wav_path,   # optional: path for voice cloning
        use_torch_compile=False,
        cfg_scale=3.0,
        temperature=1.8,
        top_p=0.90,
        cfg_filter_top_k=50,
    )  # -> list of NumPy float32 arrays at DEFAULT_SAMPLE_RATE (44100 Hz)

Default voice (no --reference): seed fixed at 42 for bench reproducibility;
    text is wrapped with "[S1] <text>" automatically.
Voice cloning (--reference provided): audio_prompt=ref_wav_path; the text
    passed to generate() must be prefixed with the reference transcript
    (read from <reference>.txt sibling file, required). The final text to
    generate is appended with "[S1] " so Dia outputs only the target content.

Non-streaming: TTFA == gen_s (full tensor returned after generation completes).
Sample rate: 44100 Hz. Works on CUDA; CPU is very slow (not recommended).

Install gotchas (Windows/Blackwell RTX 5090):
- Dia's pip deps pull cu126 torch 2.6; must reinstall cu128 torch AFTER
  `pip install dia`. install.ps1 handles this with --reinstall pattern.
- Issue #26: dtype mismatch on older torch (2.6). cu128 torch 2.11 resolves it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo
import _naq


def _read_ref_transcript(ref_wav: str) -> str | None:
    """Read transcript from sibling .txt file next to the reference wav."""
    txt_path = Path(ref_wav).with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for voice cloning. Requires sibling .txt transcript.")
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
                          "error": f"Dia is English-focused; got language={args.language}"}))
        return 1

    # Validate cloning: .txt transcript is required alongside --reference
    ref_transcript = None
    if args.reference:
        ref_transcript = _read_ref_transcript(args.reference)
        if ref_transcript is None:
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": (
                                  f"Voice cloning requires a transcript file at "
                                  f"{Path(args.reference).with_suffix('.txt')} — "
                                  "create it with the exact text spoken in the reference wav."
                              )}))
            return 1

    try:
        import torch
        import numpy as np
        import soundfile as sf
        from dia.model import Dia, DEFAULT_SAMPLE_RATE

        device = args.device if args.device in ("cpu", "cuda") else "cuda"

        if device == "cuda" and not torch.cuda.is_available():
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": "CUDA requested but not available"}))
            return 1

        model = Dia.from_pretrained(
            "nari-labs/Dia-1.6B-0626",
            compute_dtype="float16",
            device=torch.device(device),
        )
        samplerate = DEFAULT_SAMPLE_RATE  # 44100

    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text: str, out_path: str, run_index: int, write_wav: bool) -> bool:
        try:
            import torch

            _meminfo.reset_peak(args.device)

            # Build the text fed to Dia:
            # - Default voice: wrap with [S1] tag; fix seed for reproducibility.
            # - Cloning: prefix with reference transcript then target with [S1].
            if args.reference and ref_transcript:
                # Format: "<ref_transcript> [S1] <target_text>"
                # Dia returns only the audio past the audio_prompt boundary.
                dia_text = f"{ref_transcript} [S1] {text}"
            else:
                # Wrap bare text in [S1] tag for proper Dia output
                dia_text = f"[S1] {text}"
                torch.manual_seed(42)

            t0 = time.perf_counter()
            output = model.generate(
                dia_text,
                audio_prompt=args.reference,   # None for default voice
                use_torch_compile=False,
                verbose=False,
                cfg_scale=3.0,
                temperature=1.8,
                top_p=0.90,
                cfg_filter_top_k=50,
            )
            t_end = time.perf_counter()

            # output is a list of NumPy float32 arrays, one per batch item
            arr = output[0] if isinstance(output, list) else output
            if hasattr(arr, "cpu"):
                arr = arr.cpu().numpy()
            arr = arr.squeeze()

            audio_s = float(len(arr) / samplerate)
            gen_s = t_end - t0

            if write_wav:
                sf.write(out_path, arr, samplerate)

            print(json.dumps({
                "ok": True,
                "run_index": run_index,
                "ttfa_ms": gen_s * 1000,   # non-streaming: TTFA == gen_s
                "gen_s": gen_s,
                "audio_s": audio_s,
                **_meminfo.sample(args.device),
                **(_naq.score(out_path) if write_wav else
                   {"naq": None, "naq_artifact": None, "naq_naturalness": None}),
            }), flush=True)
            return True

        except Exception as e:
            print(json.dumps({
                "ok": False,
                "run_index": run_index,
                "error": f"{type(e).__name__}: {e}",
            }), flush=True)
            return False

    if args.stdin:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                print(json.dumps({"ok": False, "run_index": 0,
                                  "error": "invalid JSON on stdin"}), flush=True)
                continue
            _one(req["text"], req["out"], req.get("run_index", 0), True)
    else:
        for i in range(args.runs):
            write = (i == 0)
            _one(args.text, args.out, i, write)

    return 0


if __name__ == "__main__":
    sys.exit(main())
