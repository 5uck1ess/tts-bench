"""VoxCPM2 runner (OpenBMB, zero-shot voice cloning, multilingual, 2B params).

VoxCPM2 is the current recommended release — 2B params, 48kHz output,
30 languages, ~8 GB VRAM. The 0.5B variant is "legacy" per upstream's
own model table and (more importantly) does NOT support cloning via
`reference_wav_path` — only VoxCPM2 does. Both ship in the same `voxcpm`
pip package; the only difference is the HF id passed to `from_pretrained`.

API (voxcpm==latest):
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    wav = model.generate(text="...", cfg_value=2.0, inference_timesteps=10)
    # Cloning (wav-only, no transcript needed):
    wav = model.generate(text="...", reference_wav_path="ref.wav")
    # Returns numpy float32 at model.tts_model.sample_rate.

Device: VoxCPM auto-selects CUDA when available; CPU fallback works but is
unrepresentative (diffusion-style sampling).
"""

import argparse
import json
import sys
import time

import _meminfo


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot voice cloning (no .txt transcript needed).")
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
        import numpy as np
        import soundfile as sf
        import torch
        from voxcpm import VoxCPM

        # VoxCPM2 auto-picks device. If user asked for CPU explicitly,
        # hide CUDA to force the fallback (no clean device param on the class).
        if args.device == "cpu" and torch.cuda.is_available():
            torch.cuda.is_available = lambda: False  # type: ignore[assignment]

        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
        samplerate = int(model.tts_model.sample_rate) if hasattr(model, "tts_model") else 48000
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            if args.reference:
                wav = model.generate(text=text, reference_wav_path=args.reference)
            else:
                wav = model.generate(text=text, cfg_value=2.0, inference_timesteps=10)
            t_end = time.perf_counter()

            arr = np.asarray(wav, dtype=np.float32).reshape(-1)
            audio_s = float(len(arr) / samplerate)
            if write_wav:
                sf.write(out_path, arr, samplerate)

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
