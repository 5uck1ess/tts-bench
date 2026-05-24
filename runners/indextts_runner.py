"""IndexTTS-2 runner (Bilibili Index TTS team, zero-shot cloning + emotion control).

Source-clone install (no pip wheel) — see install.ps1 / install.sh. Loads
weights from `IndexTeam/IndexTTS-2` on HuggingFace via huggingface_hub
snapshot_download on first run; cached after.

API (from indextts.infer_v2):
    from indextts.infer_v2 import IndexTTS2
    tts = IndexTTS2(
        cfg_path=f"{model_dir}/config.yaml",
        model_dir=model_dir,
        use_fp16=True,            # FP16 on CUDA for speed/VRAM win
        use_cuda_kernel=False,    # opt-in CUDA kernels, needs compile
        use_deepspeed=False,      # DeepSpeed accel, opt-in
    )
    tts.infer(
        spk_audio_prompt=ref_wav, # wav only, no transcript needed
        text=text,
        output_path=str(out_path),
        verbose=False,
    )

Cloning flavor: wav only (matches ChatterBox, Coqui, VoxCPM). Emotion control
is supported via emo_audio_prompt or emo_text but not exposed by this runner
yet — we just take a single speaker reference. Default-voice path falls back
to bundled jo.wav (EN) / juliette.wav (FR) since IndexTTS-2 is clone-only.

License: Apache 2.0 (model) + Apache 2.0 (code).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo
import _naq


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot voice cloning (no transcript needed).")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    # Default-voice path: borrow a bundled reference (clone-only model).
    repo = Path(__file__).resolve().parent.parent
    if args.reference:
        ref_wav = Path(args.reference)
    else:
        default_ref = {"en": "jo.wav", "fr": "juliette.wav"}.get(args.language, "jo.wav")
        ref_wav = repo / "reference" / default_ref

    if not ref_wav.exists():
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference wav not found: {ref_wav}"}))
        return 1

    try:
        import soundfile as sf
        from huggingface_hub import snapshot_download
        from indextts.infer_v2 import IndexTTS2

        # Download weights (cached after first call).
        model_dir = snapshot_download("IndexTeam/IndexTTS-2")

        tts = IndexTTS2(
            cfg_path=f"{model_dir}/config.yaml",
            model_dir=model_dir,
            use_fp16=(args.device == "cuda"),
            use_cuda_kernel=False,
            use_deepspeed=False,
        )
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            import numpy as np

            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            # output_path=None routes through the "return tuple" branch in
            # IndexTTS2.infer_generator (line 702-708 of infer_v2.py), avoiding
            # its torchaudio.save call. torchaudio.save in torch 2.12+ requires
            # torchcodec which needs FFmpeg shared DLLs (not just ffmpeg.exe) -
            # not available on standard Windows installs.
            result = tts.infer(
                spk_audio_prompt=str(ref_wav),
                text=text,
                output_path=None,
                verbose=False,
            )
            t_end = time.perf_counter()

            # result is (sample_rate, int16_numpy_array). The array is shape
            # (samples, channels) after the upstream .T transpose.
            sr, wav_data = result
            arr = np.asarray(wav_data).astype(np.float32) / 32768.0  # int16 -> [-1, 1]
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.reshape(-1)
            audio_s = float(len(arr) / sr)
            if write_wav:
                sf.write(out_path, arr, sr)

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
