"""IndexTTS-2.5 runner (Bilibili Index TTS team, zero-shot cloning + emotion control).

Successor to the `indextts` (IndexTTS-2) row, but deliberately in its OWN venv
(`venvs/indextts25`): the upstream repo now pins numpy 2.2 / numba 0.63 /
torch 2.8 for 2.5, and upgrading the shared source would silently change the
code the live IndexTTS-2 row was measured on. Source pinned at index-tts
`d9e41aa` (2026-09-30, past the v2.5.0 tag + its perf fixes #813/#816).
Weights `IndexTeam/IndexTTS-2.5` resolved to `c39ce5ba98` when first benched.

API (indextts.infer_v2_5 — same class name as 2.0, different module):
    tts = IndexTTS2(cfg_path=..., model_dir=..., use_bf16=..., device=...)
    tts.infer(spk_audio_prompt=ref_wav, text=text, output_path=None, lang="en")

Gotchas:
- `lang` is REQUIRED and an unknown code silently maps to the "common" token
  (utils/tokenizer.lang_to_token) — French would "work" and produce garbage
  labelled French. So the runner refuses anything outside the upstream list
  (zh/en/ja/es/ar). Prompt 5 (fr) therefore fails cleanly; prompt 6 (es) runs.
- Output is 22.05 kHz (BigVGAN, hard-coded `sampling_rate = 22050`).
- Non-streaming: ttfa_ms == gen_s (whole clip, not a first-chunk latency).
- output_path=None returns (sr, int16 array) and skips save_pcm_wav, same as
  the 2.0 runner.
- Clone-only (no preset voice): no-reference path uses the house
  chris_hemsworth_15s.wav -> NO_PRESET_VOICE, cloning board.

License: bilibili Model Use License Agreement (code + weights) — outputs
usable/publishable; 100M-MAU commercial threshold; no training other
commercial models on it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo

# Upstream-supported languages (README + infer_v2_5 normalizer branches).
SUPPORTED_LANGS = {"zh", "en", "ja", "es", "ar"}


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
    lang = (args.language or "en").lower()
    if lang not in SUPPORTED_LANGS:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"language {lang!r} not supported by IndexTTS-2.5 "
                                   f"(supports {sorted(SUPPORTED_LANGS)})"}))
        return 1

    repo = Path(__file__).resolve().parent.parent
    ref_wav = Path(args.reference) if args.reference else repo / "reference" / "chris_hemsworth_15s.wav"
    if not ref_wav.exists():
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference wav not found: {ref_wav}"}))
        return 1

    try:
        import numpy as np
        import soundfile as sf
        import torch
        from huggingface_hub import snapshot_download
        from indextts.infer_v2_5 import IndexTTS2

        model_dir = snapshot_download("IndexTeam/IndexTTS-2.5")
        tts = IndexTTS2(
            cfg_path=f"{model_dir}/config.yaml",
            model_dir=model_dir,
            use_bf16=(args.device == "cuda"),
            device=args.device,
            use_cuda_kernel=False,
            use_deepspeed=False,
        )
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            torch.manual_seed(1234)
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            result = tts.infer(
                spk_audio_prompt=str(ref_wav),
                text=text,
                output_path=None,
                lang=lang,
                verbose=False,
            )
            t_end = time.perf_counter()

            sr, wav_data = result
            arr = np.asarray(wav_data).astype(np.float32) / 32768.0
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
