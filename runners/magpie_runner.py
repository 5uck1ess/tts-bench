"""Magpie-TTS Multilingual 357M runner (NVIDIA, predefined voices, 9 langs).

Predefined-voice only (no zero-shot cloning) — the 357M release uses fixed
speaker context embeddings. Larger Magpie variants do zero-shot but aren't
shipped under this checkpoint name.

Languages: en, es, de, it, vi, zh, fr, hi, ja.
License: NVIDIA Open Model License. HF model is gated — needs `hf auth login`.

Install note: this venv installs `nemo_toolkit` core *without* the `[tts]` extra
so we sidestep `nemo_text_processing` → `pynini`, which has no Windows build.
The runner calls `do_tts(apply_TN=False)` so the text-normalization code path
(the only consumer of pynini at inference time) is never hit.

API (nemo_toolkit >= 2.x):
    from nemo.collections.tts.models import MagpieTTSModel
    model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model = model.to(device).eval()
    audio, audio_len = model.do_tts(
        transcript,
        language="en",
        apply_TN=False,     # MUST be False on Windows w/o pynini
        speaker_index=0,
    )
    # audio: torch.Tensor [B, T] at model.sample_rate (22050 Hz typical)
"""

import argparse
import json
import sys
import time


# Magpie 357M has a small set of built-in speakers. 0 is a reasonable default
# for English; pick higher indices if you want other voices.
DEFAULT_SPEAKER_INDEX = 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Ignored — Magpie 357M uses predefined voices, no cloning.")
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
        from nemo.collections.tts.models import MagpieTTSModel

        device = args.device if args.device in ("cpu", "cuda") else "cpu"
        model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
        model = model.to(device).eval()
        samplerate = int(getattr(model, "sample_rate", 22050))
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                audio, _ = model.do_tts(
                    text,
                    language=args.language,
                    apply_TN=False,
                    speaker_index=DEFAULT_SPEAKER_INDEX,
                )
            t_end = time.perf_counter()

            arr = audio.squeeze().detach().cpu().numpy().astype(np.float32)
            audio_s = float(arr.shape[-1] / samplerate)
            if write_wav:
                sf.write(out_path, arr, samplerate)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
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
