"""ChatterBox-TTS runner (MIT, zero-shot voice cloning, diffusion-based).

API (chatterbox-tts==0.x):
    from chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained(device='cpu' | 'cuda')
    audio = m.generate(text, audio_prompt_path=ref_wav_path, ...)  # torch.Tensor [1, N]

Voice cloning: pass `audio_prompt_path` (single wav, no transcript needed).
Watermarks output via the Perth implicit watermarker.

Non-streaming: returns the full audio tensor after all 1000 sampling steps.
TTFA == gen_s (no incremental output).

CPU is unrepresentative — ChatterBox is GPU-targeted (diffusion, 1000 steps).
Bench it on CPU for completeness, expect <0.2x RTF.

Install gotchas:
- needs `setuptools<80` for pkg_resources (which perth's watermarker imports
  via perth.perth_net.__init__). install.ps1 / install.sh handle this.
"""

import argparse
import json
import sys
import time


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

    if args.language != "en":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"ChatterBox base model is EN-only; got language={args.language}. Try ChatterboxMultilingualTTS for other langs."}))
        return 1

    try:
        from chatterbox.tts import ChatterboxTTS
        import numpy as np
        import soundfile as sf

        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
        m = ChatterboxTTS.from_pretrained(device=device)
        samplerate = int(m.sr) if hasattr(m, "sr") else 24000
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            t0 = time.perf_counter()
            audio = m.generate(text, audio_prompt_path=args.reference)
            t_end = time.perf_counter()

            # audio is torch.Tensor [1, N] — convert to numpy mono
            arr = audio.squeeze().cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio).squeeze()
            audio_s = float(len(arr) / samplerate)
            if write_wav:
                sf.write(out_path, arr, samplerate)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,  # non-streaming
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
