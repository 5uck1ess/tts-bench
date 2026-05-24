"""ChatterBox-TTS runner — handles two variants under the same venv.

Variant dispatch:
    --variant absent OR --variant "base"
        → Chatterbox 1.2B (ResembleAI/chatterbox, Llama-based, diffusion, 1000 steps)
          Slow on CPU (<0.2x RTF). Zero-shot cloning via audio_prompt_path.
    --variant "turbo"
        → Chatterbox Turbo (ResembleAI/chatterbox-turbo, GPT2-based, AR, ~744M params)
          Much faster than base; has a bundled default voice (conds.pt), also supports
          zero-shot cloning via reference wav.

=== base path (1.2B) ===
API (chatterbox-tts 0.x):
    from chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained(device='cpu' | 'cuda')
    audio = m.generate(text, audio_prompt_path=ref_wav_path, ...)  # torch.Tensor [1, N]

Voice cloning: pass `audio_prompt_path` (single wav, no transcript needed).
Watermarks output via the Perth implicit watermarker.
Non-streaming: returns the full audio tensor after all 1000 sampling steps.
TTFA == gen_s (no incremental output).

=== turbo path (~744M) ===
Weights: ResembleAI/chatterbox-turbo (MIT, same as base).
Architecture: GPT2-medium T3 backbone + S3Gen vocoder.
Tokenizer: shared with base (ResembleAI/chatterbox tokenizer.json).
Inference: T3.inference_turbo() — AR generation, no diffusion steps.
Default voice: bundled conds.pt in the turbo checkpoint.
Cloning: supported — prepare_conditionals() still works for custom refs.

Install gotchas:
- needs `setuptools<80` for pkg_resources (which perth's watermarker imports
  via perth.perth_net.__init__). install.ps1 / install.sh handle this.
"""

import argparse
import json
import sys
import time

import _meminfo
import _naq


def _load_base(device):
    """Load Chatterbox 1.2B (base). Returns (model, samplerate)."""
    from chatterbox.tts import ChatterboxTTS

    m = ChatterboxTTS.from_pretrained(device=device)
    samplerate = int(m.sr) if hasattr(m, "sr") else 24000
    return m, samplerate


def _load_turbo(device):
    """Load Chatterbox Turbo (~744M, GPT2-based AR model).

    Uses ChatterboxTurboTTS.from_pretrained() which downloads from
    ResembleAI/chatterbox-turbo via snapshot_download (cached after first run).
    Includes a bundled default voice (conds.pt) and supports zero-shot cloning
    via audio_prompt_path — same generate() signature as the base model.

    Returns (model, samplerate).
    """
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    m = ChatterboxTurboTTS.from_pretrained(device=device)
    samplerate = int(m.sr) if hasattr(m, "sr") else 24000
    return m, samplerate


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot voice cloning (no .txt transcript needed).")
    p.add_argument("--variant", default=None,
                   help="'base' (default, 1.2B Llama diffusion) or 'turbo' (~744M GPT2 AR).")
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
                          "error": f"ChatterBox (base + turbo) is EN-only; got language={args.language}."}))
        return 1

    variant = (args.variant or "base").lower()
    use_turbo = variant == "turbo"

    try:
        import numpy as np
        import soundfile as sf

        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"

        if use_turbo:
            m, samplerate = _load_turbo(device)
        else:
            m, samplerate = _load_base(device)

    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()

            # Both base and turbo share the same generate() signature:
            # m.generate(text, audio_prompt_path=ref_or_None)
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
