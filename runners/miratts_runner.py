"""MiraTTS runner (Yatharth Sharma, MIT, LLM-TTS + BiCodec, 48 kHz, zero-shot cloning).

MiraTTS (PyPI project name `FastNeuTTS`) is a small (0.5B) autoregressive LLM-TTS that
clones a voice from a reference wav (wav only, no transcript) and decodes through the
author's FastBiCodec, upsampled to 48 kHz by FlashSR. High quality at low VRAM (~6 GB).

API (from mira/model.py):
    from mira.model import MiraTTS
    model = MiraTTS("YatharthS/MiraTTS")        # builds an lmdeploy TurboMind pipeline
    ctx   = model.encode_audio("ref.wav")        # codec.encode -> context tokens
    audio = model.generate("text", ctx)          # 1-D numpy float @ 48 kHz

LMDEPLOY / DEVICE NOTE: the model class hard-imports `lmdeploy` and constructs a
`TurbomindEngineConfig` — there is NO transformers fallback, so this is CUDA-only.
lmdeploy supports Linux + Windows, but its documented GPU support stops at Ada Lovelace
(sm89, 40-series); the RTX 5090 is Blackwell (sm120), which the prebuilt TurboMind wheel
may not ship kernels for. So the clean install path is the Linux-3090 (Ampere sm86,
supported); the Win-5090 is a stretch (may need lmdeploy built from source with sm120).
CPU/MPS cells fail cleanly here rather than crashing the bench.

STDOUT NOTE: the bench protocol is JSON-lines on stdout, but lmdeploy/TurboMind print
load + progress banners to stdout. We save the real stdout up front, emit every JSON row
through it, and redirect Python-level stdout -> stderr while loading and generating so
that chatter can't corrupt the protocol. (The harness parser also skips non-`{` lines,
so any C-level fd writes that slip through are tolerated.)

License: package is MIT (pyproject `License :: OSI Approved :: MIT License`). The HF
weights aren't separately tagged; treat as MIT-permissive for now.
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import _meminfo


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REF = REPO_ROOT / "reference" / "chris_hemsworth_15s.wav"

SAMPLERATE = 48_000

# Real stdout handle captured before any redirect — JSON rows always go here.
_REAL_STDOUT = sys.stdout


def _emit(obj) -> None:
    print(json.dumps(obj), file=_REAL_STDOUT, flush=True)


def _to_mono_f32(audio):
    """Coerce MiraTTS's decode output (numpy/torch/list) to a 1-D float32 numpy array."""
    import numpy as np
    try:
        import torch
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1) if 1 in arr.shape else arr.mean(axis=0)
    return arr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot voice cloning (wav only, no transcript).")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        _emit({"ok": False, "run_index": 0,
               "error": "either --stdin or both --text and --out are required"})
        return 1

    ref_wav = Path(args.reference) if args.reference else DEFAULT_REF
    if not ref_wav.exists():
        _emit({"ok": False, "run_index": 0,
               "error": f"reference wav not found: {ref_wav}"})
        return 1

    try:
        import torch

        if args.device != "cuda":
            _emit({"ok": False, "run_index": 0,
                   "error": f"MiraTTS is CUDA-only (lmdeploy/TurboMind); device={args.device}"})
            return 1
        if not torch.cuda.is_available():
            _emit({"ok": False, "run_index": 0,
                   "error": "CUDA requested but not available"})
            return 1

        # Load + reference-encode under stdout->stderr redirect (lmdeploy banners).
        with contextlib.redirect_stdout(sys.stderr):
            from mira.model import MiraTTS
            model = MiraTTS("YatharthS/MiraTTS")
            # Pin a fixed sampling seed for cross-run/cross-rig determinism, mirroring
            # echo's rng_seed=0. set_params() doesn't expose a seed, so rebuild the
            # GenerationConfig directly; fall back to the default if the field is absent.
            try:
                from lmdeploy import GenerationConfig
                model.gen_config = GenerationConfig(
                    top_p=0.95, top_k=50, temperature=0.8, max_new_tokens=1024,
                    repetition_penalty=1.2, min_p=0.05, do_sample=True, random_seed=0,
                )
            except Exception as e:
                # Best-effort: this lmdeploy build may not accept random_seed. Keep the
                # model's default gen_config (sampling stays nondeterministic). Warn on
                # stderr only — stdout is the JSON protocol channel.
                print(f"miratts: could not pin random_seed ({e}); using default "
                      "gen_config", file=sys.stderr)
            context_tokens = model.encode_audio(str(ref_wav))
    except Exception as e:
        _emit({"ok": False, "run_index": 0,
               "error": f"load failed: {type(e).__name__}: {e}"})
        return 1

    def _one(text: str, out_path: str, run_index: int, write_wav: bool) -> bool:
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                audio = model.generate(text, context_tokens)
            t_end = time.perf_counter()

            clip = _to_mono_f32(audio)
            audio_s = float(len(clip) / SAMPLERATE)

            if write_wav:
                import soundfile as sf
                sf.write(str(out_path), clip, SAMPLERATE)

            _emit({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
                **_meminfo.sample(args.device),
            })
            return True
        except Exception as e:
            _emit({"ok": False, "run_index": run_index,
                   "error": f"{type(e).__name__}: {e}"})
            return False

    if args.stdin:
        idx = 0
        _emit({"ready": True})
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError as e:
                _emit({"ok": False, "run_index": idx, "error": f"json parse: {e}"})
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
