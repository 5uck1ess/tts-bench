"""DramaBox runner (Resemble AI, LTX-2 Community License, expressive dialogue TTS).

DramaBox is an IC-LoRA fine-tune of the LTX-2.3 3.3B audio-only DiT (flow-matching).
The PROMPT controls speaker identity, emotion, delivery, laughs/sighs/pauses; an
optional 10s+ voice reference clones the target timbre. Prompt grammar is
`<speaker description>, "<dialogue>" <action direction> "<more dialogue>"` — so the
bench's plain prompts are wrapped in a neutral speaker description + quotes here
(`_dramatize`), which is what the model expects; a pre-quoted prompt is passed through.

Upstream is a SOURCE repo, NOT a pip package: code is imported from the cloned tree
(venvs/dramabox/src, which contains the `src/` package), so we add that dir to sys.path
and `from src.inference_server import TTSServer`. install.ps1/.sh clone it.

API (upstream README Quick Start):
    from src.inference_server import TTSServer
    server = TTSServer(device="cuda")            # downloads weights on first run
    server.generate_to_file(prompt='A woman says, "Hello."', output="out.wav",
                            voice_ref="ref.wav",  # optional, 10s+; omit for prompt-only voice
                            cfg_scale=2.5, stg_scale=1.5, duration_multiplier=1.1, seed=0)

Footprint: ~24 GB VRAM peak. Weights ~16 GB on first run (6.6 GB DiT + 1.9 GB audio
components + ~8 GB unsloth/gemma-3-12b-it-bnb-4bit text encoder, all auto-downloaded).
CUDA-only in harness.py (fits Win-5090 32 GB; too tight for a 24 GB card). Watermarked
(Resemble Perth). License: LTX-2 Community License (non-commercial; benchmarking not
restricted) — weights + outputs usable for this bench.
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import _meminfo


REPO_ROOT = Path(__file__).resolve().parent.parent
DRAMABOX_SRC = REPO_ROOT / "venvs" / "dramabox" / "src"
DEFAULT_REF = REPO_ROOT / "reference" / "chris_hemsworth_15s.wav"

# Generation params from the upstream README defaults.
CFG_SCALE = 2.5
STG_SCALE = 1.5
DURATION_MULTIPLIER = 1.1
SEED = 0  # fixed so the prompt-driven default voice is stable across prompts

_REAL_STDOUT = sys.stdout


def _emit(obj) -> None:
    print(json.dumps(obj), file=_REAL_STDOUT, flush=True)


def _dramatize(text: str) -> str:
    """DramaBox expects `<speaker description>, "<dialogue>"`. Plain bench prompts get a
    neutral wrapper; an already-quoted prompt is assumed pre-formatted and passes through."""
    if '"' in text:
        return text
    return f'A person speaks clearly, "{text}"'


def _trim_silence(y, sr, thresh_db=-40.0, pad_s=0.1):
    """DramaBox's duration estimator over-allocates for short lines, so a clip can start
    with multiple seconds of dead air. Trim leading/trailing silence (energy below
    thresh_db relative to peak), keeping a small natural pad. Interior pauses are left
    intact (only the first/last voiced sample bound the cut)."""
    import numpy as np

    y = np.asarray(y, dtype=np.float32)
    peak = float(np.abs(y).max())
    if peak < 1e-6:
        return y
    thr = peak * (10.0 ** (thresh_db / 20.0))
    voiced = np.where(np.abs(y) > thr)[0]
    if len(voiced) == 0:
        return y
    a = max(0, int(voiced[0]) - int(pad_s * sr))
    b = min(len(y), int(voiced[-1]) + int(pad_s * sr))
    return y[a:b]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot voice cloning (10s+; no transcript needed).")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        _emit({"ok": False, "run_index": 0,
               "error": "either --stdin or both --text and --out are required"})
        return 1

    cloning = args.reference is not None
    ref_wav = Path(args.reference) if args.reference else DEFAULT_REF
    if cloning and not ref_wav.exists():
        _emit({"ok": False, "run_index": 0, "error": f"reference wav not found: {ref_wav}"})
        return 1

    try:
        import torch

        if args.device != "cuda":
            _emit({"ok": False, "run_index": 0,
                   "error": f"DramaBox is CUDA-only here (~24 GB VRAM); device={args.device}"})
            return 1
        if not torch.cuda.is_available():
            _emit({"ok": False, "run_index": 0, "error": "CUDA requested but not available"})
            return 1

        # Source repo, not a package: the cloned tree contains the `src/` package, so add
        # the repo root to sys.path. Importing inference_server self-adds repo/src + repo/ltx2
        # to sys.path, which makes the bare `model_downloader` import below resolve.
        if str(DRAMABOX_SRC) not in sys.path:
            sys.path.insert(0, str(DRAMABOX_SRC))
        with contextlib.redirect_stdout(sys.stderr):
            from src.inference_server import TTSServer
            from model_downloader import get_all_paths
            # Fetch DiT + audio-components (+ silence latent + Gemma) into the HF cache and
            # point the server at them — TTSServer()'s default paths are local `models/`
            # files that don't exist for a pip-style install (mirrors the repo's app.py).
            paths = get_all_paths()
            server = TTSServer(
                checkpoint=paths["transformer"],          # dramabox-dit-v1.safetensors
                full_checkpoint=paths["audio_components"],  # dramabox-audio-components.safetensors
                gemma_root=paths["gemma_root"],
                device="cuda", dtype="bf16",
                compile_model=False,  # torch.compile/Triton is brittle on Windows; off for clean timing
                bnb_4bit=True,
            )
    except Exception as e:
        _emit({"ok": False, "run_index": 0, "error": f"load failed: {type(e).__name__}: {e}"})
        return 1

    voice_ref = str(ref_wav) if cloning else None

    def _one(text: str, out_path: str, run_index: int, write_wav: bool) -> bool:
        try:
            import soundfile as sf

            # generate_to_file always writes; for warm runs (i>0) write to a throwaway.
            target = str(out_path) if write_wav else str(Path(out_path).with_suffix(".warm.tmp.wav"))

            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                server.generate_to_file(
                    prompt=_dramatize(text),
                    output=target,
                    voice_ref=voice_ref,
                    cfg_scale=CFG_SCALE,
                    stg_scale=STG_SCALE,
                    duration_multiplier=DURATION_MULTIPLIER,
                    seed=SEED,
                )
            t_end = time.perf_counter()

            # Trim the duration-estimator's leading/trailing dead air; audio_s reflects
            # the trimmed speech so RTF isn't inflated by silence.
            clip, sr = sf.read(target)
            if getattr(clip, "ndim", 1) > 1:
                clip = clip.mean(axis=1)
            clip = _trim_silence(clip, sr)
            audio_s = float(len(clip)) / float(sr)
            if write_wav:
                sf.write(target, clip, sr)
            else:
                with contextlib.suppress(OSError):
                    Path(target).unlink()

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
