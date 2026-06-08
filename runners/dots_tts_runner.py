"""dots.tts runner (rednote-hilab, Apache-2.0, 2B fully-continuous AR TTS, 48 kHz).

dots.tts pairs a semantic encoder + LLM + autoregressive flow-matching acoustic head
over a 48 kHz AudioVAE (no discrete codec tokens). Zero-shot voice cloning from a
reference wav (+ its transcript for higher-quality continuation cloning). It's a pure
cloning model (no voice of its own), so per the bench convention a no-reference run
clones the bundled chris_hemsworth_15s.wav for the "default voice" lens; a user
--reference overrides it -> can_clone=True, both lenses populated.
Multilingual (24 langs incl. en/fr) so it runs the French prompt too.

Upstream is a pip PACKAGE (`dots_tts`), installed editable from the cloned tree
(venvs/dots_tts/src). install.sh/.ps1 clone + `uv pip install -e`, so the import below
resolves with no sys.path hacking.

API (upstream README quick start):
    from dots_tts.runtime import DotsTtsRuntime
    rt = DotsTtsRuntime.from_pretrained("/path/to/checkpoint", precision="bfloat16")
    result = rt.generate(text="...", prompt_audio_path="ref.wav",   # both optional
                         prompt_text="ref transcript",              # for continuation clone
                         num_steps=10, guidance_scale=1.0)
    sf.write(out, result["audio"].float().cpu().squeeze().numpy(), result["sample_rate"])

Checkpoint: rednote-hilab/dots.tts-soar (the SCA flagship — best quality; the issue's
"best TTS I've ever heard" is this one). Snapshot-downloaded to the HF cache on first
run. Footprint: 2B bf16 backbone + AudioVAE; CUDA-only here (bf16, Ampere/3090 ok).
License: Apache-2.0 (weights + outputs free to publish on the leaderboard).
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

# rednote-hilab/dots.tts-soar = the SCA flagship (highest quality). dots.tts-mf is the
# MeanFlow-distilled fast variant (fewer steps); -base is the raw pretrain. Use soar.
CHECKPOINT = "rednote-hilab/dots.tts-soar"

# Sampler params from the upstream CLI defaults (src/dots_tts/cli.py): guidance_scale
# 1.2 (the runtime default; the README quick-start's 1.0 is below it), num_steps 10,
# speaker_scale 1.5 (runtime default, not overridden). SEED matters a LOT here: like the
# CLI/gradio, we seed_everything(SEED) before each generate — UNSEEDED the AR sampler hits
# early end-of-audio and emits a ~0.3s fragment instead of the full utterance.
NUM_STEPS = 10
GUIDANCE_SCALE = 1.2
SEED = 42

_REAL_STDOUT = sys.stdout


def _emit(obj) -> None:
    print(json.dumps(obj), file=_REAL_STDOUT, flush=True)


def _read_ref_transcript(ref_path):
    """dots.tts does higher-quality continuation cloning when given the reference's
    transcript (prompt_text). The bench keeps it in a sibling <ref>.txt; if absent we
    fall back to x-vector-only cloning (prompt_audio alone, prompt_text=None)."""
    if not ref_path:
        return None
    txt = Path(ref_path).with_suffix(".txt")
    if txt.exists():
        return txt.read_text(encoding="utf-8").strip()
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav for zero-shot cloning; sibling .txt transcript used if present.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        _emit({"ok": False, "run_index": 0,
               "error": "either --stdin or both --text and --out are required"})
        return 1

    # dots.tts is a pure zero-shot cloning model (no voice of its own). Per the bench
    # convention (README "Predefined vs Cloning"), a no-reference run falls back to the
    # bundled chris_hemsworth_15s.wav, so the "default voice" is a clone of that clip and
    # is reproducible across prompts. A user --reference overrides it.
    ref_wav = Path(args.reference) if args.reference else DEFAULT_REF
    if not ref_wav.exists():
        _emit({"ok": False, "run_index": 0, "error": f"reference wav not found: {ref_wav}"})
        return 1
    ref_text = _read_ref_transcript(str(ref_wav))

    try:
        import torch

        if args.device != "cuda":
            _emit({"ok": False, "run_index": 0,
                   "error": f"dots.tts is CUDA-only here (bf16 backbone); device={args.device}"})
            return 1
        if not torch.cuda.is_available():
            _emit({"ok": False, "run_index": 0, "error": "CUDA requested but not available"})
            return 1

        with contextlib.redirect_stdout(sys.stderr):
            from dots_tts.runtime import DotsTtsRuntime
            from dots_tts.utils.util import seed_everything
            runtime = DotsTtsRuntime.from_pretrained(CHECKPOINT, precision="bfloat16")
    except Exception as e:
        _emit({"ok": False, "run_index": 0, "error": f"load failed: {type(e).__name__}: {e}"})
        return 1

    def _one(text: str, out_path: str, run_index: int, write_wav: bool) -> bool:
        try:
            import soundfile as sf

            gen_kwargs = {"text": text, "prompt_audio_path": str(ref_wav),
                          "num_steps": NUM_STEPS, "guidance_scale": GUIDANCE_SCALE}
            if ref_text:  # continuation cloning; without it, x-vector-only from the wav
                gen_kwargs["prompt_text"] = ref_text

            _meminfo.reset_peak(args.device)
            with contextlib.redirect_stdout(sys.stderr):
                seed_everything(SEED)  # without this the sampler early-stops to ~0.3s
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                result = runtime.generate(**gen_kwargs)
            t_end = time.perf_counter()

            audio = result["audio"].float().cpu().squeeze().numpy()
            sr = int(result["sample_rate"])
            audio_s = float(len(audio)) / float(sr)
            if write_wav:
                sf.write(str(out_path), audio, sr)

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
