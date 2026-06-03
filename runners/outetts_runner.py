"""OuteTTS runner (edwko / OuteAI, CC-BY-NC-SA-4.0 + Llama-3.2, ~1B, DAC, cloning + presets).

OuteTTS 1.0 (Llama-3.2-1B backbone) is one of the few bench models that does BOTH:
  - default voice: a library of preset speakers (load_default_speaker), and
  - zero-shot cloning: build a speaker from a reference wav (create_speaker), wav-only.
So it shows up in both the default and cloning leaderboards.

We use the HF (transformers) backend so there's no llama.cpp compile step — it runs on
CUDA with the cu128 torch we install. (The library also ships llama.cpp/CPU/Metal backends;
the HF path is the simplest cross-rig choice and matches how we drive the other LLM-TTS.)

API (from the model card Quick Start):
    import outetts
    interface = outetts.Interface(config=outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B, backend=outetts.Backend.HF))
    speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")   # default voice
    # speaker = interface.create_speaker("ref.wav")                    # cloning (wav only)
    output = interface.generate(config=outetts.GenerationConfig(
        text="...", generation_type=outetts.GenerationType.CHUNKED, speaker=speaker,
        sampler_config=outetts.SamplerConfig(temperature=0.4)))
    output.save("out.wav")

License: the OuteAI fine-tune is CC-BY-NC-SA-4.0; the Llama-3.2 base carries Meta's
community license. Non-commercial — within the bench's redistribute-results bar.
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
DEFAULT_PRESET = "EN-FEMALE-1-NEUTRAL"

_REAL_STDOUT = sys.stdout


def _emit(obj) -> None:
    print(json.dumps(obj), file=_REAL_STDOUT, flush=True)


def _patch_torchaudio_soundfile() -> None:
    """torchaudio 2.9+ routes load/save through torchcodec, which needs FFmpeg 4-7;
    this box has FFmpeg 8, so the codec path raises "TorchCodec is required". OuteTTS
    decodes the reference wav (create_speaker) and writes output via torchaudio, so we
    swap both for soundfile-backed equivalents (same return/arg shapes). Mirrors how the
    echo runner sidesteps torchcodec."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    def _load(path, *a, **k):
        # OuteTTS calls torchaudio.load with either a path or a file-like (BytesIO);
        # soundfile reads both — pass file-likes through, stringify real paths.
        target = path if hasattr(path, "read") else str(path)
        data, sr = sf.read(target, dtype="float32", always_2d=True)      # (frames, ch)
        return torch.from_numpy(data.T.copy()), sr                       # (ch, frames), sr

    def _save(path, src, sample_rate=None, **k):
        target = path if hasattr(path, "write") else str(path)
        arr = src.detach().cpu().numpy() if hasattr(src, "detach") else np.asarray(src)
        if arr.ndim == 1:
            arr = arr[None, :]
        sf.write(target, arr.T, int(sample_rate), subtype="PCM_16")      # (frames, ch)

    torchaudio.load = _load
    torchaudio.save = _save


def _wav_duration_s(path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return float(info.frames / info.samplerate)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot cloning (wav only). Absent -> preset voice.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        _emit({"ok": False, "run_index": 0,
               "error": "either --stdin or both --text and --out are required"})
        return 1

    cloning = bool(args.reference)
    ref_wav = Path(args.reference) if args.reference else None
    if cloning and not ref_wav.exists():
        _emit({"ok": False, "run_index": 0,
               "error": f"reference wav not found: {ref_wav}"})
        return 1

    try:
        import torch
        _patch_torchaudio_soundfile()
        import outetts

        if args.device == "cuda" and not torch.cuda.is_available():
            _emit({"ok": False, "run_index": 0,
                   "error": "CUDA requested but not available"})
            return 1

        with contextlib.redirect_stdout(sys.stderr):
            cfg = outetts.ModelConfig.auto_config(
                model=outetts.Models.VERSION_1_0_SIZE_1B,
                backend=outetts.Backend.HF,
            )
            cfg.device = args.device   # auto_config leaves device=None; pin it explicitly
            interface = outetts.Interface(config=cfg)
            if cloning:
                speaker = interface.create_speaker(str(ref_wav))
            else:
                speaker = interface.load_default_speaker(DEFAULT_PRESET)
    except Exception as e:
        _emit({"ok": False, "run_index": 0,
               "error": f"load failed: {type(e).__name__}: {e}"})
        return 1

    def _one(text: str, out_path: str, run_index: int, write_wav: bool) -> bool:
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                output = interface.generate(config=outetts.GenerationConfig(
                    text=text,
                    generation_type=outetts.GenerationType.CHUNKED,
                    speaker=speaker,
                    sampler_config=outetts.SamplerConfig(temperature=0.4),
                ))
                # OuteTTS writes the wav (native DAC sample rate) itself.
                tmp_out = out_path if write_wav else str(REPO_ROOT / "_outetts_scratch.wav")
                output.save(tmp_out)
            t_end = time.perf_counter()

            audio_s = _wav_duration_s(tmp_out)

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
