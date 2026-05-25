"""MOSS-TTS-Nano runner (OpenMOSS / MOSI.AI, ~100M params, Apache 2.0).

Lightweight cousin of the MOSS-TTS flagship. Zero-shot voice cloning at 48 kHz
(via MOSS-Audio-Tokenizer-Nano), Qwen-architecture LM head trained for
multilingual TTS, designed for CPU-realtime streaming on ~4 cores.

API (from the upstream `infer.py`, mirrored here so we own the load+gen path):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "OpenMOSS-Team/MOSS-TTS-Nano", trust_remote_code=True,
    ).to(device=device, dtype=dtype)
    model.eval()
    result = model.inference(
        text=<target_text>,
        output_audio_path=<wav_out>,
        mode="voice_clone",
        prompt_audio_path=<ref_wav>,
        audio_tokenizer_type="moss-audio-tokenizer-nano",
        audio_tokenizer_pretrained_name_or_path="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        device=device, nq=None, max_new_frames=375,
        voice_clone_max_text_tokens=75,
        voice_clone_max_memory_per_sample_gb=1.0,
        do_sample=True, use_kv_cache=True,
        text_temperature=1.0, text_top_p=1.0, text_top_k=50,
        audio_temperature=0.8, audio_top_p=0.95, audio_top_k=25,
        audio_repetition_penalty=1.2,
    )
    # result["audio_path"] now holds the wav; result["sample_rate"] is 48000.

Cloning shape: wav prompt only (no sibling transcript required). Default-voice
path: falls back to reference/chris_hemsworth_15s.wav since the model is
voice-clone-first.

WeTextProcessing / pynini are intentionally not installed (no Windows pynini
wheel; cross-platform parity). The runner sets enable_wetext_processing=0 and
relies on the in-repo `normalize_tts_text` cleanup, matching upstream's
documented fallback.

Sample rate: 48 kHz. Non-streaming (TTFA == gen_s) — the model has a streaming
mode, but draining-to-wav is what the harness measures today, so realtime/RTF
numbers understate the model's actual streaming first-chunk latency.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo
import _naq


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REF = REPO_ROOT / "reference" / "chris_hemsworth_15s.wav"

MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Nano"
AUDIO_TOKENIZER_TYPE = "moss-audio-tokenizer-nano"
AUDIO_TOKENIZER_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"


def _resolve_device(device_arg: str):
    import torch
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device_arg == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(device):
    import torch
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot voice cloning (wav only, no transcript needed).")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    # Default voice path: borrow chris_hemsworth_15s.wav (model is clone-first).
    if args.reference:
        ref_wav = Path(args.reference)
    else:
        ref_wav = DEFAULT_REF

    if not ref_wav.exists():
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference wav not found: {ref_wav}"}))
        return 1

    try:
        import torch
        import numpy as np
        import soundfile as sf
        import torchaudio
        from transformers import AutoModelForCausalLM

        # torch>=2.9 routes torchaudio.load() through torchcodec, which on
        # Windows needs FFmpeg shared DLLs that aren't on stock installs.
        # Monkey-patch torchaudio.load with a soundfile-backed equivalent
        # that returns (Tensor[C,T], sr) — same shape upstream expects.
        # (Same trick dia_runner.py uses.)
        def _sf_load(path, channels_first=True, **_kwargs):
            data, sr = sf.read(str(path), always_2d=True)  # (T, C) float64
            tensor = torch.from_numpy(data.T.astype(np.float32))  # (C, T)
            return (tensor if channels_first else tensor.T, sr)

        torchaudio.load = _sf_load

        device = _resolve_device(args.device)
        dtype = _resolve_dtype(device)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )
        model.to(device=device, dtype=dtype)
        # Best-effort: ask for SDPA if the wrapper exposes the hook (upstream does).
        if hasattr(model, "_set_attention_implementation"):
            try:
                model._set_attention_implementation("sdpa")
            except Exception:
                pass
        model.eval()
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text: str, out_path: str, run_index: int, write_wav: bool) -> bool:
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            # Mirror infer.py's inference call. enable_wetext_processing is
            # implicitly disabled (we just don't pass a normalizer manager
            # and call the lower-level model.inference directly, which doesn't
            # invoke WeTextProcessing).
            result = model.inference(
                text=text,
                output_audio_path=str(out_path),
                mode="voice_clone",
                prompt_text=None,
                prompt_audio_path=str(ref_wav),
                reference_audio_path=None,
                text_tokenizer_path=None,
                audio_tokenizer_type=AUDIO_TOKENIZER_TYPE,
                audio_tokenizer_pretrained_name_or_path=AUDIO_TOKENIZER_ID,
                device=device,
                nq=None,
                max_new_frames=375,
                voice_clone_max_text_tokens=75,
                voice_clone_max_memory_per_sample_gb=1.0,
                do_sample=True,
                use_kv_cache=True,
                text_temperature=1.0,
                text_top_p=1.0,
                text_top_k=50,
                audio_temperature=0.8,
                audio_top_p=0.95,
                audio_top_k=25,
                audio_repetition_penalty=1.2,
            )
            t_end = time.perf_counter()

            sr = int(result["sample_rate"])
            # The model writes the wav itself to output_audio_path; we need
            # audio_s for the harness. Probe from the file rather than from
            # token counts so the duration matches reality.
            import soundfile as sf
            info = sf.info(str(out_path))
            audio_s = float(info.frames / info.samplerate)

            # On warm runs the harness wants us to NOT overwrite the cold wav.
            # Easiest path: regenerate to a temp file, drop it. Cheap enough.
            if not write_wav:
                try:
                    Path(out_path + ".warm").unlink(missing_ok=True)  # no-op safety
                except Exception:
                    pass

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
                **_meminfo.sample(args.device),
                **(_naq.score(out_path) if write_wav else
                   {"naq": None, "naq_artifact": None, "naq_naturalness": None}),
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
        # Per harness contract: only run 0 writes the cold wav. For warm runs
        # we still need to generate (to measure warm gen_s); we send the
        # output to a sibling .warm.wav so the cold wav isn't clobbered.
        if i == 0:
            target = args.out
        else:
            target = str(Path(args.out).with_name(Path(args.out).stem + ".warm.wav"))
        if not _one(args.text, target, i, write_wav=(i == 0)):
            return 1
        # Clean up the warm scratch file we wrote for measurement purposes.
        if i > 0:
            try:
                Path(target).unlink(missing_ok=True)
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
