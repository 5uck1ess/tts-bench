"""VibeVoice runner — handles two variants under the same venv.

Variant dispatch:
    --variant absent OR --variant 0.5b_streaming
        → VibeVoice-Realtime-0.5B (streaming classes, pre-cached .pt voice prompts)
    --variant 1.5b
        → VibeVoice-1.5B (non-streaming classes, raw wav voice samples)

=== 0.5B Streaming path ===
API (vibevoice-community/VibeVoice fork, 2025-12-04+ streaming variant):
    from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    proc = VibeVoiceStreamingProcessor.from_pretrained('microsoft/VibeVoice-Realtime-0.5B')
    m = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(...)
    m.set_ddpm_inference_steps(num_steps=5)
    voice = torch.load(<voice.pt>, weights_only=False)
    inputs = proc.process_input_with_cached_prompt(text=..., cached_prompt=voice, ...)
    outputs = m.generate(**inputs, cfg_scale=1.5, all_prefilled_outputs=copy.deepcopy(voice), ...)
    audio = outputs.speech_outputs[0]   # torch.Tensor, 24kHz

Voice presets (.pt files, 2-4MB each) auto-download on first use to ~/.cache/vibevoice-voices/
from github.com/vibevoice-community/VibeVoice/tree/main/demo/voices/streaming_model.
Presets: en-Carter_man, en-Davis_man, en-Emma_woman, en-Frank_man, en-Grace_woman,
en-Mike_man, in-Samuel_man.

Non-streaming output (the streaming model name is misleading at the bench layer —
it streams INPUT, not output). TTFA == gen_s.

Architecture note: the HF checkpoint deliberately omits the acoustic_tokenizer
encoder. At load time transformers warns "you should probably TRAIN this model"
about 400+ uninitialized encoder weights. This is expected — the .pt voice
presets are pre-encoded representations, so the encoder is dead weight at
inference. Audio quality is unaffected.

=== 1.5B Non-streaming path ===
API (vibevoice-community/VibeVoice, non-streaming classes):
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    proc = VibeVoiceProcessor.from_pretrained('microsoft/VibeVoice-1.5B')
    m = VibeVoiceForConditionalGenerationInference.from_pretrained(...)
    m.set_ddpm_inference_steps(num_steps=10)
    inputs = proc(
        text=["Speaker 0: <text>"],
        voice_samples=["path/to/voice.wav"],   # processor resamples to 24kHz via librosa
        return_tensors="pt",
    ).to(device)
    outputs = m.generate(**inputs, cfg_scale=1.3, tokenizer=proc.tokenizer,
                         max_new_tokens=None, show_progress_bar=False)
    audio = outputs.speech_outputs[0]   # torch.Tensor, 24kHz

Default voice for 1.5B: ./reference/chris_hemsworth_15s.wav (no bundled voices
in the vibevoice package; this file is already present for other runners).

Install gotchas:
- pypi `vibevoice==0.0.1` doesn't have the streaming class. Install from the
  community fork: `git+https://github.com/vibevoice-community/VibeVoice`
- The official `github.com/microsoft/VibeVoice` is also missing the streaming
  variant post-takedown. Use the community fork specifically.
"""

import argparse
import copy
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import _meminfo
import _naq


REPO_ROOT = Path(__file__).resolve().parent.parent

# ── 0.5B streaming variant ────────────────────────────────────────────────────

VOICE_PRESETS = {
    "en-Carter_man":   None,
    "en-Davis_man":    None,
    "en-Emma_woman":   None,
    "en-Frank_man":    None,
    "en-Grace_woman":  None,
    "en-Mike_man":     None,
    "in-Samuel_man":   None,
}

DEFAULT_VOICE_05B = "en-Emma_woman"
VOICE_REPO_RAW = "https://raw.githubusercontent.com/vibevoice-community/VibeVoice/main/demo/voices/streaming_model"

# ── 1.5B non-streaming variant ────────────────────────────────────────────────

DEFAULT_VOICE_15B = str(REPO_ROOT / "reference" / "chris_hemsworth_15s.wav")


def _voice_cache_dir() -> Path:
    return Path(os.environ.get("VIBEVOICE_VOICE_DIR", Path.home() / ".cache" / "vibevoice-voices"))


def _resolve_voice_05b(voice_name: str) -> Path:
    """Download voice .pt on first use; return local path."""
    cache = _voice_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"{voice_name}.pt"
    if not local.exists():
        url = f"{VOICE_REPO_RAW}/{voice_name}.pt"
        urllib.request.urlretrieve(url, local)
    return local


def _load_05b(args):
    """Load 0.5B streaming model + voice. Returns (processor, model, voice, samplerate)."""
    import torch
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

    model_id = "microsoft/VibeVoice-Realtime-0.5B"
    device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
    load_dtype = torch.float32 if device in ("cpu", "mps") else torch.bfloat16
    attn = "sdpa"  # flash_attention_2 only on cuda + recent GPUs; sdpa is safe everywhere

    processor = VibeVoiceStreamingProcessor.from_pretrained(model_id)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        model_id, torch_dtype=load_dtype, device_map=device, attn_implementation=attn,
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)

    voice_name = args.reference or DEFAULT_VOICE_05B
    if voice_name not in VOICE_PRESETS:
        # Allow short-form aliases ("Emma" -> "en-Emma_woman")
        matched = next((k for k in VOICE_PRESETS if voice_name.lower() in k.lower()), None)
        if not matched:
            raise ValueError(
                f"Unknown voice {voice_name!r}. Choose from: {list(VOICE_PRESETS)}"
            )
        voice_name = matched
    voice_path = _resolve_voice_05b(voice_name)
    voice = torch.load(voice_path, map_location=device, weights_only=False)

    return processor, model, voice, device, 24000


def _load_15b(args):
    """Load 1.5B non-streaming model + voice wav path. Returns (processor, model, voice_wav_path, device, samplerate)."""
    import torch
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    model_id = "microsoft/VibeVoice-1.5B"
    device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
    load_dtype = torch.float32 if device in ("cpu", "mps") else torch.bfloat16
    attn = "sdpa"

    processor = VibeVoiceProcessor.from_pretrained(model_id)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_id, torch_dtype=load_dtype, device_map=device, attn_implementation=attn,
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)

    # Reference: harness always passes a wav path for cloning runs; for
    # predefined-voice runs (no --reference) we fall back to the baked-in default.
    voice_wav = args.reference or DEFAULT_VOICE_15B
    if not Path(voice_wav).exists():
        raise FileNotFoundError(
            f"Voice reference wav not found: {voice_wav}. "
            "Provide --reference <path> or ensure reference/chris_hemsworth_15s.wav exists."
        )

    return processor, model, voice_wav, device, 24000


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="0.5B: voice preset NAME (e.g. 'en-Emma_woman'); "
                        "1.5B: path to a reference wav for voice style.")
    p.add_argument("--variant", default=None,
                   help="'0.5b_streaming' (default) or '1.5b'")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    p.add_argument("--ddpm-steps", type=int, default=None,
                   help="DDPM inference steps. Default: 5 for 0.5B, 10 for 1.5B.")
    p.add_argument("--cfg-scale", type=float, default=None,
                   help="CFG scale. Default: 1.5 for 0.5B, 1.3 for 1.5B.")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    if args.language not in ("en",):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"VibeVoice ships English voice presets only; got language={args.language}"}))
        return 1

    # Determine variant and apply per-variant defaults
    variant = (args.variant or "0.5b_streaming").lower()
    use_streaming = variant in ("0.5b_streaming", "0.5b", "realtime")

    if args.ddpm_steps is None:
        args.ddpm_steps = 5 if use_streaming else 10
    if args.cfg_scale is None:
        args.cfg_scale = 1.5 if use_streaming else 1.3

    try:
        import torch
        import numpy as np
        import soundfile as sf

        if use_streaming:
            processor, model, voice, device, samplerate = _load_05b(args)
        elif variant == "1.5b":
            processor, model, voice, device, samplerate = _load_15b(args)
        else:
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": f"Unknown variant {variant!r}. Use '0.5b_streaming' or '1.5b'."}))
            return 1

    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one_05b(text, out_path, run_index, write_wav):
        try:
            inputs = processor.process_input_with_cached_prompt(
                text=text, cached_prompt=voice, padding=True,
                return_tensors="pt", return_attention_mask=True,
            )
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)

            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            outputs = model.generate(
                **inputs, max_new_tokens=None, cfg_scale=args.cfg_scale,
                tokenizer=processor.tokenizer, generation_config={"do_sample": False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(voice),
            )
            t_end = time.perf_counter()

            audio_t = outputs.speech_outputs[0]
            # CUDA bf16 path returns a BFloat16 tensor; numpy() rejects it,
            # so cast to float32 before leaving torch.
            if hasattr(audio_t, "cpu"):
                arr = audio_t.detach().cpu().float().numpy().squeeze()
            else:
                arr = np.asarray(audio_t).squeeze()
            audio_s = float(len(arr) / samplerate)
            if write_wav:
                sf.write(out_path, arr, samplerate)

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

    def _one_15b(text, out_path, run_index, write_wav):
        try:
            # Format text as "Speaker 0: <text>" — the processor's _parse_script
            # expects this format to identify the speaker and map voice samples.
            formatted = f"Speaker 0: {text}" if not text.strip().startswith("Speaker") else text

            inputs = processor(
                text=[formatted],
                voice_samples=[voice],   # voice is a wav path string; librosa resamples to 24kHz
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            # Move all tensor inputs to device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)

            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            outputs = model.generate(
                **inputs,
                cfg_scale=args.cfg_scale,
                tokenizer=processor.tokenizer,
                max_new_tokens=None,
                show_progress_bar=False,
            )
            t_end = time.perf_counter()

            audio_t = outputs.speech_outputs[0]
            if audio_t is None:
                raise RuntimeError("Model returned no audio (speech_outputs[0] is None)")
            if hasattr(audio_t, "cpu"):
                arr = audio_t.detach().cpu().float().numpy().squeeze()
            else:
                arr = np.asarray(audio_t).squeeze()
            audio_s = float(len(arr) / samplerate)
            if write_wav:
                sf.write(out_path, arr, samplerate)

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

    _one = _one_05b if use_streaming else _one_15b

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
