"""VibeVoice-Realtime-0.5B runner (community fork, predefined voices only, streaming-class).

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


VOICE_PRESETS = {
    "en-Carter_man":   None,
    "en-Davis_man":    None,
    "en-Emma_woman":   None,
    "en-Frank_man":    None,
    "en-Grace_woman":  None,
    "en-Mike_man":     None,
    "in-Samuel_man":   None,
}

DEFAULT_VOICE = "en-Emma_woman"
VOICE_REPO_RAW = "https://raw.githubusercontent.com/vibevoice-community/VibeVoice/main/demo/voices/streaming_model"


def _voice_cache_dir() -> Path:
    return Path(os.environ.get("VIBEVOICE_VOICE_DIR", Path.home() / ".cache" / "vibevoice-voices"))


def _resolve_voice(voice_name: str) -> Path:
    """Download voice .pt on first use; return local path."""
    cache = _voice_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"{voice_name}.pt"
    if not local.exists():
        url = f"{VOICE_REPO_RAW}/{voice_name}.pt"
        urllib.request.urlretrieve(url, local)
    return local


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Voice preset NAME (e.g. 'en-Emma_woman') — VibeVoice doesn't support wav cloning at runtime.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    p.add_argument("--ddpm-steps", type=int, default=5,
                   help="DDPM inference steps. Lower = faster, default 5 per the upstream demo.")
    p.add_argument("--cfg-scale", type=float, default=1.5)
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    if args.language not in ("en",):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"VibeVoice-Realtime ships English voice presets only; got language={args.language}"}))
        return 1

    try:
        import torch
        import numpy as np
        import soundfile as sf
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

        voice_name = args.reference or DEFAULT_VOICE
        if voice_name not in VOICE_PRESETS:
            # Allow short-form aliases ("Emma" -> "en-Emma_woman")
            matched = next((k for k in VOICE_PRESETS if voice_name.lower() in k.lower()), None)
            if not matched:
                raise ValueError(
                    f"Unknown voice {voice_name!r}. Choose from: {list(VOICE_PRESETS)}"
                )
            voice_name = matched
        voice_path = _resolve_voice(voice_name)
        voice = torch.load(voice_path, map_location=device, weights_only=False)
        samplerate = 24000
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
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
