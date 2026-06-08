"""MOSS-TTS flagship runner (OpenMOSS, Apache 2.0, 8B Qwen3-backbone TTS).

Zero-shot voice cloning, multilingual synthesis, long-form. Uses a Qwen3-style
LM backbone + the OpenMOSS Cat audio tokenizer. CUDA-targeted; CPU works but
is impractically slow given the 8B parameter count.

VERSIONS: this runner serves BOTH checkpoints, selected by the harness `variant`
field (see MODEL_IDS) — moss_tts → v1.0 (`OpenMOSS-Team/MOSS-TTS`), moss_tts_v15
→ v1.5 (`OpenMOSS-Team/MOSS-TTS-v1.5`). v1.5 is continued training from 1.0 (same
8B Delay architecture/API) and improves voice-cloning stability, long-ref /
short-text cloning, and punctuation prosody, and extends language coverage to 31
(from 20). It is NOT a strict superset: with the language field OMITTED, v1.5 can
regress slightly on some languages; with it SPECIFIED it is stronger than 1.0
almost everywhere — so for v1.5 ONLY this runner passes an explicit `language=`
tag (see LANG_NAMES). v1.0 keeps its original behaviour (no language kwarg). Both
are kept in the bench because 1.0 still wins on some material by ear.

NOTE on size: the upstream README's Released Models table documents two
architectures:
  - MossTTSDelay (MOSS-TTS proper): 8B params  ← what this runner loads
  - MossTTSLocal (MOSS-TTS-Local-Transformer): 1.7B (alt checkpoint)
We default to the 8B Delay model as that's the flagship advertised on the repo
front page. The Local 1.7B variant would be a separate `--variant` follow-up.

API (lifted from upstream's MOSS-TTS Basic Usage):
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained("OpenMOSS-Team/MOSS-TTS-v1.5", trust_remote_code=True)
    proc.audio_tokenizer = proc.audio_tokenizer.to(device)
    model = AutoModel.from_pretrained(
        "OpenMOSS-Team/MOSS-TTS-v1.5", trust_remote_code=True,
        attn_implementation=attn, torch_dtype=dtype,
    ).to(device); model.eval()
    # v1.5: pass language=<English name> for best multilingual quality.
    conversations = [[proc.build_user_message(text=text, reference=[ref_wav], language="English")]]
    batch = proc(conversations, mode="generation")
    outputs = model.generate(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        max_new_tokens=4096,
    )
    for message in proc.decode(outputs):
        audio = message.audio_codes_list[0]            # torch.Tensor [T]
        torchaudio.save(out, audio.unsqueeze(0), proc.model_config.sampling_rate)

Quirks handled inline:
- cuDNN SDPA is broken upstream; disable it before importing the model.
- Reference can be a wav path or URL (processor handles both); we always pass
  a local path. Default voice falls back to reference/chris_hemsworth_15s.wav.
- Predicted dtype is bfloat16 on CUDA (preferred over fp16 for AR sampling
  stability), float32 on CPU.

Devices: this runner declares CUDA-only in harness.py — the 8B param count
makes CPU loading hopeless, and MPS doesn't have a kernel set wide enough for
the Qwen3 backbone + RVQ decoder at usable speed.
"""

import os
# MOSS-TTS 8B peaks at ~22.8 GiB VRAM. On a 24 GiB card (e.g. RTX 3090) the
# default CUDA allocator fragments ~0.7 GiB past the limit and OOMs; expandable
# segments reclaims it so the model fits. Must be set before torch initialises
# CUDA. setdefault() lets an explicit outer env override still win.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REF = REPO_ROOT / "reference" / "chris_hemsworth_15s.wav"

# Two checkpoints share this runner + venv (v1.5 is continued training from 1.0,
# same Delay architecture + processor classes). The harness `variant` field picks
# which: moss_tts → v1.0 (default), moss_tts_v15 → v1.5. We keep BOTH in the bench
# because 1.0 still wins on some material by ear.
MODEL_IDS = {
    None:   "OpenMOSS-Team/MOSS-TTS",        # v1.0 (default when no --variant)
    "v1.0": "OpenMOSS-Team/MOSS-TTS",
    "v1.5": "OpenMOSS-Team/MOSS-TTS-v1.5",
}

# Only v1.5 accepts/benefits from an explicit language tag (build_user_message
# gained the `language=` kwarg in v1.5; passing it to 1.0 is unsupported). v1.5
# wants the language's English name (e.g. "French"), not an ISO code — map the
# bench's lang codes → MOSS names. Codes not listed fall through to auto-detect
# (kwarg omitted); extend only with names verified against the v1.5 table.
LANG_NAMES = {
    "en": "English",
    "fr": "French",
}


def _resolve_attn_implementation(device, dtype):
    """Prefer FlashAttention 2 on Ampere+ CUDA if installed; else SDPA on CUDA, eager on CPU."""
    import importlib.util
    import torch
    if device.type == "cuda":
        if (importlib.util.find_spec("flash_attn") is not None
                and dtype in (torch.float16, torch.bfloat16)):
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return "flash_attention_2"
        return "sdpa"
    return "eager"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot voice cloning (wav only, no transcript needed).")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    # Resolve checkpoint from the harness variant (moss_tts→v1.0, moss_tts_v15→v1.5).
    model_id = MODEL_IDS.get(args.variant, MODEL_IDS[None])
    # Only v1.5 takes the explicit language tag (bench passes --language per prompt;
    # unknown codes → None → auto-detect). v1.0 never gets the kwarg.
    lang_name = LANG_NAMES.get((args.language or "").lower()) if args.variant == "v1.5" else None

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
        import torchaudio
        from transformers import AutoModel, AutoProcessor

        # Disable the broken cuDNN SDPA backend (per upstream README).
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        if args.device == "cuda" and not torch.cuda.is_available():
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": "CUDA requested but not available"}))
            return 1
        if args.device == "mps" and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": "MPS requested but not available"}))
            return 1
        device = torch.device(args.device if args.device in ("cuda", "cpu", "mps") else "cuda")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        attn = _resolve_attn_implementation(device, dtype)

        # Windows + AutoProcessor + trust_remote_code: transformers 5.x's
        # path-normalize step turns "OpenMOSS-Team/MOSS-TTS" into a Windows
        # backslash form before the HF Hub repo-id regex sees it, so it raises
        # "Repo id must use alphanumeric chars...". Workaround: pre-download via
        # snapshot_download (uses huggingface_hub's own repo-id parser) and load
        # the processor + model from the resulting local path.
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(model_id)

        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            attn_implementation=attn,
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        samplerate = int(processor.model_config.sampling_rate)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text: str, out_path: str, run_index: int, write_wav: bool) -> bool:
        try:
            import torch

            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()

            msg_kwargs = {"text": text, "reference": [str(ref_wav)]}
            if lang_name:
                msg_kwargs["language"] = lang_name  # v1.5 multilingual tag
            conversations = [[processor.build_user_message(**msg_kwargs)]]
            batch = processor(conversations, mode="generation")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                )

            decoded = list(processor.decode(outputs))
            if not decoded:
                raise RuntimeError("processor.decode returned no messages")
            message = decoded[0]
            audio = message.audio_codes_list[0]
            t_end = time.perf_counter()

            # audio is a 1-D torch.Tensor of waveform samples. Save via torchaudio
            # using shape (1, T). Cast to float32 to dodge bfloat16 save quirks.
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().float()
            audio_2d = audio.unsqueeze(0) if audio.dim() == 1 else audio
            audio_s = float(audio_2d.shape[-1] / samplerate)

            if write_wav:
                torchaudio.save(str(out_path), audio_2d, samplerate)

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
