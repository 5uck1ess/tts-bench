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


TURBO_REPO = "ResembleAI/chatterbox-turbo"
BASE_REPO  = "ResembleAI/chatterbox"


def _load_base(device):
    """Load Chatterbox 1.2B (base). Returns (model, samplerate)."""
    from chatterbox.tts import ChatterboxTTS

    m = ChatterboxTTS.from_pretrained(device=device)
    samplerate = int(m.sr) if hasattr(m, "sr") else 24000
    return m, samplerate


def _load_turbo(device):
    """Load Chatterbox Turbo (~744M, GPT2-based AR model).

    The turbo checkpoint uses a different T3 architecture (GPT2_medium instead
    of Llama_520M) and a different file naming convention (t3_turbo_v1.safetensors
    vs t3_cfg.safetensors). We build the model manually since ChatterboxTTS.from_local()
    hardcodes the base filenames.

    Returns (model, samplerate).
    """
    from pathlib import Path
    import torch
    import torch.nn.functional as F
    from chatterbox.tts import ChatterboxTTS, Conditionals
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.tokenizers import EnTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download, snapshot_download

    # Download turbo snapshot (cached after first run)
    turbo_dir = Path(snapshot_download(TURBO_REPO))

    # Turbo has no tokenizer.json — use the base model's tokenizer
    tokenizer_json = hf_hub_download(repo_id=BASE_REPO, filename="tokenizer.json")

    # Build T3 with GPT2_medium config matching the turbo checkpoint
    hp = T3Config.english_only()
    hp.llama_config_name = "GPT2_medium"
    hp.speech_cond_prompt_len = 250      # from t3_turbo_v1.yaml
    hp.use_perceiver_resampler = False   # GPT2 path has no perceiver
    hp.input_pos_emb = "handled_internally_by_backbone"  # GPT2 handles pos internally
    hp.text_tokens_dict_size = 50276     # GPT2 vocab (full)
    hp.speech_tokens_dict_size = 6563   # turbo checkpoint dimension
    hp.start_speech_token = 6561
    hp.stop_speech_token = 6562

    # Always load to CPU first for non-CUDA devices (handles CUDA-saved weights)
    map_location = None if device == "cuda" else torch.device("cpu")

    t3 = T3(hp)
    t3.load_state_dict(load_file(turbo_dir / "t3_turbo_v1.safetensors"), strict=False)
    t3.to(device).eval()

    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(turbo_dir / "s3gen.safetensors"), strict=False)
    s3gen.to(device).eval()

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(turbo_dir / "ve.safetensors"))
    ve.to(device).eval()

    tokenizer = EnTokenizer(tokenizer_json)

    conds_path = turbo_dir / "conds.pt"
    conds = Conditionals.load(conds_path, map_location=map_location or device) if conds_path.exists() else None

    m = ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds=conds)
    samplerate = int(m.sr) if hasattr(m, "sr") else 24000
    return m, samplerate


def _turbo_generate(m, text, audio_prompt_path=None):
    """Generate audio using the turbo inference path (T3.inference_turbo).

    The base ChatterboxTTS.generate() calls t3.inference() which requires
    learned positional embeddings — absent in the GPT2 turbo model. This
    function replicates the generate() body but calls inference_turbo() instead.
    """
    import torch
    import torch.nn.functional as F
    from chatterbox.tts import punc_norm
    from chatterbox.models.s3tokenizer import drop_invalid_tokens

    if audio_prompt_path:
        m.prepare_conditionals(audio_prompt_path)

    assert m.conds is not None, (
        "No conditionals loaded. Provide --reference for turbo cloning, "
        "or ensure conds.pt exists in the turbo checkpoint."
    )

    text = punc_norm(text)
    text_tokens = m.tokenizer.text_to_tokens(text).to(m.device)

    sot = m.t3.hp.start_text_token
    eot = m.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)

    with torch.inference_mode():
        speech_tokens = m.t3.inference_turbo(
            t3_cond=m.conds.t3,
            text_tokens=text_tokens,
            max_gen_len=600,
        )
        speech_tokens = drop_invalid_tokens(speech_tokens.squeeze(0))
        speech_tokens = speech_tokens[speech_tokens < m.t3.hp.start_speech_token]
        speech_tokens = speech_tokens.to(m.device)

        wav, _ = m.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=m.conds.gen,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
        wav = m.watermarker.apply_watermark(wav, sample_rate=m.sr)

    import torch as _torch
    return _torch.from_numpy(wav).unsqueeze(0)


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

            if use_turbo:
                audio = _turbo_generate(m, text, audio_prompt_path=args.reference)
            else:
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
