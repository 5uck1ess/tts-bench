"""LongCat-AudioDiT runner (Meituan, MIT, diffusion TTS in waveform latent space, 24 kHz).

Non-autoregressive diffusion TTS that generates directly in a Wav-VAE waveform
latent space (no mel / no codec tokens): a Wav-VAE + a DiT backbone, sampled with
an ODE solver over `steps` (NFE) function evaluations. Two released sizes share
this runner; --variant picks the HF checkpoint:
    1b   -> meituan-longcat/LongCat-AudioDiT-1B
    3.5b -> meituan-longcat/LongCat-AudioDiT-3.5B

Both lenses:
  * default (no --reference): zero-shot synthesis, the model's own voice.
  * cloning (--reference wav + sibling .txt): the reference text is prepended to
    the target text for the text encoder and the reference waveform is passed as
    prompt_audio; the model continues in that voice. Like sesame/miso, cloning
    needs the literal words of the reference, read from <reference>.txt.

Guidance: upstream's headline contribution is replacing classifier-free guidance
with adaptive projection guidance (APG); the README's cloning examples use it, so
we run guidance_method="apg" for the strongest representation of the model. NFE,
CFG strength and seed follow the upstream CLI defaults (16 / 4.0 / 1024).

Source-clone import (venvs/longcat/src): inference.py is NOT a pip package, so the
runner adds that tree to sys.path and reuses its modeling code (`import audiodit`
auto-registers AudioDiTConfig/AudioDiTModel with transformers) and its text/audio
helpers (utils.normalize_text / load_audio / approx_duration_from_text). The
duration-estimation logic mirrors inference.py exactly.

Weights auto-download from HF on first run (1B ~ a few GB; 3.5B larger). The text
encoder/tokenizer is whatever model.config.text_encoder_model points at (also
auto-downloaded). ZH + EN only (no French -> multilingual=False, FR prompt skips).
CUDA-only in harness.py (DiT + fp16 VAE; sub-realtime on CPU). License: MIT.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


REPO_ROOT = Path(__file__).resolve().parent.parent
LONGCAT_SRC = REPO_ROOT / "venvs" / "longcat" / "src"

VARIANT_MODELS = {
    "1b": "meituan-longcat/LongCat-AudioDiT-1B",
    "3.5b": "meituan-longcat/LongCat-AudioDiT-3.5B",
}

# Upstream CLI defaults (inference.py).
NFE = 16                  # ODE steps
GUIDANCE_STRENGTH = 4.0   # CFG/APG strength
GUIDANCE_METHOD = "apg"   # adaptive projection guidance (the paper's improvement)
SEED = 1024


def _read_ref_transcript(ref_wav: str | None) -> str | None:
    if not ref_wav:
        return None
    txt_path = Path(ref_wav).with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for cloning. Sibling .txt transcript required (prepended to the target text, like sesame/miso).")
    p.add_argument("--variant", default="1b", help="1b | 3.5b")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    variant = (args.variant or "1b").lower()
    model_dir = VARIANT_MODELS.get(variant)
    if model_dir is None:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"unknown variant {args.variant!r} (expected one of {sorted(VARIANT_MODELS)})"}))
        return 1

    ref_wav = args.reference
    ref_text = _read_ref_transcript(ref_wav)
    if ref_wav and not ref_text:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference {ref_wav} provided but sibling .txt transcript missing "
                                   f"(cloning prepends the reference's literal words to the text)"}))
        return 1

    try:
        sys.path.insert(0, str(LONGCAT_SRC))

        import numpy as np
        import soundfile as sf
        import torch
        import torch.nn.functional as F

        import audiodit  # noqa: F401  (auto-registers AudioDiTConfig/AudioDiTModel)
        from audiodit import AudioDiTModel
        from transformers import AutoTokenizer
        from utils import normalize_text, load_audio, approx_duration_from_text

        torch.backends.cudnn.benchmark = False

        device = args.device if args.device in ("cuda", "cpu") else "cpu"
        torch.manual_seed(SEED)
        if device == "cuda":
            torch.cuda.manual_seed(SEED)

        model = AudioDiTModel.from_pretrained(model_dir).to(device)
        model.vae.to_half()   # VAE in fp16, matching upstream
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

        sr = model.config.sampling_rate
        full_hop = model.config.latent_hop
        max_duration = model.config.max_wav_duration

        # Reference (computed once; constant across prompts in a cell).
        prompt_wav = None
        prompt_dur = 0
        prompt_text = None
        if ref_wav and ref_text:
            prompt_text = normalize_text(ref_text)
            prompt_wav = load_audio(ref_wav, sr).unsqueeze(0)   # (1, 1, T)
            off = 3
            pw = load_audio(ref_wav, sr)   # (1, T); vae.encode wants (1, 1, T) below
            if pw.shape[-1] % full_hop != 0:
                pw = F.pad(pw, (0, full_hop - pw.shape[-1] % full_hop))
            pw = F.pad(pw, (0, full_hop * off))
            with torch.no_grad():
                plt = model.vae.encode(pw.unsqueeze(0).to(device))
            if off:
                plt = plt[..., :-off]
            prompt_dur = plt.shape[-1]
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            text = normalize_text(text)
            full_text = f"{prompt_text} {text}" if prompt_text else text
            inputs = tokenizer([full_text], padding="longest", return_tensors="pt")

            # Duration estimation (mirrors inference.py).
            prompt_time = prompt_dur * full_hop / sr
            dur_sec = approx_duration_from_text(text, max_duration=max_duration - prompt_time)
            if prompt_text:
                approx_pd = approx_duration_from_text(prompt_text, max_duration=max_duration)
                ratio = float(np.clip(prompt_time / approx_pd, 1.0, 1.5)) if approx_pd else 1.0
                dur_sec = dur_sec * ratio
            duration = int(dur_sec * sr // full_hop)
            duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))

            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                output = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    prompt_audio=prompt_wav,
                    duration=duration,
                    steps=NFE,
                    cfg_strength=GUIDANCE_STRENGTH,
                    guidance_method=GUIDANCE_METHOD,
                )
            t_end = time.perf_counter()

            arr = output.waveform.squeeze().detach().cpu().float().numpy().reshape(-1).astype(np.float32)
            audio_s = float(len(arr) / sr)
            if write_wav:
                sf.write(out_path, arr, sr)

            # Non-streaming diffusion: no partial-audio concept, so TTFA == full gen.
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
