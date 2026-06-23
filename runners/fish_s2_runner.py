"""Fish Audio S2-Pro runner (fishaudio/fish-speech, DualAR + DAC codec, 44.1 kHz).

Source-clone install (no pip wheel) — see install.sh. Weights (`fishaudio/s2-pro`)
download via huggingface_hub snapshot_download on first run; cached after. The
snapshot dir holds the DualAR safetensors shards + `codec.pth` (the DAC codec).

In-proc, NOT subprocess: the whole 3-stage pipeline is importable callable
functions (the click `main()` in the upstream module is just a thin CLI wrapper):
    from fish_speech.models.text2semantic.inference import (
        init_model, load_codec_model, encode_audio, decode_to_audio, generate_long)
    1. encode_audio(ref_wav, codec)        -> prompt VQ codes   (cloning ref; once at load)
    2. generate_long(text, prompt_*)       -> semantic codes    (DualAR AR sampler)
    3. decode_to_audio(codes, codec)       -> waveform @ 44.1 kHz

Cloning flavor: wav + transcript (the reference's text). Bundled refs ship a
sibling .txt (jo.txt / juliette.txt); a user --reference wav without a sibling
.txt falls back to voiceless generation (model picks a voice).

License: CC-BY-NC-SA-4.0 (model). 44.1 kHz output. CUDA-only (bf16 ~21 GB peak).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo

# KV-cache length cap. The model's native max_seq_len is 32768, which makes
# setup_caches allocate a ~6 GB KV cache (total peak ~21.4 GB on a 3090). Bench
# prompts are 1-2 sentences (+ ~300 ref codes), so 4096 is ample and reclaims
# headroom on 24 GB cards.
MAX_SEQ_LEN = 4096


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot cloning. A sibling .txt (the "
                        "reference transcript) is used if present.")
    p.add_argument("--variant", default=None)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="DualAR sampling temperature (bench default 0.8).")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    # Default-voice path: borrow a bundled reference (clone-first model).
    repo = Path(__file__).resolve().parent.parent
    if args.reference:
        ref_wav = Path(args.reference)
    else:
        default_ref = {"en": "jo.wav", "fr": "juliette.wav"}.get(args.language, "jo.wav")
        ref_wav = repo / "reference" / default_ref
    if not ref_wav.exists():
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference wav not found: {ref_wav}"}))
        return 1
    ref_txt_path = ref_wav.with_suffix(".txt")
    ref_text = ref_txt_path.read_text().strip() if ref_txt_path.exists() else None

    try:
        import torch
        import soundfile as sf
        from huggingface_hub import snapshot_download
        from fish_speech.models.text2semantic.inference import (
            init_model, load_codec_model, encode_audio, decode_to_audio, generate_long)

        precision = torch.bfloat16
        ckpt = Path(snapshot_download("fishaudio/s2-pro"))
        model, decode_one_token = init_model(ckpt, args.device, precision, compile=False)
        with torch.device(args.device):
            model.setup_caches(max_batch_size=1, max_seq_len=MAX_SEQ_LEN,
                               dtype=next(model.parameters()).dtype)
        if args.device == "cuda":
            torch.cuda.synchronize()
        codec = load_codec_model(ckpt / "codec.pth", args.device, precision)
        sample_rate = codec.sample_rate
        # Stage 1 (DAC encode of the cloning reference) depends only on the ref,
        # not the text — do it once and reuse across runs.
        prompt_tokens = encode_audio(str(ref_wav), codec, args.device).cpu() if ref_text else None
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            import torch

            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            torch.manual_seed(42)
            if args.device == "cuda":
                torch.cuda.manual_seed(42)
            gen = generate_long(
                model=model, device=args.device, decode_one_token=decode_one_token,
                text=text, num_samples=1, top_p=0.8, top_k=30, temperature=args.temperature,
                repetition_penalty=1.1, compile=False, iterative_prompt=True,
                chunk_length=300,
                prompt_text=[ref_text] if ref_text else None,
                prompt_tokens=[prompt_tokens] if ref_text else None,
            )
            codes = []
            for r in gen:
                if r.action == "sample":
                    codes.append(r.codes)
                elif r.action == "next":
                    break
            merged = torch.cat(codes, dim=1)
            audio = decode_to_audio(merged.to(args.device), codec)
            if args.device == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            arr = audio.cpu().float().numpy()
            audio_s = float(len(arr) / sample_rate)
            if write_wav:
                sf.write(out_path, arr, sample_rate)

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
