"""Miso TTS 8B runner (Miso Labs, modified-MIT, Sesame-CSM architecture, 24 kHz).

Sesame CSM scaled up: Llama-8B backbone + Llama-300M audio decoder generating
32 Mimi codebooks per 80 ms frame. Same conversational cloning paradigm as our
sesame runner: the reference (text + audio) is passed as a prior turn from
speaker 0, and the model continues in that voice. Default mode is speaker 0
with no context (the model's own voice, like sesame/csm-1b).

Upstream is a source repo, NOT a pip package: generator.py / models.py are
imported from the cloned tree (venvs/miso/src); install.ps1/.sh clone it. We
deliberately do NOT `pip install -e .` — upstream pins torch==2.4.0 (pre-
Blackwell) but the code runs fine on torch 2.7.1+cu128 with torchtune 0.6.1 /
moshi 0.2.2 / current transformers, so the venv resolves those freely and only
the imports below come from the source tree.

Two upstream deviations, both made before load:
  * Tokenizer: generator hardcodes the HF-GATED meta-llama/Llama-3.2-1B for
    text tokenization. We patch in the ungated byte-identical mirror
    unsloth/Llama-3.2-1B (same TemplateProcessing post-processor), so the
    runner needs no Meta license acceptance.
  * Watermark: generate() bakes a silentcipher watermark + 24k->44.1k->24k
    resample round-trip into every call. No other benched model watermarks its
    output, it would pollute gen_s, and silentcipher drags in its own stale
    pins - so we stub the module and no-op the watermark step. Bench clips are
    published explicitly labeled as TTS, so no provenance is lost.

Weights: MisoLabs/MisoTTS (ungated, ~16 GB safetensors, bf16 on CUDA) +
kyutai Mimi codec weights, both auto-download from HF on first run.

CUDA-only in harness.py: 8B bf16 needs ~16 GB VRAM; fp32 CPU would be ~33 GB
RAM and far sub-realtime. License: modified MIT (attribution clause only above
50M MAU / $10M-month revenue - irrelevant here).
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

import _meminfo


REPO_ROOT = Path(__file__).resolve().parent.parent
MISO_SRC = REPO_ROOT / "venvs" / "miso" / "src"

SAMPLE_RATE = 24_000   # Mimi codec
TOKENIZER_ID = "unsloth/Llama-3.2-1B"
# Generation cap: frames are 80 ms, so 30 s = 375 frames. Every bench prompt is
# well under 30 s of speech; a tighter cap bounds the no-EOS failure mode (an
# 8B AR model running to a 90 s default cap would flirt with the cell timeout).
MAX_AUDIO_MS = 30_000
# Cloning sampling: upstream defaults (temperature 0.9, topk 50) lose the
# reference voice more often than not — Tym's by-ear A/B (2026-06-10, chris ref)
# failed at 0.9/50 and held the voice at 0.7/30 and 0.5/20. Voice retention is
# still stochastic shot-to-shot; 0.7/30 keeps expressiveness while loading the
# dice toward the reference. Default mode keeps upstream sampling (passed the
# same by-ear gate as-is).
CLONE_TEMPERATURE = 0.7
CLONE_TOPK = 30


def _stub_silentcipher() -> None:
    """generator.py imports watermarking.py which imports silentcipher at module
    level; we don't install it (see module docstring). The stub satisfies the
    import; the functions that would touch it are no-op'd after import."""
    if "silentcipher" not in sys.modules:
        sc = types.ModuleType("silentcipher")
        sc.server = types.SimpleNamespace(Model=object)
        sc.get_model = lambda **kw: None
        sys.modules["silentcipher"] = sc


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
                   help="Wav path for cloning. Sibling .txt transcript required (passed as prior-turn context, same as sesame).")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    ref_wav = Path(args.reference) if args.reference else None
    ref_text = _read_ref_transcript(args.reference) if args.reference else None
    if args.reference and not ref_text:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference {args.reference} provided but sibling .txt transcript missing "
                                   f"(CSM cloning needs the literal words spoken in the wav)"}))
        return 1

    try:
        import os
        os.environ["NO_TORCH_COMPILE"] = "1"   # upstream convention; no triton on Windows
        sys.path.insert(0, str(MISO_SRC))
        _stub_silentcipher()

        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
        import generator as gen
        from tokenizers.processors import TemplateProcessing
        from transformers import AutoTokenizer

        def _load_tokenizer():
            tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
            bos, eos = tok.bos_token, tok.eos_token
            tok._tokenizer.post_processor = TemplateProcessing(
                single=f"{bos}:0 $A:0 {eos}:0",
                pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
                special_tokens=[(bos, tok.bos_token_id), (eos, tok.eos_token_id)],
            )
            return tok

        gen.load_llama3_tokenizer = _load_tokenizer
        gen.load_watermarker = lambda device=None: None
        gen.watermark = lambda wm, audio, sr, key: (audio, sr)

        device = args.device if args.device in ("cuda", "cpu") else "cpu"
        generator = gen.load_miso_8b(
            device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )

        context = []
        if ref_wav and ref_text:
            wav, sr = sf.read(str(ref_wav), dtype="float32")
            ref = torch.from_numpy(np.asarray(wav))
            if ref.ndim > 1:
                ref = ref.mean(dim=1)
            if sr != SAMPLE_RATE:
                ref = torchaudio.functional.resample(ref, orig_freq=sr, new_freq=SAMPLE_RATE)
            context = [gen.Segment(speaker=0, text=ref_text, audio=ref)]
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            sampling = ({"temperature": CLONE_TEMPERATURE, "topk": CLONE_TOPK}
                        if context else {})
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            audio = generator.generate(
                text=text, speaker=0, context=context,
                max_audio_length_ms=MAX_AUDIO_MS, **sampling,
            )
            t_end = time.perf_counter()

            arr = audio.detach().cpu().float().numpy().reshape(-1).astype(np.float32)
            audio_s = float(len(arr) / SAMPLE_RATE)
            if write_wav:
                sf.write(out_path, arr, SAMPLE_RATE)

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
