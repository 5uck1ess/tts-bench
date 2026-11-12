"""VoxMel runner (VoxDance-Lab; Apache-2.0 weights; torch, CUDA or CPU).

VoxMel is a tokenizer-free autoregressive mel TTS: a Qwen2.5-initialised decoder with a
flow-matching head generates Coarse-Mel (a two-axis Haar split of a 100-bin log-mel),
two non-autoregressive tiers restore the temporal and frequency residuals, and a frozen
Vocos (charactr/vocos-mel-24khz) renders 24 kHz audio. Trained on Emilia (100K h), EN+ZH.

Two sizes, one runner; --variant picks the checkpoint:

    06b  VoxDance-Lab/VoxMel-0.6B  model_L_final.pt   690.1M excl. vocoder (README breakdown)
    03b  VoxDance-Lab/VoxMel-0.3B  model_M_final.pt

Everything is pinned: the source clone (venvs/voxmel/src, commit below, installed -e), and
both checkpoints + Vocos by HF revision. Upstream does not download weights itself
(`from_pretrained` takes a local dir), so the runner resolves the pinned snapshots.

Reference-only (NO_PRESET_VOICE): cloning is audio continuation and needs the exact
transcript (sibling .txt). The API has a no-prompt path, but it invents an unconditioned
voice per call rather than a model-native preset, so the no-reference run clones the
house Chris reference instead, like f5tts/indextts.

Sampling: config defaults (32 flow steps, text CFG 3.5, speaker CFG 3.0) = the demo's
settings. seed 42 = the demo's fixed seed; `generate(seed=)` re-seeds every call, so
reruns are reproducible. `max_seconds` defaults to 30 s per call, a ceiling far above
the longest bench prompt (p3 ~15-19 s), so no cap is raised.

Non-streaming: ttfa_ms == gen_s. bf16 autocast is upstream's on CUDA; CPU runs fp32.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import _meminfo


REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "venvs" / "voxmel" / "src"
SRC_COMMIT = "2588e79a28ca1722bb40c43815c7aae19ef9945b"
CHECKPOINTS = {
    "06b": ("VoxDance-Lab/VoxMel-0.6B", "b64c10f97bad8a2d3b0cf4257379c405e975f684", "model_L_final.pt"),
    "03b": ("VoxDance-Lab/VoxMel-0.3B", "432750958d3321988956d16247449b4cab548c61", "model_M_final.pt"),
}
VOCOS = ("charactr/vocos-mel-24khz", "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21")
DEFAULT_VARIANT = "06b"
DEFAULT_REF = REPO / "reference" / "chris_hemsworth_15s.wav"
SEED = 42


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav + sibling .txt transcript (defaults to the house reference).")
    p.add_argument("--variant", default=None, choices=list(CHECKPOINTS))
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
                          "error": f"VoxMel is benched in English only, got language={args.language}"}))
        return 1

    ref_wav = Path(args.reference) if args.reference else DEFAULT_REF
    txt = ref_wav.with_suffix(".txt")
    if not ref_wav.exists() or not txt.exists():
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference wav + sibling .txt required: {ref_wav}"}))
        return 1
    ref_text = txt.read_text(encoding="utf-8").strip()

    variant = args.variant or DEFAULT_VARIANT
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from huggingface_hub import snapshot_download
        from voxcore import VoxMel

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        repo_id, rev, ckpt = CHECKPOINTS[variant]
        model_dir = snapshot_download(repo_id, revision=rev)
        vocos_dir = snapshot_download(VOCOS[0], revision=VOCOS[1])
        # config.yaml names the Qwen architecture by a path relative to examples/voxmel
        # ("qwen_tokenizer", "qwen_configs/qwen2_medium_300m"), so load from there.
        cwd = os.getcwd()
        os.chdir(SRC / "examples" / "voxmel")
        try:
            tts = VoxMel.from_pretrained(
                model_dir, checkpoint=ckpt,
                tokenizer_path=str(SRC / "examples" / "voxmel" / "qwen_tokenizer"),
                vocos_path=vocos_dir, device=args.device)
        finally:
            os.chdir(cwd)
        sample_rate = tts.sample_rate
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            _sync()
            t0 = time.perf_counter()
            wav = tts.generate(text, prompt_wav_path=str(ref_wav), prompt_text=ref_text, seed=SEED)
            _sync()
            gen_s = time.perf_counter() - t0
            wav = np.asarray(wav, dtype=np.float32)
            audio_s = float(len(wav) / sample_rate)
            if write_wav:
                sf.write(out_path, wav, sample_rate)
            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": gen_s * 1000, "gen_s": gen_s, "audio_s": audio_s,
                **_meminfo.sample(args.device),
            }), flush=True)
            return True
        except Exception as e:
            print(json.dumps({"ok": False, "run_index": run_index,
                              "error": f"{type(e).__name__}: {e}"}), flush=True)
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
