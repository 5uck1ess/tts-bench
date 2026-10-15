"""Fish Audio Drama 3 runner, through the third-party ComfyUI-Drama3 port (not Fish's code).

Fish ships Drama 3 (35.9B MoE, 3B active) only as an SGLang-Omni server for H200-class
GPUs. ComfyUI-Drama3 (PedroMarinhoDev, GPL-3, pinned 66c4fb9) reimplements inference
and keeps the MoE experts in system RAM with a GPU expert cache, so it runs on a
32 GB or 24 GB card. Every number this row produces measures THAT port: its speed
is the offload engine (single-core-bound kernel launches, ~46 ms/frame = realtime),
and its audio has not been verified against Fish's own output.

Weights: PedroMarinhoDev/Drama-3-ComfyUI @ 9063c07, `drama3_fp8experts_bf16.safetensors`
(38.6 GB, the original release precision: FP8-E4M3 experts + bf16 rest) and
`drama3_codec.safetensors` (Fish S2-Pro codec, 44.1 kHz, 21.5 frames/s), in
models/drama3/. The W4A8/int8 quants exist; they are not what this row measures.

In-process, no ComfyUI server: ComfyUI v0.40.0 is checked out at
venvs/drama3/ComfyUI and imported as a library (the port's `drama3/` package uses
comfy.ops / model_management / text_encoders). The port's node classes are called
directly. ComfyUI's executor runs every node under torch.inference_mode(); a direct
call without it dies in qwen35.py's `torch.add(..., out=)` with an autograd error.

Voice: no model-native preset. Without a reference the model invents a voice per
prompt (smoke p3/p5 came out at a different level and timbre), so like fish_s2 the
no-reference path clones the house chris_hemsworth_15s ref (wav + sibling .txt) and
the row is in NO_PRESET_VOICE (cloning board only).

Sampling = the port's defaults, which it documents as Fish's server defaults:
temperature 1.0, top_p 0.9, top_k 30, seed pinned at 0. max_frames 600 per ~300-byte
text chunk (27.9 s); canonical prompt 3 measured 308 (default) / 361 (cloned) frames.

Non-streaming: codes are generated in full and then decoded, so TTFA == gen_s.
Expert cache 16 GB on every rig (peak ~24 GB on the 5090) so rigs stay comparable.
"""

import argparse
import contextlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import _meminfo

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO / "venvs" / "drama3" / "ComfyUI"
PORT = COMFY / "custom_nodes" / "ComfyUI-Drama3"
WEIGHTS = REPO / "models" / "drama3"
MODEL_FILE = "drama3_fp8experts_bf16.safetensors"
CODEC_FILE = "drama3_codec.safetensors"
DEFAULT_REF = REPO / "reference" / "chris_hemsworth_15s.wav"

EXPERT_CACHE_GB = 16.0
TEMPERATURE, TOP_P, TOP_K = 1.0, 0.9, 30
MAX_FRAMES = 600
SEED = 0


def _import_port():
    sys.path.insert(0, str(COMFY))
    spec = importlib.util.spec_from_file_location(
        "drama3_port", PORT / "__init__.py", submodule_search_locations=[str(PORT)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["drama3_port"] = pkg
    spec.loader.exec_module(pkg)
    from drama3_port import nodes
    from drama3_port.drama3.loader import load_codec, load_drama3
    return nodes, load_codec, load_drama3


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1
    if args.device != "cuda":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"drama3 is CUDA-only, got device={args.device}"}))
        return 1

    ref_wav = Path(args.reference) if args.reference else DEFAULT_REF
    ref_txt = ref_wav.with_suffix(".txt")
    if not ref_wav.exists() or not ref_txt.exists():
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference wav + sibling .txt required: {ref_wav}"}))
        return 1

    try:
        # ComfyUI prints startup notices to stdout; keep stdout JSON-only.
        with contextlib.redirect_stdout(sys.stderr):
            import soundfile as sf
            import torch
            nodes, load_codec, load_drama3 = _import_port()
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            for f in (MODEL_FILE, CODEC_FILE):
                if not (WEIGHTS / f).exists():
                    raise FileNotFoundError(f"{WEIGHTS / f} missing — run install for `drama3`")
            model = load_drama3(str(WEIGHTS / MODEL_FILE), experts_in_ram=True,
                                expert_cache_gb=EXPERT_CACHE_GB)
            codec = load_codec(str(WEIGHTS / CODEC_FILE))
            wav, sr = sf.read(str(ref_wav), dtype="float32", always_2d=True)
            voice = nodes.Drama3EncodeVoice.execute(
                codec, {"waveform": torch.from_numpy(wav.T.copy())[None], "sample_rate": sr},
                ref_txt.read_text(encoding="utf-8").strip()).args[0]
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                codes = nodes.Drama3Generate.execute(
                    model, text, {"voice0": voice}, SEED, TEMPERATURE, TOP_P, TOP_K,
                    MAX_FRAMES).args[0]
                audio = nodes.Drama3Decode.execute(codec, codes).args[0]
            torch.cuda.synchronize()
            t_end = time.perf_counter()
            w = audio["waveform"][0, 0].numpy()
            audio_s = float(len(w) / audio["sample_rate"])
            if write_wav:
                sf.write(out_path, w, audio["sample_rate"])
            # Non-streaming: TTFA = gen_s (no audio until generate + decode return).
            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
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
