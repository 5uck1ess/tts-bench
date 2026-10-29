"""Fish Audio Drama 3 runner, through the third-party ComfyUI-Drama3 port (not Fish's code).

Fish ships Drama 3 (35.9B MoE, 3B active) only as an SGLang-Omni server for H200-class
GPUs. ComfyUI-Drama3 (PedroMarinhoDev, GPL-3, pinned 8cb3c64) reimplements inference
and keeps the MoE experts in system RAM with a GPU expert cache, so it runs on a
32 GB or 24 GB card. Every number this row produces measures THAT port: its speed is
the offload engine (since 2026-10-17: NVRTC kernels read cached experts in place and
missed ones straight from RAM, every layer replays as a CUDA graph; 46 ms/frame =
realtime). The original port's math was checked against Fish's fish_omni reference
(KL 0.0041, known-issues); later port versions were not re-checked.

Weights: fishaudio/drama-3 @ 78e5ab7 (the 2026-10-29 re-upload), repacked locally with the
port's own `tools/convert.py model <dir> drama3_fp8experts_bf16_78e5ab7.safetensors --format fp8`
(lossless: the checkpoint's FP8-E4M3 expert bytes + block scales copied, bf16 rest), and
`drama3_codec.safetensors` (Fish S2-Pro codec, 44.1 kHz, 21.5 frames/s), in models/drama3/.
Fish replaced the LM weights on 2026-10-29 without a note: same tensor names, shapes and
total size, but 909 of 912 repacked tensors differ (experts, attention, norms, embeddings,
audio head); codec.pth and tokenizer unchanged. The
2026-10-15 row used PedroMarinhoDev/Drama-3-ComfyUI @ 9063c07 (`drama3_fp8experts_bf16
.safetensors`), a repack of the original upload; convert.py on that upload reproduces it
tensor for tensor. Those rows live in results/_preserved_drama3_9063c07/.
The W4A8/int8 quants exist; they are not what this row measures.

In-process, no ComfyUI server: ComfyUI v0.42.5 is checked out at
venvs/drama3/ComfyUI and imported as a library (the port's `drama3/` package uses
comfy.ops / model_management / text_encoders). The port's node classes are called
directly. ComfyUI's executor runs every node under torch.inference_mode(); a direct
call without it dies in qwen35.py's `torch.add(..., out=)` with an autograd error.
A library import also skips main.py's DynamicVRAM (comfy-aimdo) startup, without which
the CUDA graphs never engage and the new port runs at the OLD speed (p3 cloned 0.6x vs
2.6x warm); `_enable_dynamic_vram` repeats it and refuses to load if it fails.
Port defaults the runner overrides: Decode's `shorten_long_pauses` (post-processing,
off here) and Generate's `top_k` (0 since 2026-10-16; 30 here, see Sampling). The port
now tidies the voice clip (Fish Space's `tidy_reference`: edge trim + peak normalize)
before encoding it; that input-side step stays on, and it changed the cloned takes.

Voice: no model-native preset. Without a reference the model invents a voice per
prompt (smoke p3/p5 came out at a different level and timbre), so like fish_s2 the
no-reference path clones the house chris_hemsworth_15s ref (wav + sibling .txt) and
the row is in NO_PRESET_VOICE (cloning board only).

Sampling = the model card's benchmark settings (the first port's defaults): temperature
1.0, top_p 0.9, top_k 30, seed pinned at 0. max_frames 600 per ~300-byte text chunk
(27.9 s); canonical prompt 3 (253 bytes, one chunk) is ~436 frames (20.3 s) on this port,
~495 (23.0 s) on 66c4fb9. The ORIGINAL weights ran Spanish prompt 6 into this cap (27.9 s,
a runaway); 78e5ab7 ends it at 7.4 s.

Non-streaming: codes are generated in full and then decoded, so TTFA == gen_s. The first
take of a process captures the CUDA graphs and fills the cache, so cold runs are slow.
Expert cache 16 GB on every rig so rigs stay comparable. Peak on the 5090: ~18 GB VRAM,
~71 GB system RAM (pinned experts + the loaded file).
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
MODEL_FILE = "drama3_fp8experts_bf16_78e5ab7.safetensors"
CODEC_FILE = "drama3_codec.safetensors"
DEFAULT_REF = REPO / "reference" / "chris_hemsworth_15s.wav"

EXPERT_CACHE_GB = 16.0
TEMPERATURE, TOP_P, TOP_K = 1.0, 0.9, 30
MAX_FRAMES = 600
SEED = 0


def _enable_dynamic_vram():
    """ComfyUI main.py's DynamicVRAM (comfy-aimdo) startup, which a library import skips.

    The port's decode CUDA graphs need `comfy.memory_management.aimdo_enabled`; without it the
    kernels still compile but every frame takes the old copy path at the old speed (measured:
    p3 cloned 0.6x either way). Must run before torch touches CUDA, as in main.py."""
    sys.path.insert(0, str(COMFY))
    from comfy.cli_args import args, enables_dynamic_vram
    import comfy_aimdo.control
    if not enables_dynamic_vram():
        raise RuntimeError("ComfyUI says DynamicVRAM is disabled; the drama3 row measures the dynamic-VRAM path")
    comfy_aimdo.control.init(
        simple_vram_headroom=None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3),
        nvml_pressure=not args.disable_nvml_pressure)
    import comfy.memory_management
    import comfy.model_management
    import comfy.model_patcher
    if not comfy_aimdo.control.init_devices((d.index, int(args.vram_headroom * 1024 ** 3))
                                            for d in comfy.model_management.get_all_torch_devices()):
        raise RuntimeError("comfy-aimdo failed to initialise the CUDA device (DynamicVRAM unavailable)")
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True


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
            _enable_dynamic_vram()
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
                # shorten_long_pauses off: it is output post-processing (Fish's Space SilenceGuard), not the model.
                audio = nodes.Drama3Decode.execute(codec, codes, shorten_long_pauses=False).args[0]
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
