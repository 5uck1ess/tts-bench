"""Step-Audio-EditX runner (stepfun-ai, vLLM-backed cloning, Linux-only, 24 kHz).

Source-clone install (no pip wheel) — see install.sh. Heavy stack: vLLM +
deepspeed + funasr + onnxruntime-gpu + whisper. Requires python 3.12 and
torch >= 2.9. Weights (`stepfun-ai/Step-Audio-EditX` ~8 GB +
`stepfun-ai/Step-Audio-Tokenizer` ~1.4 GB) download via huggingface_hub on
first run; cached after.

Two things are mandatory and easy to miss:
  1. VLLM_ATTENTION_BACKEND=TRITON_ATTN — the Step1 LLM uses alibi_sqrt
     attention, which the default FLASH_ATTN backend rejects
     (ValueError: use_alibi_sqrt is not supported). Must be set before vllm imports.
  2. vLLM uses `spawn` multiprocessing — the model must be built under
     `if __name__ == "__main__"` (we build it inside main()), or the spawned
     EngineCore re-imports this module and crashes bootstrapping.

API (StepAudioTTS, src not pip-importable -> add to sys.path):
    tok   = StepAudioTokenizer("stepfun-ai/Step-Audio-Tokenizer")
    model = StepAudioTTS("stepfun-ai/Step-Audio-EditX", tok, gpu_memory_utilization=0.6, ...)
    audio, sr = model.clone(prompt_wav_path=ref_wav, prompt_text=ref_txt, target_text=text)  # sr=24000

Cloning flavor: wav + transcript (like Fish). Bundled refs ship a sibling .txt.
License: Apache-2.0.
"""

import argparse
import ctypes
import glob
import json
import os
import sys
import time
from pathlib import Path

import _meminfo
import _naq

# Mandatory before any vllm import (Step1 alibi_sqrt attention).
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

# torchcodec (Step's audio I/O on torch 2.9) dynamically loads libtorchcodec_core8
# against FFmpeg 8 (libav*.so.60). On Linux those come from linuxbrew's default
# ffmpeg; preload them with ctypes(RTLD_GLOBAL) so torchcodec resolves them under
# bench.py without a process-level LD_LIBRARY_PATH. This runs at module import in
# both this process and the spawned vLLM EngineCore worker (which re-imports us).
_BREW_LIB = "/home/linuxbrew/.linuxbrew/lib"
if sys.platform.startswith("linux") and os.path.isdir(_BREW_LIB):
    for _stem in ("libavutil", "libswresample", "libswscale", "libavcodec",
                  "libavformat", "libavfilter", "libavdevice"):
        for _so in sorted(glob.glob(f"{_BREW_LIB}/{_stem}.so.*")):
            try:
                ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

# StepAudioTTS / StepAudioTokenizer live at the source-clone root, not as an
# installed top-level package — put the src dir on the path.
_SRC = Path(__file__).resolve().parent.parent / "venvs" / "step_editx" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

EDITX_REPO = "stepfun-ai/Step-Audio-EditX"
TOKENIZER_REPO = "stepfun-ai/Step-Audio-Tokenizer"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for cloning. A sibling .txt (the reference "
                        "transcript) is required by Step's clone().")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    # Default-voice path: borrow a bundled reference (clone-only model). Step's
    # clone() needs the reference transcript, so a sibling .txt is required.
    repo = Path(__file__).resolve().parent.parent
    if args.reference:
        ref_wav = Path(args.reference)
    else:
        default_ref = {"en": "jo.wav", "fr": "juliette.wav"}.get(args.language, "jo.wav")
        ref_wav = repo / "reference" / default_ref
    ref_txt_path = ref_wav.with_suffix(".txt")
    if not ref_wav.exists() or not ref_txt_path.exists():
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"need both reference wav and sibling .txt: {ref_wav} / {ref_txt_path}"}))
        return 1
    ref_text = ref_txt_path.read_text().strip()

    try:
        import soundfile as sf
        from tokenizer import StepAudioTokenizer
        from tts import StepAudioTTS

        tok = StepAudioTokenizer(TOKENIZER_REPO)
        # gpu_memory_utilization is vLLM's reservation; the CosyVoice vocoder +
        # FunASR + onnxruntime run in THIS process, outside that budget, and their
        # peak grows with reference length AND across back-to-back runs. On a 24 GB
        # card with a long (67 s) ref + 3 runs/cell, 0.5 still OOMs on the longest
        # prompt — so cap vLLM at 0.4 and enforce_eager (frees CUDA-graph capture)
        # to leave maximum headroom for the vocoder.
        model = StepAudioTTS(
            EDITX_REPO, tok,
            gpu_memory_utilization=0.4, max_model_len=8192, max_num_seqs=1,
            enforce_eager=True, dtype="bfloat16",
        )
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            audio, sr = model.clone(prompt_wav_path=str(ref_wav),
                                    prompt_text=ref_text, target_text=text)
            t_end = time.perf_counter()

            arr = audio.squeeze().detach().cpu().float().numpy()
            audio_s = float(len(arr) / sr)
            if write_wav:
                sf.write(out_path, arr, sr)

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
