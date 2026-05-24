"""Qwen3-TTS-Base 1.7B via faster-qwen3-tts (CUDA-graph fast path).

Sibling to qwentts_runner.py — same model weights (Qwen/Qwen3-TTS-12Hz-1.7B-Base),
different inference engine. Uses torch.cuda.CUDAGraph capture via the
faster-qwen3-tts package (MIT, @andimarafioti) for a reported 6-10x decode speedup
over vanilla transformers inference.

CUDA-only: the package raises ValueError on non-CUDA devices; we catch that and
emit a graceful {"ok": false} JSON line (exit 0) matching the harness contract.

API (faster-qwen3-tts==0.2.6):
    from faster_qwen3_tts import FasterQwen3TTS
    model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device="cuda",
        dtype=torch.bfloat16,
    )
    # generate() (default voice) is NotImplementedError — must use cloning.
    # We borrow the same jo.wav/juliette.wav fallback as qwentts_runner.py.
    wavs, sr = model.generate_voice_clone(
        text="...",
        language="English",
        ref_audio=ref_wav_path,
        ref_text=ref_text,
    )

Note: CUDA graphs are captured on the first generate_voice_clone() call (warmup),
so we time only that single call per run — warmup happens inside from_pretrained()
and the first call captures + runs in one step. The timer therefore includes
graph-capture overhead on the very first run; subsequent runs in --runs N>1 are
pure graph-replay and will be faster.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo
import _naq


# Map our ISO-style language codes to Qwen3-TTS's full names.
LANG_MAP = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


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
                   help="Wav path for zero-shot voice cloning. Needs sibling .txt transcript.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    # CUDA-only guard: fail gracefully before touching torch.
    if args.device != "cuda":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "faster-qwen3-tts requires CUDA; "
                                   f"device '{args.device}' not supported"}))
        return 0

    # Check CUDA availability before importing heavy deps.
    try:
        import torch
        if not torch.cuda.is_available():
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": "faster-qwen3-tts requires CUDA but torch.cuda.is_available() == False"}))
            return 0
    except ImportError as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"torch import failed: {e}"}))
        return 0

    # Default-voice path: borrow a bundled reference so Base (clone-only) can run.
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

    ref_text = _read_ref_transcript(str(ref_wav))
    if not ref_text:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference transcript missing: {ref_wav.with_suffix('.txt')} "
                                   f"(Qwen3-TTS Base needs wav + matching .txt)"}))
        return 1

    language = LANG_MAP.get(args.language, "Auto")

    try:
        import soundfile as sf
        import numpy as np
        from faster_qwen3_tts import FasterQwen3TTS

        model = FasterQwen3TTS.from_pretrained(
            MODEL_ID,
            device="cuda",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",  # flash-attn has no Windows wheels
        )
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=str(ref_wav),
                ref_text=ref_text,
            )
            t_end = time.perf_counter()
            gen_s = t_end - t0

            # API returns list_of_arrays; single text → single array.
            arr = wavs[0] if isinstance(wavs, list) else wavs
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            arr = np.asarray(arr).reshape(-1).astype(np.float32)
            audio_s = float(len(arr) / sr)

            if write_wav:
                sf.write(out_path, arr, sr)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                # For non-streaming generation TTFA == gen_s (full decode before any audio)
                "ttfa_ms": gen_s * 1000,
                "gen_s": gen_s,
                "audio_s": audio_s,
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
