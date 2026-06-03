"""Higgs Audio v2 runner (Boson AI, Apache-2.0, 3.6B LLM + 2.2B audio adapter, in-context cloning).

Higgs Audio v2 is a text-audio foundation model (DualFFN LLM + unified audio tokenizer).
Zero-shot cloning is done IN-CONTEXT via ChatML: show the model a (reference transcript ->
reference audio) turn, then ask for the target text in that voice. So cloning needs the
reference wav AND its transcript (a `.txt` with the same basename, like f5tts / neutts).

API (model card Quick Start + examples/generation.py voice-clone path):
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
    engine = HiggsAudioServeEngine("bosonai/higgs-audio-v2-generation-3B-base",
                                   "bosonai/higgs-audio-v2-tokenizer", device="cuda")
    messages = [
        Message(role="system",    content=SYSTEM_PROMPT),
        Message(role="user",      content=ref_transcript),
        Message(role="assistant", content=AudioContent(audio_url=str(ref_wav))),
        Message(role="user",      content=target_text),
    ]
    out = engine.generate(chat_ml_sample=ChatMLSample(messages=messages), max_new_tokens=1024,
                          temperature=0.3, top_p=0.95, top_k=50,
                          stop_strings=["<|end_of_text|>", "<|eot_id|>"])
    # out.audio (1-D numpy), out.sampling_rate

CUDA-only in practice: ~5.8B params (3.6B LLM + 2.2B adapter) — fits the 5090 (32GB) and
3090 (24GB, fp16), fails Mac 16GB. CPU/MPS cells fail cleanly here. The direct serve
engine path needs NO vLLM (vLLM is an optional throughput backend upstream).

License: Apache-2.0 (weights + code).
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import _meminfo
import _naq


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REF = REPO_ROOT / "reference" / "chris_hemsworth_15s.wav"

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
SYSTEM_PROMPT = (
    "Generate audio following instruction.\n\n"
    "<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
)

_REAL_STDOUT = sys.stdout


def _emit(obj) -> None:
    print(json.dumps(obj), file=_REAL_STDOUT, flush=True)


def _read_transcript(ref_wav: Path) -> str | None:
    txt = ref_wav.with_suffix(".txt")
    if txt.exists():
        return txt.read_text(encoding="utf-8").strip()
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for in-context cloning; needs a same-basename .txt transcript.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        _emit({"ok": False, "run_index": 0,
               "error": "either --stdin or both --text and --out are required"})
        return 1

    ref_wav = Path(args.reference) if args.reference else DEFAULT_REF
    if not ref_wav.exists():
        _emit({"ok": False, "run_index": 0, "error": f"reference wav not found: {ref_wav}"})
        return 1
    ref_transcript = _read_transcript(ref_wav)
    if ref_transcript is None:
        _emit({"ok": False, "run_index": 0,
               "error": f"in-context cloning needs a transcript: {ref_wav.with_suffix('.txt')} not found"})
        return 1

    try:
        import torch

        if args.device != "cuda":
            _emit({"ok": False, "run_index": 0,
                   "error": f"Higgs Audio v2 is CUDA-only here (~5.8B); device={args.device}"})
            return 1
        if not torch.cuda.is_available():
            _emit({"ok": False, "run_index": 0, "error": "CUDA requested but not available"})
            return 1

        with contextlib.redirect_stdout(sys.stderr):
            from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
            from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
            engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device="cuda")
    except Exception as e:
        _emit({"ok": False, "run_index": 0,
               "error": f"load failed: {type(e).__name__}: {e}"})
        return 1

    def _messages(target_text: str):
        return [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=ref_transcript),
            Message(role="assistant", content=AudioContent(audio_url=str(ref_wav))),
            Message(role="user", content=target_text),
        ]

    def _one(text: str, out_path: str, run_index: int, write_wav: bool) -> bool:
        try:
            import numpy as np
            import soundfile as sf

            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                output = engine.generate(
                    chat_ml_sample=ChatMLSample(messages=_messages(text)),
                    max_new_tokens=1024, temperature=0.3, top_p=0.95, top_k=50,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                )
            t_end = time.perf_counter()

            clip = np.asarray(output.audio, dtype=np.float32).reshape(-1)
            sr = int(output.sampling_rate)
            audio_s = float(len(clip) / sr)

            if write_wav:
                sf.write(str(out_path), clip, sr)

            _emit({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
                **_meminfo.sample(args.device),
                **(_naq.score(out_path) if write_wav else
                   {"naq": None, "naq_artifact": None, "naq_naturalness": None}),
            })
            return True
        except Exception as e:
            _emit({"ok": False, "run_index": run_index,
                   "error": f"{type(e).__name__}: {e}"})
            return False

    if args.stdin:
        idx = 0
        _emit({"ready": True})
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError as e:
                _emit({"ok": False, "run_index": idx, "error": f"json parse: {e}"})
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
