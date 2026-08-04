"""Audio8 TTS Preview 0.6B (Apache-2.0, 11 languages, zero-shot cloning).

DualAR architecture (explicitly credited to Fish Audio S2 Pro): a 24-layer slow
AR transformer emits one semantic token per audio frame, a 4-layer fast AR
transformer fills that frame's 10 codec codebooks. 601,159,424 params excluding
the codec; the bundled 44.1 kHz codec handles both reference encoding and
waveform decode, so there is no second checkpoint to install.

Two variants share this venv and runner. --variant picks the INFERENCE ENGINE,
not the weights -- both run the same Audio8/Audio8-TTS-Preview-0.6b checkpoint,
which is the whole point: it makes the speed delta attributable to the engine.

    base      -> stock transformers AutoModel.generate (eager reference path)
    fastpath  -> ScrappyLabs `fast_arktts` static-shape rewrite + torch.compile
                 (https://huggingface.co/scrappylabsai/audio8-tts-fastpath)

`fastpath` upstream defaults to their own `scrappylabsai/warble` fine-tune; we
override model_id to the base checkpoint so the two rows are the same weights.

API (base):
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,
                                      dtype=dtype).eval().to(device)
    inputs = processor(text=[...], reference_audio=[wav], reference_text=[txt],
                       return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=..., return_dict_in_generate=True)
    wavs, lens = model.decode_audio(out.codes)

API (fastpath):
    from fast_arktts import FastTTS
    tts = FastTTS(model_id=..., compile_mode=..., max_new_tokens=...)
    audio, sr = tts.speak(text, reference_audio=..., reference_text=..., seed=...)

MAX_NEW_TOKENS is 1024, NOT the fast path's upstream default of 400. The codec
runs ~21.5 frames/s, so 400 frames caps generation at ~18.6 s -- and bench
prompt 3 measured 390 frames when cloned on this rig. That is a 10-frame margin
before silent truncation, and `dynamic=False` bakes the cap into the compiled
graph. 1024 gives ~47 s of headroom. Cost is a larger preallocated KV cache, not
slower generation: the loop still exits at EOS.

Determinism: built-in-voice (no-reference) generation is sampling-sensitive on
this checkpoint -- upstream warns a bare `<|speaker:N|>` line can degenerate on
some draws. We pin SEED and a fixed speaker tag so repeated bench runs of the
same prompt produce the same audio.

Non-streaming: generate() returns the complete waveform in one call, so
TTFA == gen_s. Reported that way to stay honest against streaming models.

Language: the processor infers language from the text itself (no language arg),
so `--language` is accepted and used only to reject unsupported codes.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "venvs" / "audio8" / "src" / "Audio8-TTS-Preview-0.6b"
FASTPATH_DIR = REPO / "venvs" / "audio8" / "src" / "fastpath"

VARIANTS = ("base", "fastpath")
DEFAULT_VARIANT = "base"

# Cantonese, Chinese, Dutch, English, French, German, Italian, Japanese,
# Korean, Polish, Spanish (the Preview checkpoint's stated coverage).
SUPPORTED_LANGS = {"yue", "zh", "nl", "en", "fr", "de", "it", "ja", "ko", "pl", "es"}

# ~21.5 frames/s -> 1024 frames is ~47 s. See module docstring for why not 400.
MAX_NEW_TOKENS = 1024

# Built-in voice selector. Upstream's documented example tag; pinned with SEED
# so the "default voice" row is a stable speaker across prompts and reruns.
SPEAKER_TAG = "<|speaker:2|>"
SEED = 0

# Sampling params from the model card's reference snippet.
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 50

# torch.compile mode for the fastpath variant. Compilation happens once at load
# (warmup=True), i.e. outside the timed generate window -- same accounting as
# qwentts_fast's CUDA-graph capture. Requires Triton; on Windows that means the
# `triton-windows` wheel (PyPI `triton` is Linux-only).
COMPILE_MODE = "max-autotune"


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
    p.add_argument("--variant", default=None, choices=VARIANTS,
                   help="base (transformers eager) or fastpath (compiled fast_arktts)")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    if args.language not in SUPPORTED_LANGS:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"Audio8 TTS Preview does not support language={args.language}"}))
        return 1

    variant = args.variant or DEFAULT_VARIANT

    # The fast path is a torch.compile/CUDA-graph rewrite -- there is no CPU or
    # MPS story. Fail the cell cleanly rather than silently running eager and
    # publishing it as if it were the compiled engine.
    if variant == "fastpath" and args.device != "cuda":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"fastpath variant is CUDA-only (torch.compile), got device={args.device}"}))
        return 1

    ref_text = _read_ref_transcript(args.reference)
    if args.reference and not ref_text:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference {args.reference} provided but sibling .txt "
                                   f"transcript missing (Audio8 cloning needs wav + matching .txt)"}))
        return 1

    try:
        import torch
        import soundfile as sf

        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"{MODEL_DIR} is missing — run install.ps1/install.sh for `audio8`")

        dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

        if variant == "fastpath":
            if not (FASTPATH_DIR / "fast_arktts.py").exists():
                raise FileNotFoundError(
                    f"{FASTPATH_DIR} is missing fast_arktts.py — run install for `audio8`")
            sys.path.insert(0, str(FASTPATH_DIR))
            from fast_arktts import FastTTS
            # Point the fast path at the BASE weights (upstream defaults to
            # their own `warble` fine-tune) so this row is engine-vs-engine.
            engine = FastTTS(model_id=str(MODEL_DIR), compile_mode=COMPILE_MODE,
                             max_new_tokens=MAX_NEW_TOKENS, device=args.device,
                             dtype=dtype, warmup=True)
            sample_rate = int(engine.model.config.codec_sample_rate)
        else:
            from transformers import AutoModel, AutoProcessor
            processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
            model = AutoModel.from_pretrained(
                str(MODEL_DIR), trust_remote_code=True, dtype=dtype,
            ).eval().to(args.device)
            sample_rate = int(model.config.codec_sample_rate)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _generate(text):
        """Return the waveform as a numpy array. Both engines, one shape."""
        torch.manual_seed(SEED)
        if variant == "fastpath":
            kw = {}
            if args.reference:
                kw["reference_audio"] = str(args.reference)
                kw["reference_text"] = ref_text
            else:
                text = SPEAKER_TAG + text
            audio, _ = engine.speak(text, seed=SEED, **kw)
            return audio

        kw = {"text": [text if args.reference else SPEAKER_TAG + text]}
        if args.reference:
            kw["reference_audio"] = [str(args.reference)]
            kw["reference_text"] = [ref_text]
        inputs = processor(return_tensors="pt", **kw)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                                 do_sample=True, return_dict_in_generate=True)
            wavs, lens = model.decode_audio(out.codes)
        return wavs[0, : int(lens[0])].float().cpu().numpy()

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            audio = _generate(text)
            t_end = time.perf_counter()

            audio_s = float(len(audio) / sample_rate)
            if write_wav:
                sf.write(out_path, audio, sample_rate)

            # Non-streaming: TTFA = gen_s (no audio until the call returns).
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
