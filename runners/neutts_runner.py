"""NeuTTS Air + Nano runner.

API discovered by inspection (2026-05-22):
    from neutts import NeuTTS
    tts = NeuTTS(backbone_repo="neuphonic/neutts-air", backbone_device="cpu",
                 codec_repo="neuphonic/neucodec", codec_device="cpu")
    ref_codes = tts.encode_reference(ref_wav_path)
    audio_np = tts.infer(text, ref_codes, ref_text)
    # or stream:
    for chunk in tts.infer_stream(text, ref_codes, ref_text):
        ...

We use non-GGUF backbones (transformers path) so no extra llama-cpp-python
dependency is needed. Air is English-only; Nano has en/fr/de/es variants.

Pass 1 default voices: ./reference/jo.{wav,txt} (en) and ./reference/juliette.{wav,txt} (fr).
Both shipped by upstream as samples — same voice across Air + Nano = apples-to-apples.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo
import _naq


REPO_ROOT = Path(__file__).resolve().parent.parent


# (variant, language) -> HF repo id. Q4 GGUF is Neuphonic's recommended fast
# path (via llama-cpp-python). Non-GGUF backbones exist too but are slow.
REPO_MAP = {
    ("air", "en"):  "neuphonic/neutts-air-q4-gguf",
    ("nano", "en"): "neuphonic/neutts-nano-q4-gguf",
    ("nano", "fr"): "neuphonic/neutts-nano-french-q4-gguf",
    ("nano", "de"): "neuphonic/neutts-nano-german-q4-gguf",
    ("nano", "es"): "neuphonic/neutts-nano-spanish-q4-gguf",
}


# Pass 1 default reference voices (downloaded once into ./reference/ from neutts upstream samples)
DEFAULT_REFERENCE = {
    "en": "jo",
    "fr": "juliette",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None)
    p.add_argument("--variant", default="air")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true",
                   help="Interactive mode: read JSON jobs {text, out} from stdin, one per line.")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    try:
        from neutts import NeuTTS
        import numpy as np
        import soundfile as sf

        repo = REPO_MAP.get((args.variant, args.language))
        if repo is None:
            print(json.dumps({
                "ok": False, "run_index": 0,
                "error": f"no neutts repo for variant={args.variant!r} language={args.language!r}",
            }))
            return 1

        # Resolve reference wav + transcript.
        if args.reference:
            ref_wav = Path(args.reference)
            ref_txt = ref_wav.with_suffix(".txt")
        else:
            stem = DEFAULT_REFERENCE.get(args.language)
            if not stem:
                print(json.dumps({
                    "ok": False, "run_index": 0,
                    "error": f"no default reference for language={args.language!r}",
                }))
                return 1
            ref_wav = REPO_ROOT / "reference" / f"{stem}.wav"
            ref_txt = REPO_ROOT / "reference" / f"{stem}.txt"
        if not ref_wav.exists() or not ref_txt.exists():
            print(json.dumps({
                "ok": False, "run_index": 0,
                "error": f"reference files missing: {ref_wav} / {ref_txt}",
            }))
            return 1

        tts = NeuTTS(
            backbone_repo=repo,
            backbone_device=args.device,
            codec_repo="neuphonic/neucodec",
            codec_device=args.device,
        )
        ref_text = ref_txt.read_text(encoding="utf-8").strip()
        ref_codes = tts.encode_reference(str(ref_wav))
        samplerate = 24000  # neucodec outputs 24 kHz per homebase doc
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    # Streaming only works for the GGUF/llama.cpp backend. For the torch backbone
    # we get all audio in one shot from infer() — TTFA == gen_s in that case.
    streaming = {"on": True}

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            first = None
            chunks = []
            if streaming["on"]:
                try:
                    for chunk in tts.infer_stream(text, ref_codes, ref_text):
                        if first is None:
                            first = time.perf_counter()
                        chunks.append(np.asarray(chunk))
                except NotImplementedError:
                    streaming["on"] = False
                    chunks = []
                    first = None
            if not streaming["on"]:
                audio_np = tts.infer(text, ref_codes, ref_text)
                first = time.perf_counter()
                chunks = [np.asarray(audio_np)]
            t_end = time.perf_counter()

            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
            audio_s = float(len(audio) / samplerate)
            if write_wav:
                sf.write(out_path, audio, samplerate)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (first - t0) * 1000 if first else None,
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
