"""Flock Speech runner (Apache-2.0 acoustic models, MIT Vocos vocoder; ONNX Runtime, no torch).

Benched on **flockspeech 0.2.3** (PyPI, which bundles the v0.1 weights, so nothing
is downloaded at run time). Two single-speaker English voices, one runner; --variant
picks the voice:

    junco      clear, neutral reading voice   trained on LJSpeech
    goldfinch  conversational voice           trained on DisfluencySpeech; [laugh],
                                              [breath], [sigh], [lipsmack], [throat], [noise]

Architecture: espeak-ng phonemes (phonemizer + the espeak-ng bundled by
espeakng-loader) -> encoder (duration + prior) -> 8-step flow-matching decoder (int8)
-> Vocos vocoder -> numpy iSTFT, 24 kHz mono.

Params, counted from the ONNX initializers: encoder 3.51M + decoder 7.90M = 11.4M
acoustic per voice, plus a shared 13.5M Vocos vocoder = 24.9M per voice. The card's
"11.4M" leaves out the vocoder.

Streaming: `Voice.stream()` yields one sentence at a time (long sentences are split
again at commas past 300 chars), so TTFA is the first sentence's audio: a real
sentence-level latency, not a frame-level one. The audio concatenated from stream() is
exactly what `Voice.__call__` returns.

Determinism: `stream()` builds `np.random.default_rng(seed)` per call, so the pinned
seed (0, upstream's default) gives bit-identical reruns.

Warm-up: the phonemizer backend is created lazily on the first call (~200 ms), so a
throwaway synthesis runs during load to keep that cost out of run 0.

Threads: upstream's default is min(4, cpu_count) intra-op threads. Like every other
runner we leave the library default alone. The model's "8.7x on a single thread"
claim is their number, not ours.

CPU only: the engine hard-codes CPUExecutionProvider (CoreML is offered for the
vocoder on Apple silicon only), so the harness registers ["cpu"]. No cloning:
--reference is accepted for harness compatibility and ignored.
"""

import argparse
import json
import sys
import time

import _meminfo


VARIANTS = ("junco", "goldfinch")
DEFAULT_VARIANT = "junco"
SEED = 0  # upstream default


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Ignored — Flock Speech voices are single-speaker and cannot clone.")
    p.add_argument("--variant", default=None, choices=list(VARIANTS))
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
                          "error": f"Flock Speech is English-only, got language={args.language}"}))
        return 1
    if args.device != "cpu":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"Flock Speech runs on ONNX Runtime CPU only, got device={args.device}"}))
        return 1

    variant = args.variant or DEFAULT_VARIANT
    try:
        import flockspeech
        import numpy as np
        import soundfile as sf

        tts = flockspeech.load(variant)
        sample_rate = tts.sample_rate
        tts("Warm up.", seed=SEED)  # creates the phonemizer backend outside run 0
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            chunks = []
            first = None
            t0 = time.perf_counter()
            for chunk in tts.stream(text, seed=SEED):
                if first is None:
                    first = time.perf_counter()
                chunks.append(chunk)
            t_end = time.perf_counter()

            pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            audio_s = float(len(pcm) / sample_rate)
            if write_wav:
                sf.write(out_path, pcm, sample_rate)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (first - t0) * 1000 if first else None,
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
