"""Sopro V2 Turbo runner (Apache-2.0, multilingual zero-shot cloning).

Sopro has no preset voice: synthesize() without ``ref`` or
``ref_audio_path`` raises ``ValueError("provide ref or ref_audio_path")``. It is
therefore a cloning-only model (registered in NO_PRESET_VOICE). When ``--reference``
is omitted this runner falls back to the repo's house reference wav, matching the
convention of the other cloning-only runners, so the default lens still yields a
comparable speed row rather than a failure cell. Reference is wav only -- no
sibling transcript is needed.

Two engine variants share one checkpoint but select different vocoders:

    offline   -> tts.synthesize(), vocoder.safetensors
    streaming -> tts.stream(), vocoder_streaming.safetensors
                 (causal Vocos with three frames of lookahead)

The paths are not bit-exact. Streaming TTFA is measured when the first yielded
audio chunk is ready. Offline generation returns the complete waveform, so its
reported TTFA equals gen_s (expressed in milliseconds), not genuine streaming
latency.

The true output sample rate is ``tts.sample_rate == 24000``. Sampling is
stochastic at the upstream defaults, so torch's CPU seed and, on CUDA, all CUDA
seeds are reset before every generation for reproducible reruns. Sopro's default
segment cap is 300 characters; canonical bench prompt 3 is 253 characters, so it
is synthesized as one complete segment rather than split or token-truncated.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


MODEL_ID = "samuel-vitorino/sopro-v2-turbo"
DEFAULT_VARIANT = "offline"
VARIANTS = ("offline", "streaming")
LANGUAGES = ("en", "pt", "fr", "de")
SEED = 0


def _emit_failure(error: str, run_index: int = 0) -> None:
    print(json.dumps({"ok": False, "run_index": run_index, "error": error}),
          flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="5-20 second reference wav; no transcript needed. Defaults to the house reference.")
    p.add_argument("--variant", default=DEFAULT_VARIANT, choices=VARIANTS)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        _emit_failure("either --stdin or both --text and --out are required")
        return 1
    if args.device not in ("cpu", "cuda"):
        _emit_failure(f"Sopro supports cpu or cuda in this harness, got device={args.device}")
        return 1
    if args.language not in LANGUAGES:
        _emit_failure(
            f"Sopro supports languages {', '.join(LANGUAGES)}, got language={args.language}")
        return 1
    # Default-voice path: Sopro has no preset voice at all (synthesize() without a
    # reference raises), so borrow the house reference the way every other
    # cloning-only runner here does (qwentts_fast, zonos2, echo, indextts, ...).
    # Without this the default lens would be failure cells for sopro alone, while
    # every sibling cloning model still produces a comparable speed row.
    repo = Path(__file__).resolve().parent.parent
    ref_wav = Path(args.reference) if args.reference else repo / "reference" / "chris_hemsworth_15s.wav"
    if not ref_wav.is_file():
        _emit_failure(f"reference wav not found: {ref_wav}")
        return 1

    try:
        import torch
        from sopro import SoproTTS

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda but torch.cuda.is_available() == False")

        tts = SoproTTS.from_pretrained(
            repo_or_path=MODEL_ID,
            device=args.device,
            dtype=torch.float32,
        )
        sample_rate = int(tts.sample_rate)
        if sample_rate != 24000:
            raise RuntimeError(f"unexpected Sopro sample rate: {sample_rate}")
        reference = tts.prepare_reference(
            ref_audio_path=str(ref_wav),
            stream=(args.variant == "streaming"),
        )
    except Exception as e:
        _emit_failure(f"load failed: {type(e).__name__}: {e}")
        return 1

    def _sync_cuda() -> None:
        if args.device == "cuda":
            torch.cuda.synchronize()

    def _seed() -> None:
        torch.manual_seed(SEED)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(SEED)

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            _seed()
            _sync_cuda()
            t0 = time.perf_counter()

            if args.variant == "streaming":
                chunks = []
                ttfa_ms = None
                for chunk in tts.stream(text, ref=reference, lang=args.language):
                    if ttfa_ms is None:
                        _sync_cuda()
                        ttfa_ms = (time.perf_counter() - t0) * 1000
                    chunks.append(chunk.detach().reshape(-1))
                _sync_cuda()
                t_end = time.perf_counter()
                if not chunks:
                    raise RuntimeError("Sopro streaming produced no audio chunks")
                wav = torch.cat(chunks)
            else:
                wav = tts.synthesize(text, ref=reference, lang=args.language)
                _sync_cuda()
                t_end = time.perf_counter()
                ttfa_ms = (t_end - t0) * 1000

            wav = wav.detach().reshape(-1)
            gen_s = t_end - t0
            audio_s = float(wav.numel() / sample_rate)
            if write_wav:
                tts.save_wav(str(out_path), wav)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": ttfa_ms,
                "gen_s": gen_s,
                "audio_s": audio_s,
                **_meminfo.sample(args.device),
            }), flush=True)
            return True
        except Exception as e:
            _emit_failure(f"{type(e).__name__}: {e}", run_index)
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
                text = job["text"]
                out_path = job["out"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                _emit_failure(f"stdin job parse: {type(e).__name__}: {e}", idx)
                idx += 1
                continue
            _one(text, out_path, idx, write_wav=True)
            idx += 1
        return 0

    for i in range(args.runs):
        if not _one(args.text, args.out, i, write_wav=(i == 0)):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
