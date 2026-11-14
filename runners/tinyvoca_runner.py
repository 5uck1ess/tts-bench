"""tiny-voca runner (Apache-2.0, two preset voices, English-only, no cloning).

4,537,291 parameters (verified by numel at load), trained from scratch on
synthetic Qwen3-TTS audio. No neural vocoder: the network predicts durations,
pitch and spectral envelopes and a small source-filter synth renders 24 kHz.
Text goes through the author's own phonemizer (``dizione``) rather than eSpeak.

Pins: upstream code b8703fe (2026-11-14), dizione f8629fe, HF weights 5131e95.
The checkpoint carries a front-end fingerprint; ``match_frontend`` selects the
"own" (dizione) front end it was trained with, and ``load_model`` refuses a
mismatch, so a dizione drift fails loudly at load instead of mispronouncing.
dizione finds its lexicon through ``DIZIONE_DATA``, set here to the
``phonemizer/`` folder of the downloaded weights.

Voice is pinned to F2 (upstream's default; M2 is the other preset). Upstream's
README says "reads one sentence at a time, split long text first", but the hard
limit is ``max_tokens`` = 512 phonemes and canonical prompt 3 is 224, so every
bench prompt is one call and the runner does no splitting.

``speak`` takes a seed (default 0) and is deterministic. The first call pays a
one-off initialisation (~90 ms vs ~20 ms on CPU). There is deliberately no
warm-up call at load: run 0 is the cold run (docs/architecture.md), so that
cost belongs in it.

Output is float and can exceed full scale (prompt 1 on F2 peaks at 1.077).
Upstream writes FLOAT wavs to preserve that; this runner writes the board's
usual PCM_16, so such overshoot clips exactly as it would for any user saving a
normal wav. That is the model's output, not a runner bug.

The API is non-streaming and returns the complete waveform, so TTFA == gen_s.
``--reference`` is accepted for harness compatibility and ignored.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import _meminfo


ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "venvs" / "tinyvoca" / "src" / "weights"
CHECKPOINT = WEIGHTS / "tiny-voca.pt"
VOICE = "F2"
SEED = 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Ignored — tiny-voca has two fixed voices and cannot clone.")
    p.add_argument("--variant", default=None)
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
                          "error": f"tiny-voca is English-only, got language={args.language}"}))
        return 1

    try:
        os.environ.setdefault("DIZIONE_DATA", str(WEIGHTS / "phonemizer"))
        import soundfile as sf
        from tiny_voca.infer import load_model, match_frontend, speak

        if match_frontend(CHECKPOINT) is None:
            raise RuntimeError("no installed front end matches the checkpoint fingerprint")
        model = load_model(CHECKPOINT, args.device)
        sr = model.config.sample_rate
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            wav, _phones, _details = speak(model, text, VOICE, seed=SEED)
            t_end = time.perf_counter()

            audio_s = float(len(wav) / sr)
            if write_wav:
                sf.write(out_path, wav.numpy(), sr)

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
