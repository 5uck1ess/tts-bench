"""sanoTTS runner (GPL-3.0, English-only preset voices, no cloning).

Two variants share this venv and runner; --variant selects the voice:

    amy         -> amy,        1.46M parameters, 22.05 kHz, uncapped input
    heart-nano  -> heart-nano, 294,279 parameters, 24 kHz, 207-token hard cap

The heart-nano frontend raises ``sanotts.FrontendError`` above 207 phoneme
tokens (including canonical bench prompt 3). The per-run error handler emits
an ok:false JSON line and lets stdin-mode benchmarking continue. Deliberately
do not chunk or sentence-split long text: upstream exposes no long-form path,
and inventing one here would make this row incomparable to its neighbours.
Also do not raise ``max_tokens``: it conditions the trained duration model via
``log1p(n) / log1p(max_tokens)``, so changing it corrupts timing for all inputs.

Construct one ``sanotts.Synthesizer`` during load and reuse it for every run.
The module-level ``sanotts.synthesize()`` reloads the voice pack on each call,
which would silently turn every measured run into a cold load and destroy the
warm column.

Both voices are deterministic with sanoTTS's internal default seed. The API is
non-streaming and returns the complete waveform, so TTFA == gen_s. ``--reference``
is accepted for harness compatibility and ignored because these voices cannot
clone.
"""

import argparse
import json
import sys
import time

import _meminfo


VARIANTS = {
    "amy": "amy",
    "heart-nano": "heart-nano",
}
DEFAULT_VARIANT = "amy"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Ignored — sanoTTS preset voices cannot clone.")
    p.add_argument("--variant", default=None, choices=sorted(VARIANTS),
                   help="amy (default) or heart-nano")
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
                          "error": f"sanoTTS voices are English-only, got language={args.language}"}))
        return 1

    variant = args.variant or DEFAULT_VARIANT
    voice = VARIANTS[variant]

    try:
        import sanotts
        import soundfile as sf

        synthesizer = sanotts.Synthesizer(voice=voice)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            result = synthesizer.synthesize(text)
            t_end = time.perf_counter()

            audio_s = float(len(result.audio) / result.sample_rate)
            if write_wav:
                sf.write(out_path, result.audio, result.sample_rate)

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
