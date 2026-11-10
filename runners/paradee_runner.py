"""Paradee-8M v1.3 runner (Apache-2.0, 7.88M params, preset voices, no cloning).

Paradee is Kokoro-82M distilled to 7.88M parameters (4.23M text side + 3.65M decoder),
one int8 ONNX graph for eight Kokoro voices in four languages. The voice picks the
language, so the runner maps bench language -> the only voice upstream ships for it:

    en -> af_heart  (upstream default; same voice as the `kokoro` row's default)
    fr -> ff_siwis
    es -> em_alex

Hindi (hf_alpha / hm_omega) exists but is not a bench language.

Versions pinned: package git commit af5ca1a (install.ps1 / install.sh), and the package
itself downloads the weights from HF tag `v1.3` (resolved b569f5e on 2026-11-10).
Stock settings: int8 graph, `threads=1` (upstream default — the card's ~62x is
one-thread), CPUExecutionProvider hard-coded upstream, so CPU-only.

The package reads long text one sentence at a time (`Paradee.__call__` -> `_chunks`).
That is upstream's own long-form path, not something added here.

Text -> phonemes is misaki (English) and eSpeak NG via misaki's EspeakG2P (fr/es),
bundled through `espeakng-loader`, so no system eSpeak is needed. misaki 0.9.x does not
install on Python 3.13+, hence the 3.11 venv.

Deterministic (no sampling). Non-streaming — returns the full waveform — so
TTFA == gen_s. ``--reference`` is accepted and ignored: Paradee cannot clone.
"""

import argparse
import json
import sys
import time

import _meminfo


VOICE_FOR_LANG = {"en": "af_heart", "fr": "ff_siwis", "es": "em_alex"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Ignored — Paradee preset voices cannot clone.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    voice = VOICE_FOR_LANG.get(args.language)
    if voice is None:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"Paradee has no voice for language={args.language}"}))
        return 1
    if args.device != "cpu":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"Paradee is CPU-only upstream, got device={args.device}"}))
        return 1

    try:
        import soundfile as sf
        from paradee import Paradee, SAMPLE_RATE

        # No warm-up call: run 0 is the board's cold column, and the sibling runners
        # (sanotts, auroratts) don't pre-warm either.
        tts = Paradee(version="1.3")
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            audio = tts(text, voice=voice)
            t_end = time.perf_counter()

            audio_s = float(len(audio) / SAMPLE_RATE)
            if write_wav:
                sf.write(out_path, audio, SAMPLE_RATE)

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
