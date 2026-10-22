"""KittenTTS Nano runner (Apache 2.0, English-only, 8 predefined voices, no cloning).

Re-benched 2026-10-22 on Nano 0.8 (`kittentts` 0.8.1, the GitHub release wheel --
PyPI stops at 0.1.3). The previous row was Nano 0.1 (`kittentts` 0.1.3); its rows
and wavs are kept in results/_preserved_kittentts_0.1/.

API (kittentts==0.8.x, ONNX-backed, 15M params):
    from kittentts import KittenTTS
    m = KittenTTS("KittenML/kitten-tts-nano-0.8")
    audio = m.generate(text, voice="Jasper", speed=1.0)  # -> np.ndarray, 24kHz

Weights: HF KittenML/kitten-tts-nano-0.8 @ 7a1db645b1f3ab9420761d87428e042b9cec3f26
(the package default; pinned by name here so a future default bump can't move it).

Voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo. 0.8 keeps the 0.1
names as aliases: `expr-voice-2-m` (the 0.1 row's voice) resolves to Jasper --
same-length output on prompt 3 -- so the re-bench keeps the same speaker.
CPU only: 0.8.1's constructor has no execution-provider argument.

Non-streaming: ONNX runs the full inference and returns the complete
audio array in one call. TTFA == gen_s for this model (no incremental
output). We report TTFA = gen_s so the comparison is honest.

Install gotcha #1: KittenTTS uses `phonemizer` which calls system espeak-ng.
We bundle espeak-ng via the `espeakng-loader` wheel and point phonemizer
at it via env vars before importing kittentts.

Install gotcha #2: the bundled espeak-ng DLL has a hardcoded CI build path
('D:/a/espeakng-loader/...') baked in. Setting ESPEAK_DATA_PATH to the
real path is required, otherwise espeak errors on `phontab` lookup.

0.8.1 prints "Generating audio for text: ..." to stdout on every call; the
harness skips non-JSON lines, so it is harmless.
"""

import argparse
import json
import os
import sys
import time

import _meminfo


REPO_ID = "KittenML/kitten-tts-nano-0.8"
DEFAULT_VOICE = "Jasper"
SAMPLE_RATE = 24000


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Voice NAME (e.g. 'Luna') — KittenTTS doesn't support wav cloning.")
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
                          "error": f"KittenTTS is English-only, got language={args.language}"}))
        return 1

    try:
        # MUST happen before `from kittentts import KittenTTS` or phonemizer init fails
        import espeakng_loader
        os.environ["ESPEAK_DATA_PATH"] = espeakng_loader.get_data_path()
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeakng_loader.get_library_path()
        espeakng_loader.make_library_available()
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(espeakng_loader.get_library_path())

        from kittentts import KittenTTS
        import numpy as np
        import soundfile as sf

        m = KittenTTS(REPO_ID)
        voice = args.reference or DEFAULT_VOICE
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            audio = m.generate(text, voice=voice, speed=1.0)
            t_end = time.perf_counter()

            audio_s = float(len(audio) / SAMPLE_RATE)
            if write_wav:
                sf.write(out_path, audio, SAMPLE_RATE)

            # Non-streaming: TTFA = gen_s (audio not available until run completes)
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
