"""LuxTTS runner.

Loads the model once, then does --runs generations back-to-back. Prints one JSON
line per run on stdout. Writes the WAV from run 0 only.

LuxTTS ships as the `zipvoice` package (the GitHub repo `ysharma3501/LuxTTS`
packages itself that way); the model class is `zipvoice.luxvoice.LuxTTS`. It is a
prompt-cloning flow-matching model: encode a reference wav once with
`encode_prompt(path)` (it auto-transcribes the prompt), then call
`generate_speech(text, encode_dict)`, which returns a 48 kHz torch tensor.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


REPO_ROOT = Path(__file__).resolve().parent.parent

# LuxTTS clones whatever reference it's given (and auto-transcribes it) and cannot
# speak without one, so it has no model-native preset voice: a default-voice run is
# a clone of the house reference (see NO_PRESET_VOICE in publish.py -> cloning board
# only). Was jo (en) / juliette (fr) until 2026-07-29, which put a stranger's voice
# on the Default board -- see docs/known-issues.md.
DEFAULT_REFERENCE_STEM = "chris_hemsworth_15s"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")  # accepted for harness uniformity
    args = p.parse_args()

    # Resolve the reference voice to clone (default-voice mode uses the shared sample).
    if args.reference:
        ref_wav = Path(args.reference)
    else:
        ref_wav = REPO_ROOT / "reference" / f"{DEFAULT_REFERENCE_STEM}.wav"
    if not ref_wav.exists():
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference wav not found: {ref_wav}"}))
        return 1

    try:
        from zipvoice.luxvoice import LuxTTS  # type: ignore  # repo packages itself as `zipvoice`
        import soundfile as sf

        tts = LuxTTS(device=args.device)
        # Encode the reference once (it auto-transcribes), then reuse for every run.
        encode_dict = tts.encode_prompt(str(ref_wav))
        samplerate = 48000  # vocos returns 48 kHz when return_smooth=False (the default)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    for i in range(args.runs):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            # One-shot generation (no streaming API), so TTFA == total gen time.
            wav = tts.generate_speech(args.text, encode_dict)
            t_end = time.perf_counter()
            audio = wav.squeeze().detach().cpu().numpy()
            audio_s = float(len(audio) / samplerate)

            if i == 0:
                sf.write(args.out, audio, samplerate)

            print(json.dumps({
                "ok": True,
                "run_index": i,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0,
                "audio_s": audio_s,
                **_meminfo.sample(args.device),
            }), flush=True)
        except Exception as e:
            print(json.dumps({
                "ok": False, "run_index": i,
                "error": f"{type(e).__name__}: {e}",
            }), flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
