"""Inflect v2 runner (Apache-2.0, English-only, one fixed voice, no cloning).

Two variants share this venv and runner; --variant picks the checkpoint dir:

    nano   -> Inflect-Nano-v2   3,958,801 deployed params, 15.97 MB FP32
    micro  -> Inflect-Micro-v2  9,344,753 deployed params, 37.53 MB FP32

Both are complete text-to-waveform packages: a compressed VITS (192 latent /
96 text hidden / 3 encoder layers / 4 flow blocks) with the 24 kHz waveform
decoder built in. No external vocoder, no second learned model.

API (repo-local, not a PyPI package -- the HF repo ships its own inference.py):
    sys.path.insert(0, model_dir)
    from inference import InflectTTS
    tts = InflectTTS(model_dir, device="cpu")
    sample_rate, waveform = tts.synthesize(text, speed=1.0, variation=0.667, seed=0)

`inference.py` puts both `model_dir` and `model_dir/runtime` on sys.path at
import time, so the generically-named `models` / `commons` / `utils` / `text`
modules resolve. Harmless here (one variant per subprocess), but it does mean
a single process cannot load both variants without clearing sys.modules.

Non-streaming: synthesize() returns the complete array in one call, so
TTFA == gen_s. Reported that way to stay honest against streaming models.

Determinism: seed is fixed at 0 so repeated bench runs of the same prompt
produce the same latent sample. `variation=0.667` is upstream's default.

Install gotcha: the frontend is phonemizer + espeak-ng, bundled via the
`espeakng-loader` wheel -- the same path KittenTTS uses. Inflect's own
`_configure_espeak()` wires it up, so no env-var dance is needed here.
Phonemizer logs a noisy "words count mismatch" WARNING on prompts containing
hyphenated tokens (bench prompt 3, "test-clean"); it is cosmetic and goes to
stderr, but we silence it so runner logs stay readable.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import _meminfo


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "venvs" / "inflect" / "src"

VARIANTS = {
    "micro": SRC / "Inflect-Micro-v2",
    "nano": SRC / "Inflect-Nano-v2",
}
DEFAULT_VARIANT = "micro"

# Upstream defaults. variation is the latent noise scale (0..1); speed is a
# length-scale multiplier (0.5..2.0).
VARIATION = 0.667
SPEED = 1.0
SEED = 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Ignored — Inflect ships one fixed voice and cannot clone.")
    p.add_argument("--variant", default=None, choices=sorted(VARIANTS),
                   help="micro (default) or nano")
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
                          "error": f"Inflect v2 is English-only, got language={args.language}"}))
        return 1

    variant = args.variant or DEFAULT_VARIANT
    model_dir = VARIANTS[variant]

    try:
        if not (model_dir / "model.pth").exists():
            raise FileNotFoundError(
                f"{model_dir} is missing model.pth — run install.ps1/install.sh for `inflect`")

        logging.getLogger("phonemizer").setLevel(logging.ERROR)
        # inference.py self-inserts model_dir and model_dir/runtime on import.
        sys.path.insert(0, str(model_dir))
        from inference import InflectTTS
        import soundfile as sf

        m = InflectTTS(str(model_dir), device=args.device)
        sample_rate = m.sample_rate
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            _, audio = m.synthesize(text, speed=SPEED, variation=VARIATION, seed=SEED)
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
