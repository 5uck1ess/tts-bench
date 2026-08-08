"""Vaniq-Edge runner (MIT, English-only, one fixed voice, no cloning).

    Abiray/Vaniq-Edge   8,912,048 deployed params, 24 kHz

A compressed VITS with the waveform decoder built in -- complete text-to-waveform,
no external vocoder. Same class as Inflect v2, but an independent training, not a
repackage: 128 latent / 128 text hidden / 384 filter / 4 encoder layers / 256
upsample-initial channels, and crucially `use_sdp: true` (stochastic duration
predictor) where Inflect uses the deterministic one.

API (repo-local, not a PyPI package -- the HF repo ships its own inference.py):
    sys.path.insert(0, model_dir)
    from inference import VaniqTTS
    tts = VaniqTTS(model_dir, device="cpu")
    sample_rate, waveform = tts.synthesize(text, length_scale=1.0)

`inference.py` puts `model_dir/vits_core` on sys.path inside __init__, so the
generically-named `models` / `commons` / `utils` / `text` modules resolve. Harmless
in a dedicated subprocess, but it does mean this runner cannot share a process with
another VITS-derived model.

THE IMPORT WALL (why _stub_monotonic_align exists): vits_core/models.py does a
module-scope `import monotonic_align`, and that package's __init__ imports a
prebuilt Cython extension shipped ONLY as `core.cpython-312-x86_64-linux-gnu.so`.
That is unimportable on Windows, on Mac, and on any Python that isn't 3.12 --
including the 3.11 Linux venv. The function it provides, `maximum_path`, is used
exactly once (models.py line 480) inside the TRAINING forward and never inside
infer(), so loading the real extension is unnecessary. We register a stub module in
sys.modules before the import can fire; the stub raises if anything ever calls it,
so a future upstream change that needs it fails loudly instead of silently.

DETERMINISM -- the reason every generate pins a seed. `synthesize()` exposes no seed
argument and the model has a stochastic duration predictor, so consecutive calls on
the same text produce DIFFERENT AUDIO LENGTHS. Measured unseeded on prompt 1: two
calls, two different sample counts. With torch.manual_seed(0) before each call the
output is bit-identical run to run (verified cpu and cuda). Without this the warm
runs would not be measuring the same work as the cold run.

Non-streaming: synthesize() returns the complete array in one call, so TTFA == gen_s.
Reported that way to stay honest against streaming models.

No generation cap to worry about: VITS predicts durations rather than decoding
tokens, so length is unbounded by any max_new_tokens-style ceiling. Canonical
prompt 3 renders in full at 19.50 s of audio.

Text normalization: upstream's `_normalize_text` spells out numbers via num2words
and then strips anything outside [\\w\\s.,!?'-]. Colons and semicolons are dropped
rather than voiced as pauses. Left as-is -- it is the model's own frontend and the
bench should measure the shipped pipeline.

Frontend: phonemizer + espeak-ng via the `espeakng-loader` wheel (the KittenTTS /
Inflect stack). inference.py wires it up itself, so no env-var dance is needed.
Loading also emits a cosmetic torch `weight_norm is deprecated` FutureWarning on
stderr; silenced so runner logs stay readable.
"""

import argparse
import json
import logging
import sys
import time
import types
import warnings
from pathlib import Path

import _meminfo


REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "venvs" / "vaniq" / "src" / "Vaniq-Edge"

# Upstream default. length_scale is a duration multiplier (lower = faster speech).
LENGTH_SCALE = 1.0
SEED = 0


def _stub_monotonic_align() -> None:
    """Pre-empt vits_core/models.py's module-scope `import monotonic_align`.

    The shipped Cython .so is linux-x86_64/cp312 only and the function is
    training-only. See the module docstring for the full reasoning.
    """
    stub = types.ModuleType("monotonic_align")

    def _training_only(*_args, **_kwargs):
        raise RuntimeError(
            "monotonic_align.maximum_path is training-only and is stubbed out in the "
            "bench runner; inference must not reach it")

    stub.maximum_path = _training_only
    sys.modules["monotonic_align"] = stub


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Ignored — Vaniq-Edge ships one fixed voice and cannot clone.")
    p.add_argument("--variant", default=None, help="Ignored — single checkpoint.")
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
                          "error": f"Vaniq-Edge is English-only, got language={args.language}"}))
        return 1

    try:
        if not (MODEL_DIR / "model.pth").exists():
            raise FileNotFoundError(
                f"{MODEL_DIR} is missing model.pth — run install.ps1/install.sh for `vaniq`")

        logging.getLogger("phonemizer").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*weight_norm.*")
        _stub_monotonic_align()

        sys.path.insert(0, str(MODEL_DIR))
        import torch
        from inference import VaniqTTS
        import soundfile as sf

        m = VaniqTTS(str(MODEL_DIR), device=args.device)
        sample_rate = m.sample_rate
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            # Pinned per call: the stochastic duration predictor otherwise changes
            # the output length between identical prompts. See module docstring.
            torch.manual_seed(SEED)
            t0 = time.perf_counter()
            _, audio = m.synthesize(text, length_scale=LENGTH_SCALE)
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
