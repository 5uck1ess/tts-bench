"""SteerTTS runner (WhissleAI, code MIT / weights CC BY-NC 4.0, English-only).

A FastPitch-style acoustic model (attention + bidirectional Mamba-3, written in
pure PyTorch inside the package -- no CUDA kernels, so CPU and Windows work) plus
the frozen `charactr/vocos-mel-24khz` vocoder. 2,825 built-in anonymised voices
and experimental zero-shot cloning from a WavLM x-vector.

Two variants share this venv and runner; --variant picks the refiner:

    flow    -> upstream default (`flow=True`): a flow-matching refiner polishes
               the mel before Vocos. 10 steps, temp 0.4.
    noflow  -> `flow=False`: the "23M, real-time on CPU" configuration the
               author recommends for CPU use.

Params differ per row and must NOT be copied from the announcement (counted over
the safetensors at the pinned SHA):
    acoustic model.safetensors   9,585,639
    flow.safetensors (refiner)  17,642,598   preset voices, flow on
    pflow.safetensors           24,674,406   replaces the refiner when cloning, flow on
    Vocos                       13,531,650
  noflow = 23.1M, flow = 40.8M. Cloning additionally loads
  microsoft/wavlm-base-plus-sv as a frozen speaker encoder (both variants).

Weights: HF WhissleAI/steer-tts @ 53b28c07778519df0103f35d188656c5cdc3232a,
snapshotted to venvs/steertts/src/steer-tts and installed from there as a
package (`pip install ".[all]"`; [all] brings transformers for cloning).

API:
    from steertts import SteerTTS
    tts = SteerTTS.from_pretrained(model_dir, device=..., flow=...)
    wav = tts(text, flow=..., seed=0[, ref_wav=path])   # float32 numpy, 24 kHz

The constructor's `flow` decides whether the refiner weights load at all; the
per-call `flow` decides whether they run. We set both from the variant so the
noflow row's memory column doesn't carry an unused refiner.

Non-streaming: tts() returns the whole array, so TTFA == gen_s. (`stream()` is
sentence-level only and the bench prompts are mostly one sentence.)

Determinism: seed=0 and the default voice (voice_1361) are pinned. Renders were
bit-identical across repeated calls on both cpu and cuda in the smoke.

The first CUDA call in a process takes ~20 s (CUDA/cuDNN init); later calls are
~0.15 s. That lands in the cold column, as it does for every cuda row.

Cloning: the first 8 s of the reference are used; no transcript. The author
states cloning similarity is low -- that is the measured product, not a runner bug.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "venvs" / "steertts" / "src" / "steer-tts"

VARIANTS = {"flow": True, "noflow": False}
DEFAULT_VARIANT = "flow"
SEED = 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Reference wav for zero-shot cloning (first 8 s used, no transcript).")
    p.add_argument("--variant", default=None, choices=sorted(VARIANTS),
                   help="flow (default, upstream) or noflow")
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
                          "error": f"SteerTTS is English-only, got language={args.language}"}))
        return 1

    flow = VARIANTS[args.variant or DEFAULT_VARIANT]

    try:
        if not (MODEL_DIR / "model.safetensors").exists():
            raise FileNotFoundError(
                f"{MODEL_DIR} is missing model.safetensors — run install.ps1/install.sh for `steertts`")
        from steertts import SteerTTS
        import soundfile as sf

        m = SteerTTS.from_pretrained(str(MODEL_DIR), device=args.device, flow=flow)
        sample_rate = m.sample_rate
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    kw = {"flow": flow, "seed": SEED}
    if args.reference:
        kw["ref_wav"] = args.reference

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            audio = m(text, **kw)
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
