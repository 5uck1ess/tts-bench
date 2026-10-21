"""Phonon runner (Gradium; Rust `xn` inference runtime via the `ptts` PyPI wheel, no torch).

Benched on **ptts 0.4.0**. Phonon is a Pocket-TTS derivative (Kyutai's CALM line,
the same family as our `pocket` row) re-implemented on the `xn` Rust tensor
runtime. Two sizes, one runner; --variant picks the checkpoint family:

    31m  gradium/phonon-31m-en              English only          Apache-2.0
    40m  gradium/phonon-40m-{en,fr,es}      one checkpoint/lang   CC BY-NC 4.0 (gated)

The 40m row uses upstream's per-language checkpoints (their headline config), not
`phonon-40m-multilingual`. The 40m repos are gated: accept the terms on each HF page
(approval is automatic) with the account whose token this machine uses.

Every checkpoint is pinned by revision below — upstream ships new weights under the
same repo name. The 31m pin is the one `ptts.DEFAULT_MODEL` carries in 0.4.0.

API (ptts 0.4.0):
    import ptts
    tts = ptts.TTS("hf://gradium/phonon-31m-en@<sha>", lang="en", seed=..., temperature=...)
    pcm = tts.synth(text)                       # float32 mono, tts.sample_rate (24 kHz)
    with tts.stream(text) as audio:             # chunks as they are decoded
        for chunk in audio: ...

Streaming: TTFA is measured on `tts.stream()` (first decoded chunk), the same way the
pocket runner measures it, so it IS a streaming latency (~45-70 ms on the 9950X3D).

No cloning: Phonon's voice cloning is Gradium's hosted API (`scripts/clone-voice.py`
uploads the wav with a GRADIUM_API_KEY and returns a voice file). Nothing local
encodes a reference, so --reference is ignored and the model has no cloning row.
Each checkpoint's own default voice is used.

Devices: the PyPI wheels for Windows/Linux are CPU-only (`ptts.available_devices()`
== ['cpu']). CUDA/Vulkan exist only as Cargo features in a from-source build, so on
Linux `install.sh` builds the same 0.4.0 sdist with `--features cuda` when nvcc and
cargo are present. That build's CPU path matches the PyPI wheel (31m warm: 8.6x vs
8.4x RTF on the 5900XT), so one venv serves both cells. A --device the installed
build does not list is reported as a failed run, not silently run on another device.
peak_vram_mb stays empty on cuda: `_meminfo` reads it from torch, and this venv has none.

Threads: the runtime defaults to one thread per logical core (32 on the 9950X3D).
Like every other runner we leave the library default alone, but note it is not the
fastest setting here: prompt 3 measured ~7x RTF at 32 threads vs ~10-12x at 4-16
(`ptts.set_num_threads(n)` only takes effect if called BEFORE `ptts.TTS(...)`;
calling it after load silently changes nothing).

Determinism: seed and temperature are passed explicitly at upstream's 0.4.0
defaults, so reruns of a prompt produce identical audio.
"""

import argparse
import json
import sys
import time

import _meminfo


CHECKPOINTS = {
    ("31m", "en"): "gradium/phonon-31m-en@8c64b55ec301728e2d1695bfea87a08a267d0c2d",
    ("40m", "en"): "gradium/phonon-40m-en@9b95393286eaeeebf1330b2ae90a14a2074c415d",
    ("40m", "fr"): "gradium/phonon-40m-fr@65b5b451d235a8c24245f51d2774ec9806a9443f",
    ("40m", "es"): "gradium/phonon-40m-es@eceb8758b5ed899bd661febecffdc6be61d8e1e6",
}
DEFAULT_VARIANT = "31m"

# ptts 0.4.0 TTS() defaults, pinned so a future default change can't move a row.
SEED = 4242424242424242
TEMPERATURE = 0.3


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Ignored — Phonon cloning is Gradium's hosted API, not local.")
    p.add_argument("--variant", default=None, choices=["31m", "40m"])
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    variant = args.variant or DEFAULT_VARIANT
    repo = CHECKPOINTS.get((variant, args.language))
    if repo is None:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"phonon {variant} has no checkpoint for language={args.language}"}))
        return 1
    try:
        import ptts
        import soundfile as sf

        if args.device not in ptts.available_devices():
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": f"this ptts build has no device={args.device} "
                                       f"(available: {ptts.available_devices()})"}))
            return 1
        tts = ptts.TTS(f"hf://{repo}", lang=args.language, device=args.device,
                       seed=SEED, temperature=TEMPERATURE)
        sample_rate = tts.sample_rate
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            import numpy as np
            _meminfo.reset_peak(args.device)
            chunks = []
            first = None
            t0 = time.perf_counter()
            with tts.stream(text) as audio:
                for chunk in audio:
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
