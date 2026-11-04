"""LinnetTTS runner (VoxRT; Rust runtime via the `linnet-tts` Python binding, no torch).

Benched on **linnet-tts 0.1.1**, source commit 5ddd09c03464d2236f5e59b9695a63a38ba306af,
weights VoxRT/linnet-en-35m @ 7f0d42b860416d330488f16fcaacba5ab5a7424e (pinned below).
41.4M params: a 31.5M flow LM over a 25 Hz continuous-latent codec plus a 9.9M
Vocos-style decoder, 24 kHz mono, English only.

Windows: PyPI 0.1.1 ships wheels for manylinux x86_64/aarch64 and macOS arm64 only,
with no sdist. On Windows the wheel is built from the pinned source commit with
maturin (needs rustup + MSVC Build Tools; ~15 s compile) -- see install.ps1. Same
crate, same abi3 binding as the PyPI wheel.

API (0.1.1):
    m = linnet_tts.Model(weights_fp16, tokenizer)
    v = m.load_voice("voices/linnet-en-f1.npy")          # [T, 48] voice latent
    s = m.session(v, temp=0.3, cfg=2.0, max_frames=..., seed=0)
    s.prime(text); for chunk in s: ...                   # 80 ms per LM step

Streaming: TTFA is measured on the session (first chunk), with the clock started
BEFORE `prime()` -- prime does the text work, so starting after it reads ~3 ms.
The concatenated chunks are sample-identical to `synthesize()` (checked).

No cloning: the encoder that turns a recording into a voice latent is not released,
and the weights license reserves cloning to a separate written license. --reference
is ignored; the row uses built-in voice `linnet-en-f1` (upstream's own benchmark
voice). The other voice, m1, stops prompt 3 early (8.4 s vs f1's 15.4 s).

Generation cap: default max_frames=300 steps = 24 s. f1 prompt 3 measured 192 steps
(15.36 s); we raise the cap to 600 for margin -- the loop exits at EOS, so output is
identical (checked: same audio at 300 and 1000).

Threads: library default (`threads=0`, automatic; reports 4 on the 9950X3D). Output is
bit-identical for any thread count per upstream. CPU only: no GPU backend exists.

Determinism: temp/cfg/seed pinned at upstream defaults; reruns are bit-identical.
"""

import argparse
import json
import sys
import time

import _meminfo

REPO = "VoxRT/linnet-en-35m"
REVISION = "7f0d42b860416d330488f16fcaacba5ab5a7424e"
VOICE = "linnet-en-f1"
TEMP, CFG, SEED = 0.3, 2.0, 0
MAX_FRAMES = 600


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Ignored -- the voice encoder is not released (licence-reserved).")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1
    if args.device != "cpu":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"linnet has no device={args.device} (CPU-only runtime)"}))
        return 1
    if args.language != "en":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"linnet is English-only (language={args.language})"}))
        return 1
    try:
        import linnet_tts
        import numpy as np
        import soundfile as sf
        from huggingface_hub import snapshot_download

        d = snapshot_download(REPO, revision=REVISION, allow_patterns=[
            "linnet-en-35m.fp16.safetensors", "tokenizer.model", f"voices/{VOICE}.npy"])
        model = linnet_tts.Model(f"{d}/linnet-en-35m.fp16.safetensors", f"{d}/tokenizer.model")
        voice = model.load_voice(f"{d}/voices/{VOICE}.npy")
        sample_rate = model.sample_rate
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            chunks = []
            first = None
            t0 = time.perf_counter()
            s = model.session(voice, temp=TEMP, cfg=CFG, max_frames=MAX_FRAMES, seed=SEED)
            s.prime(text)
            for chunk in s:
                if first is None:
                    first = time.perf_counter()
                chunks.append(chunk)
            t_end = time.perf_counter()

            pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            if write_wav:
                sf.write(out_path, pcm, sample_rate)
            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (first - t0) * 1000 if first else None,
                "gen_s": t_end - t0, "audio_s": float(len(pcm) / sample_rate),
                **_meminfo.sample(args.device),
            }), flush=True)
            return True
        except Exception as e:
            print(json.dumps({"ok": False, "run_index": run_index,
                              "error": f"{type(e).__name__}: {e}"}), flush=True)
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
