"""ribt runner (Municipal Robotics; Rust + tang tensor IR, no torch, no Python API).

Benched on source commit e729c299bd095817577a47b4bb004361ac1d1843 (muni-works/ribt) and
weights municipal-robotics/ribt @ 901d7cf930f4d24261d0e5a97c1697158be59571 (release.toml
date 2026-10-17). ~54M params: 18.65M acoustic (durations, pitch, rough 100-band mel) +
21.88M one-step distilled flow decoder + 13.5M Vocos vocoder, 24 kHz mono, English only.
The voice is distilled from Kokoro-82M af_heart (135 h of renders); no Kokoro weights ship.

There is no Python package: this runner drives the `ribt-tts` CLI that install.ps1 /
install.sh build with cargo, one binary per backend:
    venvs/ribt/target-cuda/release/ribt-tts[.exe]   (--features cuda; nvrtc at load)
    venvs/ribt/target-cpu/release/ribt-tts[.exe]    (--features cpu; C lowered + compiled
                                                     with the system cc at load)
On Windows the CPU backend needs a gcc-compatible `cc` (it calls `cc -shared -fPIC -O3
-march=native`; MSVC cl and clang-msvc reject those flags). We use WinLibs GCC
(winget BrechtSanders.WinLibs.POSIX.UCRT); set RIBT_CC to override the lookup.

Mode: `synth` (whole utterance), with upstream's shipped settings from release.toml /
the README: --nfe 1 --shortcut --temp 0.3 --seed 0 --pitch 1 --accent minnesota.
Non-streaming, so ttfa_ms == gen_s. We do NOT use `stream --stdin`: it re-runs the
encoder on the whole token prefix after every word, so when a full sentence arrives at
once its first chunk lands after ~450 ms on prompt 3 (CUDA, warm) and the total is ~7x
slower than `synth` (~75 ms). Upstream's 17.6 ms TTFA is a short sentence on Metal.

Timing: each runner call is ONE `ribt-tts synth --lines` process holding `--runs` copies
of the text, so runs 2..N reuse the compiled programs (shapes are fixed per token/frame
bucket, compiled on first use). Each run's gen_s is the wall time between consecutive
per-utterance stderr lines (eprintln is unbuffered); run 0 starts at the "loaded
weights" line, so the cold run includes device init + program compile -- that IS
ribt's cold start (~5-7 s on CUDA, nvrtc; the CPU .so cache in ~/.cache/tang-cpu makes
later CPU colds cheaper). Model/weight load is excluded, like the other runners.

Threads (CPU): upstream's default is TANG_THREADS=1 (one core; a robot shares its CPU).
We set TANG_THREADS to the physical core count, the same default torch uses for the
other CPU rows. Output is bit-identical at any thread count (checked, prompt 3: 1 vs 16
threads). Measured on the 9950X3D, prompt 3 warm: 1 thread 5.1 s (3.4x), 16 threads
0.47 s (37x).

Voice / cloning: the acoustic model has a reference encoder (a speaker vector pooled from
a 24 kHz clip, first ~6 s after a 0.3 s skip). The default voice is upstream's bundled
`ref.wav` (an af_heart corpus clip) -- a genuine preset, the voice it was fine-tuned on.
--reference is resampled to 24 kHz mono and passed as --ref; it moves the voice partway
(house male ref: median F0 147 Hz vs 204 Hz default, prompt 2) but the model was
fine-tuned on one speaker, so expect weak similarity. No transcript is used.

Memory: peak_mem_mb is the CLI child's peak RSS (polled), not this wrapper's.
peak_vram_mb stays None (no torch to ask; the CLI does not report it).

Determinism: seed pinned; reruns are bit-identical (checked, CPU prompt 3).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import _meminfo  # noqa: F401  (kept for the shared contract; child memory is sampled here)

ROOT = Path(__file__).resolve().parent.parent
HOME = ROOT / "venvs" / "ribt"
WEIGHTS = HOME / "weights"
EXE = ".exe" if os.name == "nt" else ""
SR = 24000
SYNTH_ARGS = ["--nfe", "1", "--shortcut", "--temp", "0.3", "--seed", "0",
              "--pitch", "1", "--accent", "minnesota"]
UTT_RE = re.compile(r"^(r\d+)_\S*: \d+ tokens")


def _binary(device: str) -> Path:
    return HOME / f"target-{device}" / "release" / f"ribt-tts{EXE}"


def _find_cc() -> str | None:
    if os.environ.get("RIBT_CC"):
        return os.environ["RIBT_CC"]
    if os.name != "nt":
        return shutil.which("cc") or shutil.which("gcc")
    hit = shutil.which("gcc")
    if hit:
        return hit
    pkgs = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for d in sorted(pkgs.glob("BrechtSanders.WinLibs*")):
        g = d / "mingw64" / "bin" / "gcc.exe"
        if g.is_file():
            return str(g)
    return None


def _env(device: str) -> dict:
    env = dict(os.environ)
    if device == "cpu":
        cc = _find_cc()
        if not cc:
            raise RuntimeError("cpu backend needs a gcc-compatible cc (set RIBT_CC)")
        env["TANG_CC"] = cc
        # libgomp / libgcc DLLs live next to gcc on Windows
        env["PATH"] = str(Path(cc).parent) + os.pathsep + env.get("PATH", "")
        try:
            import psutil
            n = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        except ImportError:
            n = os.cpu_count() or 1
        env.setdefault("TANG_THREADS", str(n))
    if os.name == "nt" and "HOME" not in env:
        env["HOME"] = env.get("USERPROFILE", "")
    return env


def _prep_ref(reference: str | None, tmp: Path) -> Path:
    if not reference:
        return WEIGHTS / "ref.wav"
    import numpy as np
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    x, sr = sf.read(reference, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    out = tmp / "ref24k.wav"
    sf.write(out, np.clip(x, -1.0, 1.0), SR, subtype="PCM_16")
    return out


def _synth(texts, device, ref, tmp):
    """Run one synth process over `texts`; return [(gen_s, wav_path)], peak_rss_mb."""
    lines = tmp / "lines.tsv"
    lines.write_text("".join(f"r{i}\t{t}\n" for i, t in enumerate(texts)), encoding="utf-8")
    out_dir = tmp / "out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [str(_binary(device)), "synth", "--run", str(WEIGHTS), "--ckpt", "acoustic.safetensors",
           "--flow", str(WEIGHTS / "flow.safetensors"), "--voc", str(WEIGHTS / "vocoder.safetensors"),
           "--ref", str(ref), *SYNTH_ARGS, "--lines", str(lines), "--out-dir", str(out_dir)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            env=_env(device), cwd=str(tmp))
    peak = [0.0]

    def _poll():
        try:
            import psutil
            p = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    mi = p.memory_info()
                    peak[0] = max(peak[0], getattr(mi, "peak_wset", 0) or mi.rss)
                except psutil.Error:
                    break
                time.sleep(0.05)
        except ImportError:
            pass

    th = threading.Thread(target=_poll, daemon=True)
    th.start()
    t_prev = None
    marks = {}
    log = []
    for raw in proc.stderr:
        now = time.perf_counter()
        line = raw.decode("utf-8", "replace").rstrip()
        log.append(line)
        if line.startswith("loaded ") and t_prev is None:
            t_prev = now
            continue
        m = UTT_RE.match(line)
        if m and t_prev is not None:
            marks[m.group(1)] = now - t_prev
            t_prev = now
    rc = proc.wait()
    th.join(timeout=1)
    if rc != 0 or len(marks) != len(texts):
        raise RuntimeError(f"ribt-tts exit {rc}, {len(marks)}/{len(texts)} utterances: "
                           + " | ".join(log[-5:]))
    res = []
    for i in range(len(texts)):
        wavs = sorted(out_dir.glob(f"r{i}_*.wav"))
        if not wavs:
            raise RuntimeError(f"no wav for r{i}")
        res.append((marks[f"r{i}"], wavs[0]))
    return res, round(peak[0] / (1024 * 1024), 1) if peak[0] else None


def _emit(run_index, gen_s, wav, out, device, peak_mb, write):
    import soundfile as sf
    info = sf.info(str(wav))
    if write:
        shutil.copyfile(wav, out)
    print(json.dumps({"ok": True, "run_index": run_index, "ttfa_ms": gen_s * 1000,
                      "gen_s": gen_s, "audio_s": info.frames / info.samplerate,
                      "peak_mem_mb": peak_mb, "peak_vram_mb": None}), flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1
    if args.device not in ("cpu", "cuda"):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"ribt runner supports cpu/cuda, not {args.device}"}))
        return 1
    if args.language != "en":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"ribt is English-only (language={args.language})"}))
        return 1
    if not _binary(args.device).is_file():
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"missing {_binary(args.device)} (run install for ribt)"}))
        return 1

    with tempfile.TemporaryDirectory(prefix="ribt_") as td:
        tmp = Path(td)
        try:
            ref = _prep_ref(args.reference, tmp)
        except Exception as e:
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": f"reference prep failed: {type(e).__name__}: {e}"}))
            return 1

        if args.stdin:
            # one process per job: each job is a cold start (texts differ)
            print(json.dumps({"ready": True}), flush=True)
            idx = 0
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    job = json.loads(line)
                    res, peak = _synth([job["text"]], args.device, ref, tmp)
                    _emit(idx, res[0][0], res[0][1], job["out"], args.device, peak, True)
                except Exception as e:
                    print(json.dumps({"ok": False, "run_index": idx,
                                      "error": f"{type(e).__name__}: {e}"}), flush=True)
                idx += 1
            return 0

        try:
            res, peak = _synth([args.text] * args.runs, args.device, ref, tmp)
        except Exception as e:
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": f"{type(e).__name__}: {e}"}), flush=True)
            return 1
        for i, (gen_s, wav) in enumerate(res):
            _emit(i, gen_s, wav, args.out, args.device, peak, write=(i == 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
