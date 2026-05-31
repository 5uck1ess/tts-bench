"""Shared model registry + subprocess plumbing for bench.py and compare.py.

A "cell" is one (model, device) pair that's actually runnable on this machine
(venv installed AND device supported by the runner AND torch reports the
device as available). Each cell call spawns the runner subprocess once,
reads JSON-line results from stdout, and returns one dict per run.
"""

import json
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent


# (name, venv_dir, runner_relpath, multilingual?, devices, variant, can_clone)
# can_clone: True  = accepts user-supplied reference wav at inference (zero-shot)
#            False = predefined voice list only (Kokoro, KittenTTS, Piper)
#            "gated" = cloning works but requires HF accept-terms login
MODELS = [
    # Zero-shot voice cloning candidates
    ("pocket",      "pocket",     "runners/pocket_runner.py",     True,  ["cpu"],                None,   "gated"),
    ("neutts_air",  "neutts",     "runners/neutts_runner.py",     False, ["cpu", "cuda", "mps"], "air",  True),
    ("neutts_nano", "neutts",     "runners/neutts_runner.py",     True,  ["cpu", "cuda", "mps"], "nano", True),
    ("luxtts",      "luxtts",     "runners/luxtts_runner.py",     False, ["cpu", "cuda", "mps"], None,   True),
    ("chatterbox",       "chatterbox", "runners/chatterbox_runner.py", False, ["cpu", "cuda", "mps"], None,    True),
    ("chatterbox_turbo", "chatterbox", "runners/chatterbox_runner.py", False, ["cpu", "cuda", "mps"], "turbo", True),
    ("f5tts",       "f5tts",      "runners/f5tts_runner.py",      False, ["cpu", "cuda", "mps"], None,   True),
    ("coqui",       "coqui",      "runners/coqui_runner.py",      True,  ["cpu", "cuda", "mps"], None,   True),
    ("omnivoice",   "omnivoice",  "runners/omnivoice_runner.py",  True,  ["cpu", "cuda", "mps"], None,   True),
    ("zipvoice",    "zipvoice",   "runners/zipvoice_runner.py",   True,  ["cpu", "cuda", "mps"], None,   True),
    ("voxcpm",      "voxcpm",     "runners/voxcpm_runner.py",     True,  ["cpu", "cuda"],        None,   True),
    # base qwentts: cloning disabled — long prompts blow the 600s cell timeout on
    # this autoregressive sampler. qwentts_fast (CUDA-graph) handles cloning instead.
    ("qwentts",      "qwentts",      "runners/qwentts_runner.py",      True,  ["cpu", "cuda"],  "base", False),
    ("qwentts_fast", "qwentts_fast", "runners/qwentts_fast_runner.py", True,  ["cuda"],         "base", True),
    ("indextts",    "indextts",   "runners/indextts_runner.py",   False, ["cpu", "cuda"],        None,   True),
    ("sesame",      "sesame",     "runners/sesame_runner.py",     False, ["cpu", "cuda"],        None,   "gated"),
    ("mars5",       "mars5",      "runners/mars5_runner.py",      False, ["cpu", "cuda"],        None,   True),
    ("dia",         "dia",        "runners/dia_runner.py",        False, ["cuda"],               None,   True),
    ("fish_15",     "fish",       "runners/fish_runner.py",       True,  ["cpu", "cuda", "mps"], None,   True),
    # Predefined-voice-only (no cloning)
    ("kokoro",      "kokoro",     "runners/kokoro_runner.py",     True,  ["cpu", "cuda", "mps"], None,   False),
    ("kittentts",   "kittentts",  "runners/kittentts_runner.py",  False, ["cpu"],                None,   False),
    ("piper",       "piper",      "runners/piper_runner.py",      True,  ["cpu", "cuda"],        None,   False),
    ("vibevoice",      "vibevoice",  "runners/vibevoice_runner.py",  False, ["cpu", "cuda", "mps"], None,    False),
    ("vibevoice_15b",  "vibevoice",  "runners/vibevoice_runner.py",  False, ["cpu", "cuda", "mps"], "1.5b", False),
    ("vibevoice_7b",   "vibevoice",  "runners/vibevoice_runner.py",  False, ["cuda"],               "7b",   True),
    ("magpie",      "magpie",     "runners/magpie_runner.py",     True,  ["cpu", "cuda"],        None,   False),
    ("soprano",     "soprano",    "runners/soprano_runner.py",    False, ["cpu", "cuda", "mps"], None,   False),
    ("moss_tts_nano", "moss_tts_nano", "runners/moss_tts_nano_runner.py", True,  ["cpu", "cuda", "mps"], None, True),
    ("moss_tts",      "moss_tts",      "runners/moss_tts_runner.py",      True,  ["cuda"],               None, True),
    ("supertonic",  "supertonic", "runners/supertonic_runner.py", True,  ["cpu"],                None,   False),
    ("maya1",       "maya1",      "runners/maya1_runner.py",      False, ["cpu", "cuda", "mps"], None,   False),
]


# GPU-class models: measured sub-realtime (<0.5x RTF) on the best available
# CPU/MPS device, or they OOM / time out there. They only make sense on a CUDA
# rig. On a machine where CUDA isn't available, build_cells skips them by
# default (pass include_gpu_class=True / `bench.py --all` to force them in) so a
# non-CUDA pass doesn't burn time on models that can't be deploy candidates
# there. Tagged from the May 2026 Apple-M4 pass; they still run on Windows/Linux
# CUDA rigs unchanged. See docs/known-issues.md.
GPU_CLASS = {
    "f5tts", "indextts", "voxcpm", "qwentts", "sesame", "mars5",
    "chatterbox", "magpie",
}


def venv_python(venv_dir: str) -> Path:
    """Resolve the python.exe / bin/python path for a venv on this OS."""
    root = REPO / "venvs" / venv_dir
    if sys.platform.startswith("win"):
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _probe(py: Path, code: str) -> bool:
    try:
        out = subprocess.run(
            [str(py), "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        return "True" in out.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False


def detect_cuda(py: Path) -> bool:
    return _probe(py, "import torch; print(torch.cuda.is_available())")


def detect_mps(py: Path) -> bool:
    return _probe(
        py,
        "import torch; b = getattr(torch.backends, 'mps', None); "
        "print(bool(b and b.is_available()))",
    )


def build_cells(reference=None, requested_models=None, requested_devices=None,
                verbose=True, include_gpu_class=False, skipped_out=None):
    """Return the list of runnable (model, device) cells on this machine.

    `requested_models` / `requested_devices`: optional sets to filter. If
    `reference` is truthy, predefined-voice-only models (can_clone=False) are
    skipped, matching bench.py's behavior. Cells with missing venvs or
    unavailable devices are silently dropped (with a printed note if verbose).

    GPU-class models (see GPU_CLASS) are skipped on a rig where CUDA isn't
    available unless `include_gpu_class` is True. When `skipped_out` is a list,
    one dict per skipped (model, device) is appended so callers can mark them in
    their output instead of dropping them silently.
    """
    cells = []
    for (model_name, venv_dir, runner_rel, multilingual,
         model_devices, variant, can_clone) in MODELS:
        if requested_models and model_name not in requested_models:
            continue
        if reference and can_clone is False:
            continue
        py = venv_python(venv_dir)
        if not py.exists():
            if verbose:
                print(f"skip {model_name}: venv not installed ({py})")
            continue
        cuda_ok = ("cuda" in model_devices) and detect_cuda(py)
        mps_ok = ("mps" in model_devices) and detect_mps(py)
        if model_name in GPU_CLASS and not cuda_ok and not include_gpu_class:
            if verbose:
                print(f"skip {model_name}: gpu-class — sub-realtime without CUDA "
                      f"on this rig (pass --all to include)")
            if skipped_out is not None:
                for device in model_devices:
                    if device == "cuda":
                        continue
                    if requested_devices and device not in requested_devices:
                        continue
                    skipped_out.append({
                        "model": model_name, "device": device,
                        "variant": variant, "can_clone": can_clone,
                        "multilingual": multilingual,
                        "reason": "gpu-class: sub-realtime without CUDA (--all to include)",
                    })
            continue
        for device in model_devices:
            if device == "cuda" and not cuda_ok:
                continue
            if device == "mps" and not mps_ok:
                continue
            if requested_devices and device not in requested_devices:
                continue
            cells.append({
                "model": model_name, "device": device, "variant": variant,
                "multilingual": multilingual, "can_clone": can_clone,
                "venv_python": py, "runner": REPO / runner_rel,
            })
    return cells


def run_cell(cell, text, out_wav, language="en", runs=1, reference=None,
             timeout=600) -> list[dict]:
    """Run one (model, device) cell with N back-to-back generations.

    Returns one dict per run from the runner's JSON-line stdout. On failure
    returns a single dict with ok=False. The first row gets a wall_s key
    measuring total subprocess wall time (load + all runs).
    """
    cmd = [
        str(cell["venv_python"]), str(cell["runner"]),
        "--text", text, "--out", str(out_wav),
        "--device", cell["device"], "--runs", str(runs),
        "--language", language,
    ]
    if cell.get("variant"):
        cmd += ["--variant", cell["variant"]]
    if reference:
        cmd += ["--reference", str(reference)]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return [{"ok": False, "error": f"timeout {timeout}s", "run_index": 0,
                 "wall_s": time.perf_counter() - t0}]
    wall = time.perf_counter() - t0

    parsed = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as e:
            parsed.append({"ok": False, "error": f"json parse failed: {e}", "run_index": -1})

    if not parsed:
        tail = " ".join(proc.stderr.strip().splitlines()[-3:])[:300] if proc.stderr else ""
        return [{"ok": False, "error": f"no json in stdout. stderr: {tail}",
                 "run_index": 0, "wall_s": wall}]

    parsed[0]["wall_s"] = wall
    return parsed


def play_wav(wav_path) -> None:
    """Play a wav file synchronously through the OS default player. Best-effort."""
    wav_path = str(wav_path)
    if sys.platform == "win32":
        try:
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
            return
        except Exception as e:
            print(f"  (playback failed: {e})")
            return
    if sys.platform == "darwin":
        subprocess.run(["afplay", wav_path], check=False)
        return
    for tool in ("aplay", "paplay"):
        try:
            subprocess.run([tool, wav_path], check=False)
            return
        except FileNotFoundError:
            continue
