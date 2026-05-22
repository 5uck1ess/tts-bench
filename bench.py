"""Quick TTS bench: time 4 models on 5 prompts, write CSV + WAVs.

Usage:
    python bench.py                                      # default voices, all available devices
    python bench.py --reference my_voice.wav             # clone a voice (also needs my_voice.txt)
    python bench.py --models pocket --prompts 1,2        # subset
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parent


PROMPTS = [
    (1, "Open the browser and read my email."),
    (2, "I'll start a new git branch, push the changes, and open a pull request when the tests pass."),
    (3,
     "The Parakeet TDT zero point six billion parameter model achieves "
     "one point six nine percent word error rate on LibriSpeech test-clean, "
     "beating Whisper Large V3 at two point seven percent while running at "
     "over two thousand times realtime on a single GPU."),
    (4, "Run pytest tests slash test underscore voice dot py with verbose flag and capture flag set to no."),
    (5, "Bonjour, je m'appelle Cicero et je vais vous aider avec votre code aujourd'hui."),
]


# (name, venv_python_relpath, runner_relpath, multilingual?, devices, variant)
MODELS = [
    ("pocket",      "venvs/pocket/Scripts/python.exe",  "runners/pocket.py",  True,  ["cpu"],         None),
    ("neutts_air",  "venvs/neutts/Scripts/python.exe",  "runners/neutts.py",  False, ["cpu", "cuda"], "air"),
    ("neutts_nano", "venvs/neutts/Scripts/python.exe",  "runners/neutts.py",  True,  ["cpu", "cuda"], "nano"),
    ("luxtts",      "venvs/luxtts/Scripts/python.exe",  "runners/luxtts.py",  False, ["cpu", "cuda"], None),
]


def detect_cuda(venv_python: Path) -> bool:
    try:
        out = subprocess.run(
            [str(venv_python), "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=30,
        )
        return "True" in out.stdout
    except Exception:
        return False


def run_one(venv_python, runner, text, out_wav, device, variant, reference) -> dict:
    cmd = [str(venv_python), str(runner), "--text", text, "--out", str(out_wav), "--device", device]
    if variant:
        cmd += ["--variant", variant]
    if reference:
        cmd += ["--reference", str(reference)]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout 180s", "wall_s": time.perf_counter() - t0}
    wall = time.perf_counter() - t0

    parsed = {"ok": False, "error": "no json line in stdout"}
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as e:
                parsed = {"ok": False, "error": f"json parse failed: {e}"}
            break
    if not parsed.get("ok") and proc.stderr:
        tail = proc.stderr.strip().splitlines()[-3:]
        parsed["error"] = (parsed.get("error", "") + " | " + " ".join(tail))[:300]
    parsed["wall_s"] = wall
    return parsed


def main() -> int:
    p = argparse.ArgumentParser(description="Quick TTS bench.")
    p.add_argument("--reference", default=None, help="Path to a reference wav for cloning (omit for each model's default voice).")
    p.add_argument("--prompts", default=None, help="Comma-sep prompt ids; default: all 5.")
    p.add_argument("--models", default=None, help="Comma-sep model names; default: all 4.")
    p.add_argument("--devices", default=None, help="Comma-sep devices to attempt; default: cpu + cuda (auto-detect).")
    args = p.parse_args()

    if args.prompts:
        wanted = {int(x) for x in args.prompts.split(",")}
        selected_prompts = [(pid, t) for pid, t in PROMPTS if pid in wanted]
    else:
        selected_prompts = list(PROMPTS)

    requested_models = set(args.models.split(",")) if args.models else None
    requested_devices = set(args.devices.split(",")) if args.devices else None

    out_dir = REPO / "results" / datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    print(f"Output: {out_dir}\n")

    rows = []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "device", "prompt_id", "ttfa_ms", "wall_s", "audio_s", "rtf", "ok", "error",
        ])
        writer.writeheader()

        for model_name, py_rel, runner_rel, multilingual, model_devices, variant in MODELS:
            if requested_models and model_name not in requested_models:
                continue

            venv_python = REPO / py_rel
            if not venv_python.exists():
                print(f"skip {model_name}: venv not installed ({py_rel})")
                continue

            cuda_ok = ("cuda" in model_devices) and detect_cuda(venv_python)
            for device in model_devices:
                if device == "cuda" and not cuda_ok:
                    continue
                if requested_devices and device not in requested_devices:
                    continue

                for prompt_id, text in selected_prompts:
                    if prompt_id == 5 and not multilingual:
                        continue

                    wav = out_dir / f"{model_name}_{device}_p{prompt_id}.wav"
                    print(f"  {model_name}/{device}/p{prompt_id} ...", end=" ", flush=True)

                    result = run_one(venv_python, REPO / runner_rel,
                                     text, wav, device, variant, args.reference)

                    ttfa = result.get("ttfa_ms")
                    audio_s = result.get("audio_s")
                    wall = result.get("wall_s", 0) or 0
                    rtf = (audio_s / wall) if (audio_s and wall) else None

                    row = {
                        "model": model_name,
                        "device": device,
                        "prompt_id": prompt_id,
                        "ttfa_ms": round(ttfa, 1) if ttfa else "",
                        "wall_s": round(wall, 3),
                        "audio_s": round(audio_s, 3) if audio_s else "",
                        "rtf": round(rtf, 2) if rtf else "",
                        "ok": result.get("ok", False),
                        "error": (result.get("error") or "")[:200],
                    }
                    writer.writerow(row)
                    f.flush()
                    rows.append(row)

                    if result.get("ok"):
                        msg = f"ttfa={ttfa:.0f}ms rtf={rtf:.2f}x" if (ttfa and rtf) else "ok"
                        print(msg)
                    else:
                        print(f"FAIL: {row['error'][:80]}")

    print(f"\nDone. CSV: {csv_path}")

    print("\n=== Summary (RTF, higher = faster than realtime) ===")
    header = f"{'model':<14} {'device':<6} " + " ".join(f"{'p' + str(i):>6}" for i in (1, 2, 3, 4, 5))
    print(header)
    by_md: dict = {}
    for r in rows:
        if not r["ok"]:
            continue
        by_md.setdefault((r["model"], r["device"]), {})[r["prompt_id"]] = r["rtf"]
    for (model, device), pmap in by_md.items():
        cells = [str(pmap.get(i, "—")) for i in (1, 2, 3, 4, 5)]
        print(f"{model:<14} {device:<6} " + " ".join(f"{c:>6}" for c in cells))

    return 0


if __name__ == "__main__":
    sys.exit(main())
