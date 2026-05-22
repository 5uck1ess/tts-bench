"""Quick TTS bench: cold + warm timings for 4 models on 5 prompts.

Loop order is prompt-outer so for each prompt you see all models back-to-back
(easier to grab side-by-side clips for video).

Each cell (model × device × prompt) is one subprocess that loads the model once,
then generates N times. Run 1 = cold (JIT not primed yet). Runs 2..N = warm.

Usage:
    python bench.py                                # default voices, all available devices
    python bench.py --reference my_voice.wav       # clone a voice (also needs my_voice.txt next to it)
    python bench.py --models pocket --prompts 1,2  # subset
    python bench.py --runs 5                       # 1 cold + 4 warm per cell (default 3 = 1c + 2w)
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
    except (subprocess.TimeoutExpired, OSError):
        return False


def run_cell(venv_python, runner, text, out_wav, device, variant, reference, runs) -> list[dict]:
    """Run one (model, device, prompt) cell with N runs in a single subprocess.

    Returns a list of dicts — one per run. Each has at minimum: ok, run_index,
    and on success: ttfa_ms, gen_s, audio_s.
    """
    cmd = [str(venv_python), str(runner),
           "--text", text, "--out", str(out_wav),
           "--device", device, "--runs", str(runs)]
    if variant:
        cmd += ["--variant", variant]
    if reference:
        cmd += ["--reference", str(reference)]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return [{"ok": False, "error": "timeout 300s", "run_index": 0,
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

    # If the runner died partway through, the surviving rows will have ok=False
    # or fewer than `runs` entries. That's fine — bench just records what came back.
    parsed[0]["wall_s"] = wall  # only meaningful for the first (cold) row
    return parsed


def main() -> int:
    p = argparse.ArgumentParser(description="Quick TTS bench (cold + warm).")
    p.add_argument("--reference", default=None,
                   help="Reference wav for voice cloning (omit for each model's default voice).")
    p.add_argument("--prompts", default=None, help="Comma-sep prompt ids; default: all 5.")
    p.add_argument("--models", default=None, help="Comma-sep model names; default: all 4.")
    p.add_argument("--devices", default=None,
                   help="Comma-sep devices to attempt; default: cpu + cuda (auto-detect).")
    p.add_argument("--runs", type=int, default=3,
                   help="Generations per cell (run 1 = cold, runs 2..N = warm). Default 3.")
    args = p.parse_args()

    if args.prompts:
        wanted = {int(x) for x in args.prompts.split(",")}
        selected_prompts = [(pid, t) for pid, t in PROMPTS if pid in wanted]
    else:
        selected_prompts = list(PROMPTS)

    requested_models = set(args.models.split(",")) if args.models else None
    requested_devices = set(args.devices.split(",")) if args.devices else None

    # Build the list of cells (model, device, variant, py, runner) to run.
    cells = []
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
            cells.append({
                "model": model_name, "device": device, "variant": variant,
                "multilingual": multilingual,
                "venv_python": venv_python, "runner": REPO / runner_rel,
            })

    if not cells:
        print("No cells to run. Check --models / --devices and that venvs are installed.")
        return 2

    out_dir = REPO / "results" / datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    print(f"Output: {out_dir}\n")
    print(f"Plan: {len(selected_prompts)} prompts × {len(cells)} cells × {args.runs} runs/cell")
    print()

    rows = []
    fieldnames = ["prompt_id", "model", "device", "variant",
                  "run_index", "is_cold",
                  "ttfa_ms", "gen_s", "audio_s", "rtf",
                  "wall_s", "ok", "error"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for prompt_id, text in selected_prompts:
            print(f"===== Prompt {prompt_id}: {text[:60]}{'...' if len(text) > 60 else ''} =====")
            for cell in cells:
                if prompt_id == 5 and not cell["multilingual"]:
                    continue

                wav = out_dir / f"{cell['model']}_{cell['device']}_p{prompt_id}.wav"
                label = f"  {cell['model']}/{cell['device']:<4}"
                print(label, end=" ", flush=True)

                run_results = run_cell(
                    cell["venv_python"], cell["runner"],
                    text, wav, cell["device"], cell["variant"],
                    args.reference, args.runs,
                )

                for r in run_results:
                    run_index = r.get("run_index", 0)
                    ttfa = r.get("ttfa_ms")
                    gen_s = r.get("gen_s")
                    audio_s = r.get("audio_s")
                    rtf = (audio_s / gen_s) if (audio_s and gen_s) else None

                    row = {
                        "prompt_id": prompt_id,
                        "model": cell["model"],
                        "device": cell["device"],
                        "variant": cell["variant"] or "",
                        "run_index": run_index,
                        "is_cold": run_index == 0,
                        "ttfa_ms": round(ttfa, 1) if ttfa else "",
                        "gen_s": round(gen_s, 4) if gen_s else "",
                        "audio_s": round(audio_s, 3) if audio_s else "",
                        "rtf": round(rtf, 2) if rtf else "",
                        "wall_s": round(r.get("wall_s", 0), 3),
                        "ok": r.get("ok", False),
                        "error": (r.get("error") or "")[:200],
                    }
                    writer.writerow(row)
                    rows.append(row)
                f.flush()

                # Inline summary for this cell: cold + warm-avg
                ok_rows = [r for r in run_results if r.get("ok")]
                if not ok_rows:
                    err = run_results[0].get("error", "?")
                    print(f"FAIL: {err[:80]}")
                    continue
                cold = ok_rows[0]
                warms = ok_rows[1:]
                cold_msg = f"cold ttfa={cold.get('ttfa_ms', 0):.0f}ms rtf={(cold['audio_s']/cold['gen_s']):.1f}x" if cold.get('gen_s') else "cold ok"
                if warms:
                    warm_ttfa = sum(w["ttfa_ms"] for w in warms) / len(warms)
                    warm_rtf = sum(w["audio_s"]/w["gen_s"] for w in warms) / len(warms)
                    warm_msg = f"warm-avg ttfa={warm_ttfa:.0f}ms rtf={warm_rtf:.1f}x"
                    print(f"{cold_msg}  |  {warm_msg}")
                else:
                    print(cold_msg)
            print()

    print(f"Done. CSV: {csv_path}\n")
    _print_summary(rows, selected_prompts)
    return 0


def _print_summary(rows, prompts):
    """Per-prompt comparison table: TTFA(cold) and RTF(warm-avg) per (model, device)."""
    print("=== Per-prompt summary ===\n")
    for prompt_id, text in prompts:
        print(f"Prompt {prompt_id}: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"  {'model':<14} {'device':<6} {'TTFA cold':>10} {'TTFA warm':>10} {'RTF cold':>9} {'RTF warm':>9}")
        cells = {}
        for r in rows:
            if r["prompt_id"] != prompt_id or not r["ok"]:
                continue
            key = (r["model"], r["device"])
            cells.setdefault(key, []).append(r)
        for (model, device), cell_rows in cells.items():
            cold = next((r for r in cell_rows if r["is_cold"]), None)
            warms = [r for r in cell_rows if not r["is_cold"]]
            def fmt_t(r):
                return f"{r['ttfa_ms']:.0f}ms" if r and r["ttfa_ms"] != "" else "—"
            def fmt_r(r):
                return f"{r['rtf']:.1f}x" if r and r["rtf"] != "" else "—"
            warm_ttfa_avg = (
                f"{sum(r['ttfa_ms'] for r in warms)/len(warms):.0f}ms" if warms else "—"
            )
            warm_rtf_avg = (
                f"{sum(r['rtf'] for r in warms)/len(warms):.1f}x" if warms else "—"
            )
            print(f"  {model:<14} {device:<6} {fmt_t(cold):>10} {warm_ttfa_avg:>10} {fmt_r(cold):>9} {warm_rtf_avg:>9}")
        print()


if __name__ == "__main__":
    sys.exit(main())
