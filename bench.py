"""Quick TTS bench: cold + warm timings for all installed models on 5 prompts.

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
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Windows console defaults to cp1252; force UTF-8 so em-dashes / arrows don't
# render as � (or crash on a print).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from harness import REPO, build_cells, run_cell


def _run_cmd(cmd):
    """Run a shell command; return stripped stdout on success, else None.

    Returns None for missing executable (OSError), non-zero exit, or timeout.
    """
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _default_rig_label(sys_name, cpu, gpu):
    os_short = {"Windows": "windows", "Darwin": "mac", "Linux": "linux"}.get(sys_name, "unknown")
    if gpu and "RTX" in gpu:
        return f"{os_short}-{gpu.split()[-1].lower()}"  # "windows-5090"
    if cpu and "Apple" in cpu:
        suffix = cpu.replace("Apple ", "").lower().replace(" ", "-")
        return f"{os_short}-{suffix}"  # "mac-m4" or "mac-m4-pro"
    if cpu:
        words = cpu.lower().split()
        for i, w in enumerate(words):
            if w in ("ryzen", "core", "epyc", "xeon"):
                return f"{os_short}-{'-'.join(words[i:i+3])}"
    return os_short


def detect_rig(label=None):
    """Detect machine info for tagging a bench run. Pure stdlib."""
    sys_name = platform.system()
    meta = {
        "rig": label,
        "os": f"{sys_name} {platform.release()}",
        "os_version": platform.version(),
        "python": platform.python_version(),
        "cpu": None,
        "cpu_cores_logical": os.cpu_count(),
        "cpu_cores_physical": None,
        "ram_gb": None,
        "gpu": None,
        "gpu_vram_gb": None,
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if sys_name == "Windows":
        ps = ["powershell", "-NoProfile", "-Command"]
        meta["cpu"] = _run_cmd(ps + ["(Get-CimInstance Win32_Processor).Name"])
        ram = _run_cmd(ps + ["(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
        if ram and ram.isdigit():
            meta["ram_gb"] = round(int(ram) / (1024**3))
        cores = _run_cmd(ps + ["(Get-CimInstance Win32_Processor).NumberOfCores"])
        if cores and cores.isdigit():
            meta["cpu_cores_physical"] = int(cores)
    elif sys_name == "Darwin":
        meta["cpu"] = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        ram = _run_cmd(["sysctl", "-n", "hw.memsize"])
        if ram and ram.isdigit():
            meta["ram_gb"] = round(int(ram) / (1024**3))
        cores = _run_cmd(["sysctl", "-n", "hw.physicalcpu"])
        if cores and cores.isdigit():
            meta["cpu_cores_physical"] = int(cores)
        if meta["cpu"] and "Apple" in meta["cpu"]:
            meta["gpu"] = f"{meta['cpu']} GPU (MPS)"
    elif sys_name == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        meta["cpu"] = line.split(":", 1)[1].strip()
                        break
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        meta["ram_gb"] = round(int(line.split()[1]) / (1024**2))
                        break
        except OSError:
            pass

    if shutil.which("nvidia-smi"):
        out = _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total",
                        "--format=csv,noheader,nounits"])
        if out:
            first = out.splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if parts:
                meta["gpu"] = parts[0]
                if len(parts) >= 2:
                    try:
                        meta["gpu_vram_gb"] = round(int(parts[1]) / 1024)
                    except ValueError:
                        pass

    if not meta["rig"]:
        meta["rig"] = _default_rig_label(sys_name, meta["cpu"], meta["gpu"])
    return meta


def _auto_label(reference, models_filter, prompts_filter):
    """Build a short human-readable label from the bench's filter args."""
    parts = []
    parts.append("cloning" if reference else "default voice")
    if reference:
        parts.append(Path(reference).stem)
    if models_filter:
        names = sorted(models_filter)
        parts.append(",".join(names) if len(names) <= 3 else f"{len(names)} models")
    if prompts_filter:
        parts.append("prompts " + ",".join(str(x) for x in sorted(prompts_filter)))
    return " · ".join(parts)


def write_meta(run_dir: Path, rig_label=None, label=None,
               reference=None, models_filter=None, prompts_filter=None):
    """Detect rig info and write meta.json into run_dir. Returns the meta dict.

    Includes the bench's filter arguments so the published index can label
    each run without anyone having to click into it.
    """
    meta = detect_rig(label=rig_label)
    meta["label"] = label or _auto_label(reference, models_filter, prompts_filter)
    meta["reference"] = Path(reference).name if reference else None
    meta["models_filter"] = sorted(models_filter) if models_filter else None
    meta["prompts_filter"] = sorted(prompts_filter) if prompts_filter else None
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


PROMPTS = [
    (1, "en", "Open the browser and read my email."),
    (2, "en", "I'll start a new git branch, push the changes, and open a pull request when the tests pass."),
    (3, "en",
     "The Parakeet TDT zero point six billion parameter model achieves "
     "one point six nine percent word error rate on LibriSpeech test-clean, "
     "beating Whisper Large V3 at two point seven percent while running at "
     "over two thousand times realtime on a single GPU."),
    (4, "en", "Run pytest tests slash test underscore voice dot py with verbose flag and capture flag set to no."),
    (5, "fr", "Bonjour, je m'appelle Cicero et je vais vous aider avec votre code aujourd'hui."),
]


def main() -> int:
    p = argparse.ArgumentParser(description="Quick TTS bench (cold + warm).")
    p.add_argument("--reference", default=None,
                   help="Reference wav for voice cloning (omit for each model's default voice).")
    p.add_argument("--prompts", default=None, help="Comma-sep prompt ids; default: all 5.")
    p.add_argument("--models", default=None, help="Comma-sep model names; default: all.")
    p.add_argument("--devices", default=None,
                   help="Comma-sep devices to attempt; default: cpu + cuda + mps (auto-detect).")
    p.add_argument("--runs", type=int, default=3,
                   help="Generations per cell (run 1 = cold, runs 2..N = warm). Default 3.")
    p.add_argument("--rig", default=None,
                   help="Short rig label (e.g. 'windows-5090'). Auto-detected if omitted.")
    p.add_argument("--label", default=None,
                   help="Human-readable label for this run (shown in the published index). "
                        "Auto-derived from --reference and --models if omitted.")
    p.add_argument("--write-meta", metavar="DIR", default=None,
                   help="Just write meta.json into an existing results dir and exit (no bench run).")
    args = p.parse_args()

    if args.write_meta:
        run_dir = Path(args.write_meta)
        if not run_dir.is_absolute():
            run_dir = REPO / args.write_meta
        if not run_dir.exists():
            raise SystemExit(f"Not found: {run_dir}")
        meta = write_meta(run_dir, rig_label=args.rig, label=args.label,
                          reference=args.reference,
                          models_filter=set(args.models.split(",")) if args.models else None,
                          prompts_filter={int(x) for x in args.prompts.split(",")} if args.prompts else None)
        print(f"Wrote {run_dir}/meta.json — rig: {meta['rig']} — label: {meta['label']}")
        return 0

    if args.prompts:
        wanted = {int(x) for x in args.prompts.split(",")}
        selected_prompts = [(pid, lang, t) for pid, lang, t in PROMPTS if pid in wanted]
    else:
        selected_prompts = list(PROMPTS)

    requested_models = set(args.models.split(",")) if args.models else None
    requested_devices = set(args.devices.split(",")) if args.devices else None

    cells = build_cells(args.reference, requested_models, requested_devices)
    if not cells:
        print("No cells to run. Check --models / --devices and that venvs are installed.")
        return 2

    out_dir = REPO / "results" / datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    meta = write_meta(
        out_dir, rig_label=args.rig, label=args.label,
        reference=args.reference, models_filter=requested_models,
        prompts_filter={int(x) for x in args.prompts.split(",")} if args.prompts else None,
    )
    print(f"Output: {out_dir}")
    print(f"Rig: {meta['rig']} ({meta.get('cpu') or '?'} / {meta.get('gpu') or 'no GPU detected'})")
    print(f"Label: {meta['label']}\n")
    print(f"Plan: {len(selected_prompts)} prompts × {len(cells)} cells × {args.runs} runs/cell\n")

    rows = []
    fieldnames = ["prompt_id", "model", "device", "variant", "can_clone",
                  "run_index", "is_cold",
                  "ttfa_ms", "gen_s", "audio_s", "rtf",
                  "peak_mem_mb", "peak_vram_mb",
                  "wall_s", "ok", "error"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for prompt_id, lang, text in selected_prompts:
            print(f"===== Prompt {prompt_id} ({lang}): {text[:60]}{'...' if len(text) > 60 else ''} =====")
            for cell in cells:
                if lang != "en" and not cell["multilingual"]:
                    continue

                wav = out_dir / f"{cell['model']}_{cell['device']}_p{prompt_id}.wav"
                label = f"  {cell['model']}/{cell['device']:<4}"
                print(label, end=" ", flush=True)

                run_results = run_cell(cell, text, wav, lang, args.runs, args.reference)

                for r in run_results:
                    run_index = r.get("run_index", 0)
                    ttfa = r.get("ttfa_ms")
                    gen_s = r.get("gen_s")
                    audio_s = r.get("audio_s")
                    rtf = (audio_s / gen_s) if (audio_s and gen_s) else None
                    peak_mem = r.get("peak_mem_mb")
                    peak_vram = r.get("peak_vram_mb")

                    row = {
                        "prompt_id": prompt_id,
                        "model": cell["model"],
                        "device": cell["device"],
                        "variant": cell["variant"] or "",
                        "can_clone": cell["can_clone"],
                        "run_index": run_index,
                        "is_cold": run_index == 0,
                        "ttfa_ms": round(ttfa, 1) if ttfa else "",
                        "gen_s": round(gen_s, 4) if gen_s else "",
                        "audio_s": round(audio_s, 3) if audio_s else "",
                        "rtf": round(rtf, 2) if rtf else "",
                        "peak_mem_mb": round(peak_mem, 1) if peak_mem is not None else "",
                        "peak_vram_mb": round(peak_vram, 1) if peak_vram is not None else "",
                        "wall_s": round(r.get("wall_s", 0), 3),
                        "ok": r.get("ok", False),
                        "error": (r.get("error") or "")[:200],
                    }
                    writer.writerow(row)
                    rows.append(row)
                f.flush()

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

    print(f"Done. CSV: {csv_path}")
    try:
        from report import build_report, build_index
        html_path = build_report(out_dir)
        build_index()
        print(f"Report: {html_path}\n")
    except Exception as e:
        print(f"(report generation skipped: {e})\n")
    _print_summary(rows, selected_prompts)
    return 0


def _print_summary(rows, prompts):
    """Per-prompt comparison table: TTFA(cold) and RTF(warm-avg) per (model, device)."""
    print("=== Per-prompt summary ===\n")
    for prompt_id, lang, text in prompts:
        print(f"Prompt {prompt_id} ({lang}): {text[:60]}{'...' if len(text) > 60 else ''}")
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
