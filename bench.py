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
import sys
from datetime import datetime

from harness import REPO, build_cells, run_cell


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
    args = p.parse_args()

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
    print(f"Output: {out_dir}\n")
    print(f"Plan: {len(selected_prompts)} prompts × {len(cells)} cells × {args.runs} runs/cell\n")

    rows = []
    fieldnames = ["prompt_id", "model", "device", "variant", "can_clone",
                  "run_index", "is_cold",
                  "ttfa_ms", "gen_s", "audio_s", "rtf",
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

    print(f"Done. CSV: {csv_path}\n")
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
