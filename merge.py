"""Merge new-model rows + wavs from a scratch results dir into a canonical dir.

Non-destructive: appends the rows for the named models from a scratch run (e.g.
results/2026-05-30_2241/) to an existing canonical dir (e.g.
results/windows-default/), copies their cold-run wavs, and regenerates the
report. Use when you've benched new models to a scratch dir and want them folded
into an existing canonical without re-running everything (bench.py --canonical
can only rmtree+recreate, never append — see docs).

The scratch run MUST share the canonical's run parameters (same prompts, same
--runs, same rig, same voice mode / reference) or the merged rows won't be
comparable. merge.py checks the CSV header matches and that none of the named
models already exist in the canonical (pass --force to override).

Usage:
    python merge.py --into results/windows-default \
        --from results/2026-05-30_2241 \
        --models fish_15,vibevoice_7b,maya1,styletts2,zonos,openvoice
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from harness import REPO

# Must match bench.py's writer fieldnames exactly.
FIELDNAMES = ["prompt_id", "model", "device", "variant", "can_clone",
              "run_index", "is_cold", "ttfa_ms", "gen_s", "audio_s", "rtf",
              "peak_mem_mb", "peak_vram_mb", "naq", "naq_artifact",
              "naq_naturalness", "wall_s", "ok", "error"]


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def main() -> int:
    p = argparse.ArgumentParser(description="Merge new-model rows + wavs from a scratch dir into a canonical dir.")
    p.add_argument("--into", required=True, help="Canonical results dir to append into (e.g. results/windows-default).")
    p.add_argument("--from", dest="src", required=True, help="Scratch results dir to take rows/wavs from.")
    p.add_argument("--models", required=True, help="Comma-separated model names to merge in.")
    p.add_argument("--force", action="store_true", help="Append even if a named model already exists in the canonical (skips the duplicate guard).")
    p.add_argument("--replace", action="store_true", help="Replace the named models in --into: drop their existing rows + wavs first, then add the ones from --from. Use when re-benching a model already in the canonical (e.g. a corrected reference).")
    p.add_argument("--no-report", action="store_true", help="Skip report regeneration (just CSV + wavs).")
    args = p.parse_args()

    into = Path(args.into) if Path(args.into).is_absolute() else REPO / args.into
    src = Path(args.src) if Path(args.src).is_absolute() else REPO / args.src
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    into_csv = into / "results.csv"
    src_csv = src / "results.csv"
    for label, pth in (("--into", into_csv), ("--from", src_csv)):
        if not pth.exists():
            print(f"ERROR: {label} has no results.csv: {pth}", file=sys.stderr)
            return 2

    into_fields, into_rows = _read_csv(into_csv)
    src_fields, src_rows = _read_csv(src_csv)

    if into_fields != src_fields:
        print("ERROR: CSV headers differ between --into and --from; refusing to merge.", file=sys.stderr)
        print(f"  into: {into_fields}\n  from: {src_fields}", file=sys.stderr)
        return 2

    existing_models = {r["model"] for r in into_rows}
    clash = sorted(set(models) & existing_models)
    if clash and not args.force and not args.replace:
        print(f"ERROR: these models already exist in {into.name}: {', '.join(clash)}", file=sys.stderr)
        print("  Re-run with --replace to overwrite them, or --force to append anyway (duplicates rows).", file=sys.stderr)
        return 2

    add_rows = [r for r in src_rows if r["model"] in set(models)]
    if not add_rows:
        print(f"ERROR: no rows for {models} found in {src_csv}", file=sys.stderr)
        return 2

    found_models = sorted({r["model"] for r in add_rows})
    missing = sorted(set(models) - set(found_models))
    if missing:
        print(f"WARNING: no rows in --from for: {', '.join(missing)}")

    # Write rows. --replace drops the named models' existing rows first and
    # rewrites the whole CSV; otherwise we just append.
    before = len(into_rows)
    removed_rows = 0
    if args.replace:
        kept = [r for r in into_rows if r["model"] not in set(models)]
        removed_rows = before - len(kept)
        with into_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=into_fields)
            writer.writeheader()
            for r in (kept + add_rows):
                writer.writerow(r)
    else:
        with into_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=into_fields)
            for r in add_rows:
                writer.writerow(r)

    # Copy wavs for the merged models. Filenames are <model>_<device>_p<N>.wav;
    # match precisely so e.g. "vibevoice_7b" never grabs a "vibevoice" wav.
    # Under --replace, delete the model's old wavs in --into first.
    copied = removed_wavs = 0
    for model in found_models:
        pat = re.compile(rf"^{re.escape(model)}_(cpu|cuda|mps)_p\d+\.wav$")
        if args.replace:
            for wav in into.glob(f"{model}_*.wav"):
                if pat.match(wav.name):
                    wav.unlink()
                    removed_wavs += 1
        for wav in src.glob(f"{model}_*.wav"):
            if pat.match(wav.name):
                shutil.copy2(wav, into / wav.name)
                copied += 1

    new_total = (before - removed_rows) + len(add_rows)
    print(f"{'Replaced' if args.replace else 'Merged'} {len(add_rows)} rows "
          f"({', '.join(found_models)}) into {into.name}/results.csv")
    print(f"  rows: {before} -> {new_total}" + (f" (dropped {removed_rows} old)" if args.replace else ""))
    print(f"  wavs copied: {copied}" + (f" (removed {removed_wavs} old)" if args.replace else ""))

    if not args.no_report:
        try:
            from report import build_report, build_index
            html = build_report(into)
            build_index()
            print(f"  report regenerated: {html}")
        except Exception as e:
            print(f"  WARNING: report regeneration failed: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
