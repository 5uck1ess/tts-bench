"""One-shot A/B listening tool: run a single phrase through every installed
model on every available device, dump one wav per (model, device) cell, and
print a side-by-side latency table.

This is the "play the same line through everything and listen" tool. For
formal benchmarking with cold + warm averages over multiple prompts, use
bench.py. For interactive feel-testing, use speak.py.

Usage:
    python compare.py "hello, this is a quick test of every model"
    python compare.py --file reference/reference.txt
    python compare.py --text "..." --devices cpu               # CPU only
    python compare.py --text "..." --devices cuda              # GPU only
    python compare.py --text "..." --reference reference/chris_hemsworth.wav
    python compare.py --text "..." --models kokoro,piper       # subset
    python compare.py --text "..." --language fr               # non-English (skips English-only models)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from harness import REPO, build_cells, run_cell, play_wav


def main() -> int:
    p = argparse.ArgumentParser(description="One-shot A/B compare across all models × devices.")
    p.add_argument("positional_text", nargs="?", help="Text (positional form of --text).")
    p.add_argument("--text", help="Text to speak.")
    p.add_argument("--file", help="Read text from file (whole contents).")
    p.add_argument("--reference", default=None,
                   help="Reference wav for cloning. Skips predefined-voice-only models.")
    p.add_argument("--models", default=None, help="Comma-sep model names; default: all installed.")
    p.add_argument("--devices", default=None,
                   help="Comma-sep devices; default: cpu + cuda + mps (auto-detected per model).")
    p.add_argument("--language", default="en")
    p.add_argument("--timeout", type=int, default=300, help="Per-cell timeout in seconds.")
    p.add_argument("--no-play", action="store_true",
                   help="Don't play wavs out loud after generation. Default: plays each cell as it finishes.")
    args = p.parse_args()

    if args.positional_text and not args.text and not args.file:
        text = args.positional_text
    elif args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8").strip()
    else:
        print("Provide text via positional arg, --text, or --file.")
        return 2

    if not text:
        print("Empty text.")
        return 2

    requested_models = set(args.models.split(",")) if args.models else None
    requested_devices = set(args.devices.split(",")) if args.devices else None

    cells = build_cells(args.reference, requested_models, requested_devices)
    if not cells:
        print("No cells to run. Check --models / --devices and that venvs are installed.")
        return 2

    # Filter out non-multilingual models when language isn't English.
    if args.language != "en":
        cells = [c for c in cells if c["multilingual"]]
        if not cells:
            print(f"No multilingual models match language={args.language!r}.")
            return 2

    out_dir = REPO / "results" / "compare" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    snippet = text if len(text) <= 70 else text[:70] + "..."
    print(f"Text ({len(text)} chars): {snippet}")
    print(f"Output: {out_dir}")
    print(f"Plan: {len(cells)} cells (model × device)\n")

    results = []
    for cell in cells:
        wav = out_dir / f"{cell['model']}_{cell['device']}.wav"
        label = f"  {cell['model']:<12} / {cell['device']:<4}"
        print(label, end=" ", flush=True)

        run_results = run_cell(cell, text, wav, args.language, runs=1,
                               reference=args.reference, timeout=args.timeout)
        r = run_results[0]
        ok = r.get("ok", False)
        ttfa = r.get("ttfa_ms")
        gen_s = r.get("gen_s")
        audio_s = r.get("audio_s")
        rtf = (audio_s / gen_s) if (audio_s and gen_s) else None
        wall_s = r.get("wall_s")

        results.append({
            "model": cell["model"], "device": cell["device"],
            "ok": ok, "ttfa_ms": ttfa, "gen_s": gen_s,
            "audio_s": audio_s, "rtf": rtf, "wall_s": wall_s,
            "error": r.get("error"), "wav": wav,
        })

        if not ok:
            print(f"FAIL: {(r.get('error') or '')[:80]}")
        else:
            parts = []
            if ttfa is not None: parts.append(f"ttfa={ttfa:.0f}ms")
            if gen_s is not None: parts.append(f"gen={gen_s:.2f}s")
            if audio_s is not None: parts.append(f"audio={audio_s:.2f}s")
            if rtf is not None: parts.append(f"rtf={rtf:.2f}x")
            if wall_s is not None: parts.append(f"wall={wall_s:.2f}s")
            print(" ".join(parts))
            if not args.no_play and wav.exists():
                play_wav(wav)

    print()
    _print_table(results)
    print(f"\nWavs in: {out_dir}")
    return 0


def _print_table(results):
    print("=== Comparison ===")
    print(f"  {'model':<12} {'device':<5} {'ttfa':>9} {'gen':>8} {'audio':>8} {'rtf':>8} {'wall':>8}  status")
    for r in results:
        def fmt(v, unit, fmtstr=".0f"):
            return f"{v:{fmtstr}}{unit}" if v is not None else "—"
        ttfa = fmt(r["ttfa_ms"], "ms")
        gen = fmt(r["gen_s"], "s", ".2f")
        audio = fmt(r["audio_s"], "s", ".2f")
        rtf = fmt(r["rtf"], "x", ".2f")
        wall = fmt(r["wall_s"], "s", ".2f")
        status = "ok" if r["ok"] else (r.get("error") or "fail")[:50]
        print(f"  {r['model']:<12} {r['device']:<5} {ttfa:>9} {gen:>8} {audio:>8} {rtf:>8} {wall:>8}  {status}")


if __name__ == "__main__":
    sys.exit(main())
