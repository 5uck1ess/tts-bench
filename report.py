"""Generate a self-contained HTML report from a bench.py results directory.

The report shows TTFA cold/warm, RTF cold/warm per (model, device, prompt), with
an inline <audio> player for the cold run of each cell so you can click-play
each wav directly in the browser without leaving the page.

Also builds results/index.html — a top-level page listing every dated run.

Usage:
    python report.py results/2026-05-23_1409              # one report
    python report.py results/2026-05-23_1409 --open       # ... and open it
    python report.py --index                              # results/index.html only
    python report.py --all                                # regenerate everything
"""

import argparse
import csv
import sys
import webbrowser
from collections import defaultdict
from html import escape
from pathlib import Path

REPO = Path(__file__).parent
RESULTS = REPO / "results"

try:
    from bench import PROMPTS as _BENCH_PROMPTS
    PROMPT_INFO = {str(pid): (lang, text) for pid, lang, text in _BENCH_PROMPTS}
except Exception:
    PROMPT_INFO = {}


STYLE = """<style>
  :root { color-scheme: dark; }
  body { font-family: ui-monospace, Menlo, Consolas, monospace;
         background: #1a1a1a; color: #e0e0e0; margin: 2rem; max-width: 1400px; }
  h1, h2 { color: #fff; margin-top: 0; }
  h2 { font-size: 1.05em; }
  .meta { color: #888; font-size: 0.88em; margin-bottom: 0.4rem; }
  .prompt { background: #242424; padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 1.4rem; }
  .prompt-text { color: #9c9; font-style: italic; display: block; margin: 0.2rem 0 0.8rem 0; }
  .lang { color: #6cf; font-style: normal; margin-right: 0.4rem; }
  table { border-collapse: collapse; width: 100%; font-size: 0.92em; }
  th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #333; }
  th { background: #2c2c2c; color: #ccc; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; color: #cfc; }
  td.fail { color: #f88; font-style: italic; }
  td.dev-cuda { color: #9cf; }
  td.dev-mps  { color: #fc9; }
  td.dev-cpu  { color: #ddd; }
  .muted { color: #666; }
  audio { width: 220px; height: 28px; vertical-align: middle; }
  tr:hover { background: #2a2a2a; }
  code { background: #2a2a2a; padding: 1px 5px; border-radius: 3px; }
  a { color: #6cf; }
  a:hover { color: #9cf; }
  .nav { margin-bottom: 1rem; }
</style>"""


def _fmt_ttfa(ms):
    if ms is None:
        return "—"
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{ms:.0f}ms"


def _fmt_rtf(x):
    if x is None:
        return "—"
    return f"{x:.2f}×"


def _read_csv(csv_path):
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("ttfa_ms", "gen_s", "audio_s", "rtf", "wall_s"):
                v = r.get(k)
                r[k] = float(v) if v not in (None, "") else None
            r["run_index"] = int(r.get("run_index") or 0)
            r["is_cold"] = str(r.get("is_cold")).lower() == "true"
            r["ok"] = str(r.get("ok")).lower() == "true"
            rows.append(r)
    return rows


def _sort_prompt_ids(ids):
    return sorted(ids, key=lambda s: int(s) if str(s).isdigit() else 1_000_000)


def build_report(run_dir: Path) -> Path:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise SystemExit(f"No results.csv in {run_dir}")
    rows = _read_csv(csv_path)
    if not rows:
        raise SystemExit(f"Empty results.csv in {run_dir}")

    cells = defaultdict(list)
    for r in rows:
        cells[(r["prompt_id"], r["model"], r["device"])].append(r)

    prompts_seen = _sort_prompt_ids({r["prompt_id"] for r in rows})
    models_seen = sorted({r["model"] for r in rows})
    devices_seen = sorted({r["device"] for r in rows})
    runs_per_cell = max(len(v) for v in cells.values())

    out = ["<!doctype html>",
           '<html lang="en"><head><meta charset="utf-8">',
           f"<title>TTS Bench — {escape(run_dir.name)}</title>",
           STYLE,
           "</head><body>"]

    out.append('<div class="nav"><a href="../index.html">← all runs</a></div>')
    out.append(f"<h1>TTS Bench — {escape(run_dir.name)}</h1>")
    out.append(f'<div class="meta">{len(models_seen)} model(s) · '
               f'{len(devices_seen)} device(s) · '
               f'{len(prompts_seen)} prompt(s) · '
               f'{runs_per_cell} run(s) per cell</div>')
    out.append(f'<div class="meta">Source: <code>results.csv</code></div>')

    for pid in prompts_seen:
        out.append(f'<div class="prompt"><h2>Prompt {escape(pid)}</h2>')
        ptext = PROMPT_INFO.get(pid)
        if ptext:
            lang, text = ptext
            out.append(f'<span class="prompt-text"><span class="lang">[{escape(lang)}]</span>'
                       f'"{escape(text)}"</span>')

        out.append("<table><thead><tr>")
        for col in ("Model", "Device", "TTFA cold", "TTFA warm",
                    "RTF cold", "RTF warm", "Audio (cold)"):
            out.append(f"<th>{col}</th>")
        out.append("</tr></thead><tbody>")

        cell_keys = sorted([k for k in cells if k[0] == pid], key=lambda k: (k[1], k[2]))
        for (_, model, device) in cell_keys:
            cell_rows = cells[(pid, model, device)]
            cold = next((r for r in cell_rows if r["is_cold"] and r["ok"]), None)
            warms = [r for r in cell_rows if not r["is_cold"] and r["ok"]]
            failed = next((r for r in cell_rows if not r["ok"]), None)

            dev_class = f"dev-{device}"
            out.append("<tr>")
            out.append(f'<td>{escape(model)}</td><td class="{dev_class}">{escape(device)}</td>')

            if not cold:
                err = (failed.get("error") if failed else "") or "no successful run"
                out.append(f'<td colspan="5" class="fail">FAIL: {escape(err.strip()[:140])}</td>')
                out.append("</tr>")
                continue

            ttfa_cold = cold["ttfa_ms"]
            ttfa_warm = (sum(w["ttfa_ms"] for w in warms) / len(warms)) if warms else None

            def _rtf(r):
                a, g = r["audio_s"], r["gen_s"]
                return (a / g) if (a and g) else None

            rtf_cold = _rtf(cold)
            warm_rtfs = [v for v in (_rtf(w) for w in warms) if v is not None]
            rtf_warm = (sum(warm_rtfs) / len(warm_rtfs)) if warm_rtfs else None

            wav_name = f"{model}_{device}_p{pid}.wav"
            audio_html = (f'<audio controls preload="none" src="{escape(wav_name)}"></audio>'
                          if (run_dir / wav_name).exists()
                          else '<span class="muted">missing</span>')

            out.append(f'<td class="num">{_fmt_ttfa(ttfa_cold)}</td>')
            out.append(f'<td class="num">{_fmt_ttfa(ttfa_warm)}</td>')
            out.append(f'<td class="num">{_fmt_rtf(rtf_cold)}</td>')
            out.append(f'<td class="num">{_fmt_rtf(rtf_warm)}</td>')
            out.append(f"<td>{audio_html}</td>")
            out.append("</tr>")

        out.append("</tbody></table></div>")

    out.append("</body></html>")
    html_path = run_dir / "report.html"
    html_path.write_text("\n".join(out), encoding="utf-8")
    return html_path


def build_index() -> Path:
    if not RESULTS.exists():
        raise SystemExit(f"No results dir at {RESULTS}")

    runs = []
    for d in sorted(RESULTS.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        csv_path = d / "results.csv"
        if not csv_path.exists():
            continue
        try:
            rows = _read_csv(csv_path)
        except Exception:
            continue
        if not rows:
            continue
        runs.append({
            "name": d.name,
            "models": sorted({r["model"] for r in rows}),
            "devices": sorted({r["device"] for r in rows}),
            "prompts": _sort_prompt_ids({r["prompt_id"] for r in rows}),
            "rows": len(rows),
            "ok": sum(1 for r in rows if r["ok"]),
            "has_html": (d / "report.html").exists(),
        })

    out = ["<!doctype html>",
           '<html lang="en"><head><meta charset="utf-8">',
           "<title>TTS Bench — All Runs</title>",
           STYLE,
           "</head><body>",
           "<h1>TTS Bench — All Runs</h1>",
           f'<div class="meta">{len(runs)} run(s) with <code>results.csv</code></div>',
           "<table><thead><tr>"]
    for col in ("Date", "Models", "Devices", "Prompts", "Rows", "OK", "Report"):
        out.append(f"<th>{col}</th>")
    out.append("</tr></thead><tbody>")

    for r in runs:
        models = (", ".join(r["models"])
                  if len(r["models"]) <= 5
                  else f"{len(r['models'])} models")
        link = (f'<a href="{escape(r["name"])}/report.html">view</a>'
                if r["has_html"] else '<span class="muted">no report</span>')
        out.append("<tr>")
        out.append(f"<td>{escape(r['name'])}</td>")
        out.append(f"<td>{escape(models)}</td>")
        out.append(f"<td>{escape(', '.join(r['devices']))}</td>")
        out.append(f"<td class='num'>{len(r['prompts'])}</td>")
        out.append(f"<td class='num'>{r['rows']}</td>")
        out.append(f"<td class='num'>{r['ok']}/{r['rows']}</td>")
        out.append(f"<td>{link}</td>")
        out.append("</tr>")
    out.append("</tbody></table></body></html>")

    index_path = RESULTS / "index.html"
    index_path.write_text("\n".join(out), encoding="utf-8")
    return index_path


def main() -> int:
    p = argparse.ArgumentParser(description="Generate HTML reports from bench.py CSVs.")
    p.add_argument("run_dir", nargs="?", help="Results subdir (e.g. results/2026-05-23_1409).")
    p.add_argument("--index", action="store_true", help="Build only results/index.html.")
    p.add_argument("--all", action="store_true", help="Regenerate report.html for every run + index.")
    p.add_argument("--open", action="store_true", help="Open the generated HTML in a browser.")
    args = p.parse_args()

    if args.all:
        for d in sorted(RESULTS.iterdir(), reverse=True):
            if not d.is_dir() or not (d / "results.csv").exists():
                continue
            html = build_report(d)
            print(f"  wrote {html.relative_to(REPO)}")
        idx = build_index()
        print(f"wrote {idx.relative_to(REPO)}")
        if args.open:
            webbrowser.open(idx.as_uri())
        return 0

    if args.index:
        idx = build_index()
        print(f"wrote {idx.relative_to(REPO)}")
        if args.open:
            webbrowser.open(idx.as_uri())
        return 0

    if not args.run_dir:
        p.error("Provide a run dir, --index, or --all.")
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO / args.run_dir
    if not run_dir.exists():
        raise SystemExit(f"Not found: {run_dir}")

    html = build_report(run_dir)
    print(f"wrote {html.relative_to(REPO)}")
    build_index()
    if args.open:
        webbrowser.open(html.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
