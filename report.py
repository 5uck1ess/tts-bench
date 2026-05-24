"""Generate a self-contained HTML report from a bench.py results directory.

Shows TTFA cold/warm + RTF cold/warm per (model, device, prompt), with an
inline <audio> player for the cold run of each cell so you can click-play
each wav in the browser without leaving the page.

Features:
- dark/light theme toggle (persisted to localStorage)
- sortable columns (click any header — cycles asc / desc / unsorted)
- live text filter across all visible rows

All inline — no external CSS, JS, or fonts; works offline.

Also builds results/index.html — a top-level page listing every dated run.

Usage:
    python report.py results/2026-05-23_1409              # one report
    python report.py results/2026-05-23_1409 --open       # ... and open it
    python report.py --index                              # results/index.html only
    python report.py --all                                # regenerate everything
"""

import argparse
import csv
import json
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
  :root {
    color-scheme: dark light;
    --bg: #1a1a1a;
    --panel: #242424;
    --text: #e0e0e0;
    --muted: #888;
    --accent: #6cf;
    --num: #cfc;
    --fail: #f88;
    --border: #333;
    --row-hover: #2a2a2a;
    --th-bg: #2c2c2c;
    --th-hover: #353535;
    --input-bg: #2a2a2a;
    --input-border: #444;
    --controls-border: #2a2a2a;
    --dev-cpu: #ddd;
    --dev-cuda: #9cf;
    --dev-mps: #fc9;
    --lang: #6cf;
    --prompt-text: #9c9;
    --code-bg: #2a2a2a;
  }
  [data-theme="light"] {
    color-scheme: light;
    --bg: #f7f8fa;
    --panel: #ffffff;
    --text: #1a1d22;
    --muted: #6b7280;
    --accent: #1a73e8;
    --num: #137333;
    --fail: #c5221f;
    --border: #e2e6eb;
    --row-hover: #f0f4f8;
    --th-bg: #eef1f4;
    --th-hover: #e2e6eb;
    --input-bg: #ffffff;
    --input-border: #cdd3da;
    --controls-border: #e2e6eb;
    --dev-cpu: #555;
    --dev-cuda: #1a73e8;
    --dev-mps: #c2410c;
    --lang: #1a73e8;
    --prompt-text: #2d6a4f;
    --code-bg: #eef1f4;
  }
  * { box-sizing: border-box; }
  body { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
         background: var(--bg); color: var(--text);
         margin: 0; padding: 1.5rem 2rem 3rem;
         max-width: 1400px; margin-inline: auto;
         transition: background 0.15s, color 0.15s; }
  h1, h2 { color: var(--text); margin-top: 0; }
  h1 { font-size: 1.5em; }
  h2 { font-size: 1.05em; }
  .meta { color: var(--muted); font-size: 0.88em; margin-bottom: 0.4rem; }
  .prompt { background: var(--panel); padding: 1rem 1.2rem;
            border-radius: 10px; margin-bottom: 1.4rem;
            border: 1px solid var(--border); }
  .prompt-text { color: var(--prompt-text); font-style: italic;
                 display: block; margin: 0.2rem 0 0.8rem 0; }
  .lang { color: var(--lang); font-style: normal; margin-right: 0.4rem;
          font-weight: 500; }
  table { border-collapse: collapse; width: 100%; font-size: 0.92em; }
  th, td { padding: 7px 10px; text-align: left;
           border-bottom: 1px solid var(--border); }
  th { background: var(--th-bg); color: var(--text);
       cursor: pointer; user-select: none; white-space: nowrap;
       font-weight: 600; }
  th:hover { background: var(--th-hover); }
  td.num { text-align: right; font-variant-numeric: tabular-nums;
           color: var(--num); }
  td.fail { color: var(--fail); font-style: italic; }
  td.dev-cuda { color: var(--dev-cuda); }
  td.dev-mps  { color: var(--dev-mps); }
  td.dev-cpu  { color: var(--dev-cpu); }
  .muted { color: var(--muted); }
  audio { width: 220px; height: 30px; vertical-align: middle; }
  tr:hover td { background: var(--row-hover); }
  code { background: var(--code-bg); padding: 1px 6px; border-radius: 3px;
         font-size: 0.92em; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .nav { margin-bottom: 0.6rem; }
  .controls { position: sticky; top: 0; background: var(--bg); z-index: 10;
              padding: 0.7rem 0 0.7rem; margin-bottom: 1rem;
              display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap;
              border-bottom: 1px solid var(--controls-border); }
  .controls input { background: var(--input-bg); color: var(--text);
                    border: 1px solid var(--input-border);
                    border-radius: 6px; padding: 7px 11px; font: inherit;
                    min-width: 320px; }
  .controls input:focus { outline: none; border-color: var(--accent); }
  .controls .hint { color: var(--muted); font-size: 0.85em; }
  .controls button { background: var(--input-bg); color: var(--text);
                     border: 1px solid var(--input-border);
                     border-radius: 6px; padding: 6px 12px; font: inherit;
                     cursor: pointer; transition: border-color 0.15s; }
  .controls button:hover { border-color: var(--accent); }
  .spacer { flex: 1; }
</style>"""


CONTROLS = '''<div class="controls">
<input id="filter" type="search" placeholder="filter rows (model, device, value)…" autocomplete="off">
<button type="button" id="reset-sort">reset sort</button>
<span class="hint">click any column header to sort</span>
<span class="spacer"></span>
<button type="button" id="theme-toggle" title="Toggle theme">☾ dark</button>
</div>'''


SCRIPT = r'''<script>
(function(){
  // ---------- theme toggle ----------
  const themeBtn = document.getElementById('theme-toggle');
  function applyTheme(name) {
    document.documentElement.setAttribute('data-theme', name);
    themeBtn.textContent = name === 'light' ? '☀ light' : '☾ dark';
    try { localStorage.setItem('tts-bench-theme', name); } catch (e) {}
  }
  let stored = null;
  try { stored = localStorage.getItem('tts-bench-theme'); } catch (e) {}
  applyTheme(stored === 'light' ? 'light' : 'dark');
  themeBtn.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme') || 'dark';
    applyTheme(cur === 'dark' ? 'light' : 'dark');
  });

  // ---------- sort + filter ----------
  const tables = document.querySelectorAll('table');
  const filterInput = document.getElementById('filter');
  const resetBtn = document.getElementById('reset-sort');
  const sortState = { col: -1, dir: 0 };

  tables.forEach(table => {
    table.querySelectorAll('thead th').forEach(th => { th.dataset.label = th.textContent; });
    table.querySelectorAll('tbody tr').forEach((row, i) => { row.dataset.origIdx = String(i); });
  });

  function applyFilter() {
    const q = filterInput.value.toLowerCase().trim();
    tables.forEach(t => {
      t.querySelectorAll('tbody tr').forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = (!q || text.includes(q)) ? '' : 'none';
      });
    });
  }
  filterInput.addEventListener('input', applyFilter);

  function cellSortValue(row, colIdx) {
    const cell = row.children[colIdx];
    if (!cell) return Number.POSITIVE_INFINITY;
    if (cell.hasAttribute('data-sort')) {
      const raw = cell.getAttribute('data-sort');
      if (raw === '' || raw === null) return Number.POSITIVE_INFINITY;
      const v = parseFloat(raw);
      return Number.isNaN(v) ? Number.POSITIVE_INFINITY : v;
    }
    return cell.textContent.toLowerCase();
  }

  function renderArrows() {
    tables.forEach(t => {
      t.querySelectorAll('thead th').forEach((th, i) => {
        const arrow = (i === sortState.col && sortState.dir !== 0)
                    ? (sortState.dir === 1 ? ' ▲' : ' ▼') : '';
        th.textContent = th.dataset.label + arrow;
      });
    });
  }

  function restoreOrder() {
    tables.forEach(table => {
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((a, b) => parseInt(a.dataset.origIdx) - parseInt(b.dataset.origIdx));
      rows.forEach(r => tbody.appendChild(r));
    });
  }

  function sortAll(colIdx) {
    if (sortState.col === colIdx) {
      sortState.dir = sortState.dir === 1 ? -1 : (sortState.dir === -1 ? 0 : 1);
    } else {
      sortState.col = colIdx;
      sortState.dir = 1;
    }
    renderArrows();

    if (sortState.dir === 0) {
      restoreOrder();
      return;
    }

    const dir = sortState.dir;
    tables.forEach(table => {
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((a, b) => {
        const aSpan = a.querySelector('td[colspan]') !== null;
        const bSpan = b.querySelector('td[colspan]') !== null;
        if (aSpan !== bSpan) return aSpan ? 1 : -1;
        const av = cellSortValue(a, colIdx);
        const bv = cellSortValue(b, colIdx);
        if (av < bv) return -dir;
        if (av > bv) return dir;
        return 0;
      });
      rows.forEach(r => tbody.appendChild(r));
    });
  }

  tables.forEach(table => {
    table.querySelectorAll('thead th').forEach((th, idx) => {
      th.addEventListener('click', () => sortAll(idx));
    });
  });

  resetBtn.addEventListener('click', () => {
    sortState.col = -1; sortState.dir = 0;
    renderArrows();
    restoreOrder();
    filterInput.value = '';
    applyFilter();
  });
})();
</script>'''


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


def _fmt_mb(x):
    """Format MiB compactly: <1000 MB shows as 'NNN MB', >=1024 as 'X.XX GB'."""
    if x is None:
        return "—"
    if x >= 1024:
        return f"{x/1024:.2f} GB"
    return f"{x:.0f} MB"


def _fmt_naq(x):
    """Format NAQ score 0-100; em-dash if missing."""
    if x is None:
        return "—"
    return f"{x:.0f}"


# Display sizes for the Size column. Numbers are the model's weight count
# (params), not weights-on-disk. Hand-curated — keep in sync with README.
MODEL_SIZE = {
    "pocket":      "100M",
    "neutts_air":  "748M",
    "neutts_nano": "748M",
    "luxtts":      "—",
    "chatterbox":  "1.2B",
    "f5tts":       "330M",
    "coqui":       "750M",
    "vibevoice":   "0.5B",
    "omnivoice":   "~1B",
    "voxcpm":      "2B",
    "magpie":      "357M",
    "qwentts":     "1.7B",
    "indextts":    "1.5B",
    "sesame":      "1B",
    "mars5":       "1.2B",
    "kokoro":      "82M",
    "kittentts":   "<100M",
    "piper":       "~25MB",
    "supertonic":  "99M",
}


def _ds(val):
    """data-sort attribute for numeric cells; empty when None."""
    return f' data-sort="{val}"' if val is not None else ' data-sort=""'


def _read_meta(run_dir):
    """Read meta.json from a results dir; return None if missing or invalid."""
    p = run_dir / "meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _rig_summary(meta):
    """One-line human summary like 'AMD Ryzen 9 9950X3D · RTX 5090 32GB · 128 GB RAM · Windows 11'."""
    if not meta:
        return ""
    parts = []
    if meta.get("cpu"):
        cpu = meta["cpu"]
        if meta.get("cpu_cores_physical"):
            cpu = f"{cpu} ({meta['cpu_cores_physical']}C)"
        parts.append(cpu)
    if meta.get("gpu"):
        gpu = meta["gpu"]
        if meta.get("gpu_vram_gb"):
            gpu = f"{gpu} {meta['gpu_vram_gb']}GB"
        parts.append(gpu)
    if meta.get("ram_gb"):
        parts.append(f"{meta['ram_gb']} GB RAM")
    if meta.get("os"):
        parts.append(meta["os"])
    return " · ".join(parts)


def _read_csv(csv_path):
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("ttfa_ms", "gen_s", "audio_s", "rtf",
                      "peak_mem_mb", "peak_vram_mb",
                      "naq", "naq_harm", "naq_buzz",
                      "wall_s"):
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

    meta = _read_meta(run_dir)

    out = ['<!doctype html>',
           '<html lang="en"><head><meta charset="utf-8">',
           f'<title>TTS Bench — {escape(run_dir.name)}</title>',
           STYLE,
           '</head><body>',
           CONTROLS,
           '<div class="nav"><a href="../index.html">← all runs</a></div>',
           f'<h1>TTS Bench — {escape(run_dir.name)}</h1>']
    if meta:
        out.append(f'<div class="meta"><strong>Rig:</strong> '
                   f'<code>{escape(meta.get("rig") or "?")}</code> — '
                   f'{escape(_rig_summary(meta))}</div>')
        if meta.get("label"):
            ref = meta.get("reference")
            ref_html = (f' — ref <code>{escape(ref)}</code>'
                        if ref else "")
            out.append(f'<div class="meta"><strong>Label:</strong> '
                       f'{escape(meta["label"])}{ref_html}</div>')
    out.extend([
        f'<div class="meta">{len(models_seen)} model(s) · '
        f'{len(devices_seen)} device(s) · '
        f'{len(prompts_seen)} prompt(s) · '
        f'{runs_per_cell} run(s) per cell</div>',
        '<div class="meta">Source: <code>results.csv</code></div>',
    ])

    for pid in prompts_seen:
        out.append(f'<div class="prompt"><h2>Prompt {escape(pid)}</h2>')
        ptext = PROMPT_INFO.get(pid)
        if ptext:
            lang, text = ptext
            out.append(f'<span class="prompt-text"><span class="lang">[{escape(lang)}]</span>'
                       f'"{escape(text)}"</span>')

        out.append('<table><thead><tr>')
        for col in ("Model", "Size", "Device", "TTFA cold", "TTFA warm",
                    "RTF cold", "RTF warm", "Mem", "VRAM", "NAQ", "Audio (cold)"):
            out.append(f'<th>{col}</th>')
        out.append('</tr></thead><tbody>')

        cell_keys = sorted([k for k in cells if k[0] == pid], key=lambda k: (k[1], k[2]))
        for (_, model, device) in cell_keys:
            cell_rows = cells[(pid, model, device)]
            cold = next((r for r in cell_rows if r["is_cold"] and r["ok"]), None)
            warms = [r for r in cell_rows if not r["is_cold"] and r["ok"]]
            failed = next((r for r in cell_rows if not r["ok"]), None)

            dev_class = f"dev-{device}"
            size_str = MODEL_SIZE.get(model, "—")
            out.append('<tr>')
            out.append(f'<td>{escape(model)}</td>'
                       f'<td class="muted">{escape(size_str)}</td>'
                       f'<td class="{dev_class}">{escape(device)}</td>')

            if not cold:
                err = (failed.get("error") if failed else "") or "no successful run"
                out.append(f'<td colspan="8" class="fail">FAIL: {escape(err.strip()[:140])}</td>')
                out.append('</tr>')
                continue

            ttfa_cold = cold["ttfa_ms"]
            ttfa_warm = (sum(w["ttfa_ms"] for w in warms) / len(warms)) if warms else None

            def _rtf(r):
                a, g = r["audio_s"], r["gen_s"]
                return (a / g) if (a and g) else None

            rtf_cold = _rtf(cold)
            warm_rtfs = [v for v in (_rtf(w) for w in warms) if v is not None]
            rtf_warm = (sum(warm_rtfs) / len(warm_rtfs)) if warm_rtfs else None

            mem_cold = cold.get("peak_mem_mb")
            vram_cold = cold.get("peak_vram_mb")
            naq_cold = cold.get("naq")
            harm_cold = cold.get("naq_harm")
            buzz_cold = cold.get("naq_buzz")
            naq_tooltip = ""
            if naq_cold is not None:
                naq_tooltip = (
                    f"HARM {_fmt_naq(harm_cold)}  "
                    f"BUZZ {_fmt_naq(buzz_cold)}"
                )

            wav_name = f"{model}_{device}_p{pid}.wav"
            audio_html = (f'<audio controls preload="none" src="{escape(wav_name)}"></audio>'
                          if (run_dir / wav_name).exists()
                          else '<span class="muted">missing</span>')

            out.append(f'<td class="num"{_ds(ttfa_cold)}>{_fmt_ttfa(ttfa_cold)}</td>')
            out.append(f'<td class="num"{_ds(ttfa_warm)}>{_fmt_ttfa(ttfa_warm)}</td>')
            out.append(f'<td class="num"{_ds(rtf_cold)}>{_fmt_rtf(rtf_cold)}</td>')
            out.append(f'<td class="num"{_ds(rtf_warm)}>{_fmt_rtf(rtf_warm)}</td>')
            out.append(f'<td class="num"{_ds(mem_cold)}>{_fmt_mb(mem_cold)}</td>')
            out.append(f'<td class="num"{_ds(vram_cold)}>{_fmt_mb(vram_cold)}</td>')
            out.append(
                f'<td class="num"{_ds(naq_cold)} title="{escape(naq_tooltip)}">'
                f'{_fmt_naq(naq_cold)}</td>'
            )
            out.append(f'<td>{audio_html}</td>')
            out.append('</tr>')

        out.append('</tbody></table></div>')

    out.append(SCRIPT)
    out.append('</body></html>')
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
        meta = _read_meta(d)
        runs.append({
            "name": d.name,
            "models": sorted({r["model"] for r in rows}),
            "devices": sorted({r["device"] for r in rows}),
            "prompts": _sort_prompt_ids({r["prompt_id"] for r in rows}),
            "rows": len(rows),
            "ok": sum(1 for r in rows if r["ok"]),
            "has_html": (d / "report.html").exists(),
            "rig": (meta or {}).get("rig"),
            "rig_full": _rig_summary(meta),
            "label": (meta or {}).get("label"),
        })

    out = ['<!doctype html>',
           '<html lang="en"><head><meta charset="utf-8">',
           '<title>TTS Bench — All Runs</title>',
           STYLE,
           '</head><body>',
           CONTROLS,
           '<h1>TTS Bench — All Runs</h1>',
           f'<div class="meta">{len(runs)} run(s) with <code>results.csv</code></div>',
           '<table><thead><tr>']
    for col in ("Date", "Label", "Rig", "Models", "Devices", "Prompts", "Rows", "OK", "Report"):
        out.append(f'<th>{col}</th>')
    out.append('</tr></thead><tbody>')

    for r in runs:
        models = (", ".join(r["models"])
                  if len(r["models"]) <= 5
                  else f"{len(r['models'])} models")
        link = (f'<a href="{escape(r["name"])}/report.html">view</a>'
                if r["has_html"] else '<span class="muted">no report</span>')
        rig_cell = (f'<code title="{escape(r["rig_full"])}">{escape(r["rig"])}</code>'
                    if r["rig"] else '<span class="muted">—</span>')
        label_cell = (escape(r["label"])
                      if r["label"]
                      else '<span class="muted">—</span>')
        out.append('<tr>')
        out.append(f"<td>{escape(r['name'])}</td>")
        out.append(f"<td>{label_cell}</td>")
        out.append(f"<td>{rig_cell}</td>")
        out.append(f"<td{_ds(len(r['models']))}>{escape(models)}</td>")
        out.append(f"<td>{escape(', '.join(r['devices']))}</td>")
        out.append(f"<td class='num'{_ds(len(r['prompts']))}>{len(r['prompts'])}</td>")
        out.append(f"<td class='num'{_ds(r['rows'])}>{r['rows']}</td>")
        out.append(f"<td class='num'{_ds(r['ok'])}>{r['ok']}/{r['rows']}</td>")
        out.append(f'<td>{link}</td>')
        out.append('</tr>')
    out.append('</tbody></table>')
    out.append(SCRIPT)
    out.append('</body></html>')

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
