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
  th.num { text-align: right; }
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

  /* Lens picker grid */
  .lens-grid { display: grid; grid-template-columns: repeat(3, 1fr);
               gap: 1rem; margin: 1.5rem 0; }
  @media (max-width: 760px) { .lens-grid { grid-template-columns: 1fr; } }
  .lens-card { background: var(--panel); border: 1px solid var(--border);
               border-radius: 10px; padding: 1.2rem 1.4rem;
               transition: border-color 0.15s; }
  .lens-card:hover { border-color: var(--accent); }
  .lens-card a { display: block; text-decoration: none; color: var(--text); }
  .lens-card h3 { margin: 0 0 0.4rem 0; font-size: 1.1em; color: var(--accent); }
  .lens-card .desc { color: var(--muted); font-size: 0.9em; }

  /* TLDR callout used by speed.html + quality.html */
  .tldr { background: var(--panel); border-left: 3px solid var(--accent);
          padding: 0.8rem 1.1rem; margin: 1rem 0 1.4rem; border-radius: 4px; }
  .tldr h2 { margin: 0 0 0.4rem 0; font-size: 1em; }
  .tldr p { margin: 0.2rem 0; color: var(--text); }
  .tldr strong { color: var(--accent); }

  /* Secondary-axis "badge" pill cells */
  td.pill { font-size: 0.82em; color: var(--muted); }
  td.pill[data-sort] { font-variant-numeric: tabular-nums; }
  td.pill a { color: inherit; }

  /* Prompt jumper for samples.html */
  .prompt-jumper { position: sticky; top: 3rem; background: var(--bg);
                   padding: 0.4rem 0; margin-bottom: 0.8rem;
                   border-bottom: 1px solid var(--border); font-size: 0.9em; }
  .prompt-jumper a { margin-right: 0.6rem; }

  /* Lens-to-lens nav tabs at top of each lens page */
  .lens-tabs { display: inline-flex; gap: 0.3rem; margin-right: 0.6rem; }
  .lens-tab { display: inline-block; padding: 4px 12px; border-radius: 6px;
              border: 1px solid var(--border); color: var(--text);
              text-decoration: none; font-size: 0.92em;
              transition: border-color 0.15s, background 0.15s; }
  .lens-tab:hover { border-color: var(--accent); text-decoration: none; }
  .lens-tab.active { background: var(--accent); color: var(--bg);
                     border-color: var(--accent); }

  /* "Reading this report" explainer above tables */
  .reading-guide { background: var(--panel); border-left: 3px solid var(--muted);
                   padding: 0.55rem 0.9rem; margin: 0.7rem 0 1rem;
                   border-radius: 4px; font-size: 0.88em; color: var(--text); }
  .reading-guide strong { color: var(--accent); }
  .reading-guide a { color: var(--accent); }
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

  function sortAll(colIdx, explicitDir) {
    if (explicitDir === 1 || explicitDir === -1) {
      sortState.col = colIdx;
      sortState.dir = explicitDir;
    } else if (sortState.col === colIdx) {
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

  // Page-load default sort: pages can set window.__defaultSort = {colIdx, dir}
  // (set in a tiny inline <script> earlier in the page).
  if (window.__defaultSort &&
      typeof window.__defaultSort.colIdx === 'number' &&
      (window.__defaultSort.dir === 1 || window.__defaultSort.dir === -1)) {
    sortAll(window.__defaultSort.colIdx, window.__defaultSort.dir);
  }
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
# Display name shown in the rendered HTML for each venv-keyed model.
# CSV column still uses the raw venv key; this is presentation-only.
# Keys not present here fall through to the raw key (Title-cased by convention).
MODEL_DISPLAY_NAMES = {
    "sesame":        "Sesame CSM-1B",
    "coqui":         "Coqui XTTS-v2",
    "vibevoice":     "VibeVoice Realtime 0.5B",
    "vibevoice_15b": "VibeVoice 1.5B",
    "qwentts":       "Qwen3-TTS 1.7B",
    "qwentts_fast":  "Qwen3-TTS 1.7B (CUDA-graph)",
    "neutts_air":    "NeuTTS Air",
    "neutts_nano":   "NeuTTS Nano",
    "voxcpm":        "VoxCPM2 2B",
    "mars5":         "Mars5-TTS",
    "pocket":        "Pocket-TTS",
    "magpie":        "Magpie-TTS",
    "indextts":      "IndexTTS-2",
    "f5tts":         "F5-TTS",
    "chatterbox":       "Chatterbox",
    "chatterbox_turbo": "Chatterbox Turbo",
    "dia":           "Dia 1.6B",
    "omnivoice":     "OmniVoice",
    "zipvoice":      "ZipVoice 123M",
    "piper":         "Piper",
    "kokoro":        "Kokoro",
    "kittentts":     "KittenTTS",
    "supertonic":    "Supertonic",
    "luxtts":        "LuxTTS",
}


def _display_name(model):
    """Return display name for a model venv key; falls back to raw key."""
    return MODEL_DISPLAY_NAMES.get(model, model)


MODEL_SIZE = {
    "pocket":        "100M",
    "neutts_air":    "748M",
    "neutts_nano":   "748M",
    "luxtts":        "—",
    "chatterbox":       "1.2B",
    "chatterbox_turbo": "744M",
    "f5tts":         "330M",
    "coqui":         "750M",
    "vibevoice":     "0.5B",
    "vibevoice_15b": "3B",
    "omnivoice":     "~1B",
    "zipvoice":      "123M",
    "voxcpm":        "2B",
    "magpie":        "357M",
    "qwentts":       "1.7B",
    "qwentts_fast":  "1.7B",
    "indextts":      "1.5B",
    "sesame":        "1B",
    "mars5":         "1.2B",
    "dia":           "1.6B",
    "kokoro":        "82M",
    "kittentts":     "<100M",
    "piper":         "~25MB",
    "supertonic":    "99M",
}

# Whether a model supports zero-shot voice cloning at runtime.
# "predefined": fixed speaker(s) / preset voices only.
# "cloning":    accepts a reference wav (or wav+text) at inference and matches that voice.
MODEL_KIND = {
    "piper":         "predefined",
    "kokoro":        "predefined",
    "kittentts":     "predefined",
    "magpie":        "predefined",
    "vibevoice":     "predefined",
    "vibevoice_15b": "predefined",
    "supertonic":    "predefined",
    "luxtts":        "predefined",
    "pocket":        "cloning",
    "chatterbox":       "cloning",
    "chatterbox_turbo": "cloning",
    "f5tts":         "cloning",
    "indextts":      "cloning",
    "omnivoice":     "cloning",
    "zipvoice":      "cloning",
    "voxcpm":        "cloning",
    "coqui":         "cloning",
    "qwentts":       "cloning",
    "qwentts_fast":  "cloning",
    "sesame":        "cloning",
    "mars5":         "cloning",
    "dia":           "cloning",
    "neutts_air":    "cloning",
    "neutts_nano":   "cloning",
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
                      "naq", "naq_artifact", "naq_naturalness",
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


def _build_context(rows, run_dir, meta):
    """Pre-compute everything the four lens renderers need from a CSV row list.

    Returns a dict with:
      per_cell:    {(prompt_id, model, device) -> {"cold": row|None, "warms": [row], "fail": row|None}}
                   Note: only the FIRST failure row per cell is retained in "fail".
                   Subsequent failures for the same cell are dropped.
      per_model:   {(model, device) -> aggregated dict with avg TTFA/RTF, peak mem/vram, avg NAQ,
                    n_ok, n_fail, n_total}
      per_prompt:  {prompt_id -> [{model, device, naq, ttfa_warm, rtf_warm, wav, wav_exists,
                    naq_artifact, naq_naturalness} ranked by NAQ desc, successful-cold rows only]}
      tldr_speed:  {"predefined": (model, device, rtf_warm, ttfa_warm) | None,
                    "cloning":    (model, device, rtf_warm, ttfa_warm) | None}
      tldr_quality:{"predefined": [(model, device, naq) top 3],
                    "cloning":    [(model, device, naq) top 3]}
      prompts_seen, models_seen, devices_seen, runs_per_cell, meta, run_name, run_dir
    """
    cells = defaultdict(lambda: {"cold": None, "warms": [], "fail": None})
    for r in rows:
        key = (r["prompt_id"], r["model"], r["device"])
        if not r["ok"]:
            if cells[key]["fail"] is None:
                cells[key]["fail"] = r
        elif r["is_cold"]:
            cells[key]["cold"] = r
        else:
            cells[key]["warms"].append(r)

    prompts_seen = _sort_prompt_ids({r["prompt_id"] for r in rows})
    models_seen = sorted({r["model"] for r in rows})
    devices_seen = sorted({r["device"] for r in rows})
    # Intentionally caps fail at 1 per cell (matches the first-failure-wins above)
    # so the "N runs per cell" badge reflects warm-run count, not failure noise.
    runs_per_cell = max(
        (1 if c["cold"] else 0) + len(c["warms"]) + (1 if c["fail"] else 0)
        for c in cells.values()
    )

    def _rtf(r):
        a, g = r.get("audio_s"), r.get("gen_s")
        return (a / g) if (a and g) else None

    per_model = {}
    for (pid, model, dev), c in cells.items():
        key = (model, dev)
        per_model.setdefault(key, {
            "ttfa_cold": [], "ttfa_warm": [], "rtf_cold": [], "rtf_warm": [],
            "peak_mem": [], "peak_vram": [], "naq": [],
            "ok_prompts": set(), "fail_prompts": set(), "all_prompts": set(),
        })
        bucket = per_model[key]
        bucket["all_prompts"].add(pid)
        if c["cold"]:
            bucket["ok_prompts"].add(pid)
            bucket["ttfa_cold"].append(c["cold"]["ttfa_ms"])
            rc = _rtf(c["cold"])
            if rc is not None:
                bucket["rtf_cold"].append(rc)
            if c["cold"].get("peak_mem_mb") is not None:
                bucket["peak_mem"].append(c["cold"]["peak_mem_mb"])
            if c["cold"].get("peak_vram_mb") is not None:
                bucket["peak_vram"].append(c["cold"]["peak_vram_mb"])
            if c["cold"].get("naq") is not None:
                bucket["naq"].append(c["cold"]["naq"])
            if c["warms"]:
                bucket["ttfa_warm"].extend(w["ttfa_ms"] for w in c["warms"])
                warm_rtfs = [v for v in (_rtf(w) for w in c["warms"]) if v is not None]
                bucket["rtf_warm"].extend(warm_rtfs)
        elif c["fail"]:
            bucket["fail_prompts"].add(pid)

    def _avg(xs):
        return (sum(xs) / len(xs)) if xs else None

    def _maxv(xs):
        return max(xs) if xs else None

    agg = {}
    for key, b in per_model.items():
        agg[key] = {
            "ttfa_cold": _avg(b["ttfa_cold"]),
            "ttfa_warm": _avg(b["ttfa_warm"]),
            "rtf_cold":  _avg(b["rtf_cold"]),
            "rtf_warm":  _avg(b["rtf_warm"]),
            "peak_mem":  _maxv(b["peak_mem"]),
            "peak_vram": _maxv(b["peak_vram"]),
            "naq":       _avg(b["naq"]),
            "n_ok":      len(b["ok_prompts"]),
            "n_fail":    len(b["fail_prompts"]),
            "n_total":   len(b["all_prompts"]),
        }

    per_prompt = {}
    for pid in prompts_seen:
        items = []
        for (p2, model, dev), c in cells.items():
            if p2 != pid:
                continue
            if not c["cold"]:
                continue
            cold = c["cold"]
            warm_ttfas = [w["ttfa_ms"] for w in c["warms"]]
            ttfa_warm = (sum(warm_ttfas) / len(warm_ttfas)) if warm_ttfas else None
            warm_rtfs = [v for v in (_rtf(w) for w in c["warms"]) if v is not None]
            rtf_warm = (sum(warm_rtfs) / len(warm_rtfs)) if warm_rtfs else None
            items.append({
                "model": model,
                "device": dev,
                "naq": cold.get("naq"),
                "ttfa_warm": ttfa_warm,
                "rtf_warm": rtf_warm,
                "wav": f"{model}_{dev}_p{pid}.wav",
                "wav_exists": (run_dir / f"{model}_{dev}_p{pid}.wav").exists(),
                "naq_artifact": cold.get("naq_artifact"),
                "naq_naturalness": cold.get("naq_naturalness"),
            })
        # Sort by NAQ desc; rows with no NAQ sink to the bottom
        items.sort(key=lambda it: (it["naq"] is None, -(it["naq"] or 0)))
        per_prompt[pid] = items

    def _pick_best_speed(kind):
        best = None
        for (model, dev), a in agg.items():
            if MODEL_KIND.get(model) != kind:
                continue
            if a["rtf_warm"] is None:
                continue
            if best is None or a["rtf_warm"] > best[2]:
                best = (model, dev, a["rtf_warm"], a["ttfa_warm"])
        return best

    def _pick_top_quality(kind, n=3):
        scored = []
        for (model, dev), a in agg.items():
            if MODEL_KIND.get(model) != kind:
                continue
            if a["naq"] is None:
                continue
            scored.append((model, dev, a["naq"]))
        scored.sort(key=lambda t: -t[2])
        return scored[:n]

    return {
        "per_cell": cells,
        "per_model": agg,
        "per_prompt": per_prompt,
        "tldr_speed": {
            "predefined": _pick_best_speed("predefined"),
            "cloning":    _pick_best_speed("cloning"),
        },
        "tldr_quality": {
            "predefined": _pick_top_quality("predefined"),
            "cloning":    _pick_top_quality("cloning"),
        },
        "prompts_seen": prompts_seen,
        "models_seen": models_seen,
        "devices_seen": devices_seen,
        "runs_per_cell": runs_per_cell,
        "meta": meta,
        "run_name": run_dir.name,
        "run_dir": run_dir,
    }


def _render_lens_picker(ctx):
    """Render the per-rig landing card with three lens links."""
    meta = ctx["meta"]
    out = ['<!doctype html>',
           '<html lang="en"><head><meta charset="utf-8">',
           f'<title>TTS Bench — {escape(ctx["run_name"])}</title>',
           STYLE,
           '</head><body>',
           CONTROLS,
           '<div class="nav"><a href="../index.html">← all runs</a></div>',
           f'<h1>TTS Bench — {escape(ctx["run_name"])}</h1>']
    if meta:
        out.append(f'<div class="meta"><strong>Rig:</strong> '
                   f'<code>{escape(meta.get("rig") or "?")}</code> — '
                   f'{escape(_rig_summary(meta))}</div>')
        if meta.get("label"):
            ref = meta.get("reference")
            ref_html = f' — ref <code>{escape(ref)}</code>' if ref else ""
            out.append(f'<div class="meta"><strong>Label:</strong> '
                       f'{escape(meta["label"])}{ref_html}</div>')
    n_models = len(ctx["models_seen"])
    n_devices = len(ctx["devices_seen"])
    n_prompts = len(ctx["prompts_seen"])
    out.append(
        f'<div class="meta">{n_models} model(s) · '
        f'{n_devices} device(s) · '
        f'{n_prompts} prompt(s) · '
        f'{ctx["runs_per_cell"]} run(s) per cell</div>'
    )
    out.append('<div class="lens-grid">')
    out.append(
        '<div class="lens-card"><a href="speed.html">'
        '<h3>▶ Speed</h3>'
        f'<div class="desc">TTFA, RTF, memory · {n_models} models · sortable</div>'
        '</a></div>'
    )
    out.append(
        '<div class="lens-card"><a href="quality.html">'
        '<h3>▶ Quality</h3>'
        f'<div class="desc">NAQ + sub-scores · audio embedded · {n_models} models</div>'
        '</a></div>'
    )
    out.append(
        '<div class="lens-card"><a href="samples.html">'
        '<h3>▶ Samples</h3>'
        '<div class="desc">By-prompt gallery · A/B every model</div>'
        '</a></div>'
    )
    out.append('</div>')
    out.append('<div class="meta">'
               '<a href="results.csv">results.csv</a> · '
               '<a href="meta.json">meta.json</a></div>')
    out.append(SCRIPT)
    out.append('</body></html>')
    return "\n".join(out)


_LENSES = (("speed", "Speed"), ("quality", "Quality"), ("samples", "Samples"))

_READING_GUIDE = {
    "speed": (
        '<div class="reading-guide">'
        '<strong>TTFA</strong> = time to first audio (ms; lower is better). '
        '<strong>RTF</strong> = real-time factor (× realtime; higher is better; e.g. 10× means '
        '10 sec of audio generated per 1 sec of compute). '
        '<strong>Cold</strong> = first run after process start; <strong>warm</strong> = subsequent runs. '
        '<strong>NAQ</strong> = quality score 0-100 (click NAQ pill to jump to quality lens).'
        '</div>'
    ),
    "quality": (
        '<div class="reading-guide">'
        '<strong>NAQ</strong> (Naturalness-Artifact Quotient): 0-100 objective quality score; '
        'higher = more natural. '
        'Composed of <strong>ARTIFACT</strong> macro (artifact absence: HARM + BUZZ) and '
        '<strong>NATURALNESS</strong> macro (positive prosody: DYN + PROSODY + RHYTHM + PITCH_MVMT) '
        'at 50/50. Hover any NAQ cell for the two-macro breakdown. '
        '<a href="https://github.com/5uck1ess/tts-bench/blob/main/docs/naq.md">Full spec →</a>'
        '</div>'
    ),
    "samples": (
        '<div class="reading-guide">'
        'Each prompt section shows every model\'s audio output, ranked by '
        '<strong>NAQ</strong> (0-100 quality score; higher = better). '
        'Click any audio player to hear that model\'s rendering.'
        '</div>'
    ),
}


def _lens_nav(active):
    """Emit the nav strip with lens tabs (Speed / Quality / Samples) + 'all runs' link.

    `active` is one of "speed", "quality", "samples". The matching tab gets
    .lens-tab.active styling so the user can see which page they're on.
    """
    parts = ['<div class="nav"><span class="lens-tabs">']
    for slug, label in _LENSES:
        cls = "lens-tab active" if slug == active else "lens-tab"
        parts.append(f'<a class="{cls}" href="{slug}.html">{label}</a>')
    parts.append('</span> · <a href="../index.html">all runs</a></div>')
    return "".join(parts)


def _render_speed(ctx):
    """Render speed.html — aggregated per (model, device), no audio."""
    meta = ctx["meta"]
    has_ref = bool((meta or {}).get("reference"))
    out = ['<!doctype html>',
           '<html lang="en"><head><meta charset="utf-8">',
           f'<title>TTS Bench — Speed — {escape(ctx["run_name"])}</title>',
           STYLE,
           '</head><body>',
           CONTROLS,
           _lens_nav("speed"),
           f'<h1>TTS Bench — Speed — {escape(ctx["run_name"])}</h1>']
    if meta:
        out.append(f'<div class="meta"><strong>Rig:</strong> '
                   f'<code>{escape(meta.get("rig") or "?")}</code> — '
                   f'{escape(_rig_summary(meta))}</div>')
        if meta.get("label"):
            ref = meta.get("reference")
            ref_html = f' — ref <code>{escape(ref)}</code>' if ref else ""
            out.append(f'<div class="meta"><strong>Label:</strong> '
                       f'{escape(meta["label"])}{ref_html}</div>')

    out.append(_READING_GUIDE["speed"])

    # TLDR
    out.append('<div class="tldr"><h2>Speed winners</h2>')
    tldr = ctx["tldr_speed"]
    def _fmt_tldr(entry, label):
        if not entry:
            return f'<p>{label}: <span class="muted">no data</span></p>'
        model, dev, rtf, ttfa = entry
        return (f'<p>{label}: <strong>{escape(_display_name(model))}</strong> ({escape(dev)}) — '
                f'{_fmt_rtf(rtf)} warm RTF, {_fmt_ttfa(ttfa)} warm TTFA</p>')
    if has_ref:
        # Cloning-only run; collapse to single line
        best = tldr["cloning"] or tldr["predefined"]
        out.append(_fmt_tldr(best, "Fastest (this cloning run)"))
    else:
        out.append(_fmt_tldr(tldr["predefined"], "Fastest predefined-voice"))
        out.append(_fmt_tldr(tldr["cloning"],    "Fastest cloning-capable"))
    out.append('</div>')

    # Table
    cols = ("Model", "Device", "TTFA cold", "TTFA warm",
            "RTF cold", "RTF warm", "Peak RAM", "Peak VRAM", "Size", "NAQ")
    num_cols = {"TTFA cold", "TTFA warm", "RTF cold", "RTF warm",
                "Peak RAM", "Peak VRAM", "NAQ"}
    rtf_warm_idx = cols.index("RTF warm")
    out.append('<table><thead><tr>')
    for c in cols:
        cls = ' class="num"' if c in num_cols else ''
        out.append(f'<th{cls}>{c}</th>')
    out.append('</tr></thead><tbody>')

    # Sort rows by RTF warm desc for stable origIdx ordering
    keys_sorted = sorted(
        ctx["per_model"].keys(),
        key=lambda k: (ctx["per_model"][k]["rtf_warm"] is None,
                       -(ctx["per_model"][k]["rtf_warm"] or 0))
    )
    for (model, dev) in keys_sorted:
        a = ctx["per_model"][(model, dev)]
        row_id = f"speed-{model}-{dev}".lower().replace("/", "-")
        dev_class = f"dev-{dev}"
        size_str = MODEL_SIZE.get(model, "—")
        n_ok = a["n_ok"]; n_total = a["n_total"]
        partial = (n_ok < n_total) and n_ok > 0
        partial_tag = (f' <span class="muted">({n_ok}/{n_total} ok)</span>'
                       if partial else "")
        if n_ok == 0:
            err_row = next(
                (c["fail"] for k, c in ctx["per_cell"].items()
                 if k[1] == model and k[2] == dev and c["fail"]),
                None,
            )
            err = (err_row.get("error") if err_row else "") or "no successful run"
            out.append(f'<tr id="{escape(row_id)}">'
                       f'<td>{escape(_display_name(model))}</td>'
                       f'<td class="{dev_class}">{escape(dev)}</td>'
                       f'<td colspan="{len(cols)-2}" class="fail">FAIL: {escape(err.strip()[:140])}</td>'
                       '</tr>')
            continue
        out.append(f'<tr id="{escape(row_id)}">')
        out.append(f'<td>{escape(_display_name(model))}{partial_tag}</td>'
                   f'<td class="{dev_class}">{escape(dev)}</td>')
        out.append(f'<td class="num"{_ds(a["ttfa_cold"])}>{_fmt_ttfa(a["ttfa_cold"])}</td>')
        out.append(f'<td class="num"{_ds(a["ttfa_warm"])}>{_fmt_ttfa(a["ttfa_warm"])}</td>')
        out.append(f'<td class="num"{_ds(a["rtf_cold"])}>{_fmt_rtf(a["rtf_cold"])}</td>')
        out.append(f'<td class="num"{_ds(a["rtf_warm"])}>{_fmt_rtf(a["rtf_warm"])}</td>')
        out.append(f'<td class="num"{_ds(a["peak_mem"])}>{_fmt_mb(a["peak_mem"])}</td>')
        out.append(f'<td class="num"{_ds(a["peak_vram"])}>{_fmt_mb(a["peak_vram"])}</td>')
        out.append(f'<td class="muted">{escape(size_str)}</td>')
        naq_val = a["naq"]
        out.append(
            f'<td class="num pill"{_ds(naq_val)}>'
            f'<a href="quality.html" title="See NAQ details on quality view">{_fmt_naq(naq_val)}</a>'
            '</td>'
        )
        out.append('</tr>')
    out.append('</tbody></table>')

    # Default sort: RTF warm desc
    out.append(f'<script>window.__defaultSort = {{colIdx: {rtf_warm_idx}, dir: -1}};</script>')
    out.append(SCRIPT)
    out.append('</body></html>')
    return "\n".join(out)


def _render_quality(ctx):
    """Render quality.html — per-prompt grouping, NAQ-first columns, audio embedded."""
    meta = ctx["meta"]
    has_ref = bool((meta or {}).get("reference"))
    out = ['<!doctype html>',
           '<html lang="en"><head><meta charset="utf-8">',
           f'<title>TTS Bench — Quality — {escape(ctx["run_name"])}</title>',
           STYLE,
           '</head><body>',
           CONTROLS,
           _lens_nav("quality"),
           f'<h1>TTS Bench — Quality — {escape(ctx["run_name"])}</h1>']
    if meta:
        out.append(f'<div class="meta"><strong>Rig:</strong> '
                   f'<code>{escape(meta.get("rig") or "?")}</code> — '
                   f'{escape(_rig_summary(meta))}</div>')
        if meta.get("label"):
            ref = meta.get("reference")
            ref_html = f' — ref <code>{escape(ref)}</code>' if ref else ""
            out.append(f'<div class="meta"><strong>Label:</strong> '
                       f'{escape(meta["label"])}{ref_html}</div>')

    out.append(_READING_GUIDE["quality"])

    # TLDR
    out.append('<div class="tldr"><h2>Quality winners (NAQ)</h2>')
    def _fmt_top(entries, label):
        if not entries:
            return f'<p>{label}: <span class="muted">no data</span></p>'
        parts = [f'<strong>{escape(_display_name(m))}</strong> ({_fmt_naq(n)})'
                 for (m, _d, n) in entries]
        return f'<p>{label}: {" · ".join(parts)}</p>'
    if has_ref:
        out.append(_fmt_top(ctx["tldr_quality"]["cloning"], "Top 3 (this cloning run)"))
    else:
        out.append(_fmt_top(ctx["tldr_quality"]["predefined"], "Top 3 predefined-voice"))
        out.append(_fmt_top(ctx["tldr_quality"]["cloning"],    "Top 3 cloning-capable"))
    out.append('</div>')

    # Per-prompt tables
    cols = ("Model", "Device", "NAQ", "Size",
            "TTFA warm", "RTF warm", "Audio (cold)")
    num_cols = {"NAQ", "TTFA warm", "RTF warm"}
    naq_idx = cols.index("NAQ")
    for pid in ctx["prompts_seen"]:
        out.append(f'<div class="prompt"><h2>Prompt {escape(pid)}</h2>')
        ptext = PROMPT_INFO.get(pid)
        if ptext:
            lang, text = ptext
            out.append(f'<span class="prompt-text"><span class="lang">[{escape(lang)}]</span>'
                       f'"{escape(text)}"</span>')
        out.append('<table><thead><tr>')
        for c in cols:
            cls = ' class="num"' if c in num_cols else ''
            out.append(f'<th{cls}>{c}</th>')
        out.append('</tr></thead><tbody>')

        cell_keys = sorted([k for k in ctx["per_cell"] if k[0] == pid],
                           key=lambda k: (k[1], k[2]))
        for (_, model, dev) in cell_keys:
            c = ctx["per_cell"][(pid, model, dev)]
            row_id = f"quality-{model}-{dev}-p{pid}".lower().replace("/", "-")
            dev_class = f"dev-{dev}"
            size_str = MODEL_SIZE.get(model, "—")
            if not c["cold"]:
                err = (c["fail"].get("error") if c["fail"] else "") or "no successful run"
                out.append(
                    f'<tr id="{escape(row_id)}">'
                    f'<td>{escape(_display_name(model))}</td>'
                    f'<td class="{dev_class}">{escape(dev)}</td>'
                    f'<td colspan="{len(cols)-2}" class="fail">FAIL: {escape(err.strip()[:140])}</td>'
                    '</tr>'
                )
                continue
            cold = c["cold"]
            warm_ttfas = [w["ttfa_ms"] for w in c["warms"]]
            ttfa_warm = (sum(warm_ttfas) / len(warm_ttfas)) if warm_ttfas else None
            def _rtf(r):
                a, g = r["audio_s"], r["gen_s"]
                return (a / g) if (a and g) else None
            warm_rtfs = [v for v in (_rtf(w) for w in c["warms"]) if v is not None]
            rtf_warm = (sum(warm_rtfs) / len(warm_rtfs)) if warm_rtfs else None

            wav_name = f"{model}_{dev}_p{pid}.wav"
            audio_html = (f'<audio controls preload="none" src="{escape(wav_name)}"></audio>'
                          if (ctx["run_dir"] / wav_name).exists()
                          else '<span class="muted">missing</span>')

            out.append(f'<tr id="{escape(row_id)}">')
            out.append(f'<td>{escape(_display_name(model))}</td>'
                       f'<td class="{dev_class}">{escape(dev)}</td>')
            naq_val = cold.get("naq")
            naq_art = cold.get("naq_artifact")
            naq_nat = cold.get("naq_naturalness")
            tooltip = (f"artifact: {naq_art}, naturalness: {naq_nat}"
                       if (naq_art is not None and naq_nat is not None) else "")
            out.append(
                f'<td class="num"{_ds(naq_val)} title="{escape(tooltip)}">{_fmt_naq(naq_val)}</td>'
            )
            out.append(f'<td class="muted">{escape(size_str)}</td>')
            out.append(f'<td class="num pill"{_ds(ttfa_warm)}>{_fmt_ttfa(ttfa_warm)}</td>')
            out.append(f'<td class="num pill"{_ds(rtf_warm)}>{_fmt_rtf(rtf_warm)}</td>')
            out.append(f'<td>{audio_html}</td>')
            out.append('</tr>')
        out.append('</tbody></table></div>')

    # Default sort: NAQ desc (applies to all tables — sortAll sorts every table)
    out.append(f'<script>window.__defaultSort = {{colIdx: {naq_idx}, dir: -1}};</script>')
    out.append(SCRIPT)
    out.append('</body></html>')
    return "\n".join(out)


def _render_samples(ctx):
    """Render samples.html — by-prompt MOS-style gallery, successful rows only, ranked by NAQ."""
    meta = ctx["meta"]
    prompts = ctx["prompts_seen"]
    out = ['<!doctype html>',
           '<html lang="en"><head><meta charset="utf-8">',
           f'<title>TTS Bench — Samples — {escape(ctx["run_name"])}</title>',
           STYLE,
           '</head><body>',
           CONTROLS,
           _lens_nav("samples"),
           f'<h1>TTS Bench — Samples — {escape(ctx["run_name"])}</h1>']
    if meta:
        out.append(f'<div class="meta"><strong>Rig:</strong> '
                   f'<code>{escape(meta.get("rig") or "?")}</code> — '
                   f'{escape(_rig_summary(meta))}</div>')
        if meta.get("label"):
            ref = meta.get("reference")
            ref_html = f' — ref <code>{escape(ref)}</code>' if ref else ""
            out.append(f'<div class="meta"><strong>Label:</strong> '
                       f'{escape(meta["label"])}{ref_html}</div>')
    out.append(f'<div class="meta">{len(prompts)} prompt(s) · '
               f'one section per prompt · all models ranked by NAQ within each</div>')

    out.append(_READING_GUIDE["samples"])

    if len(prompts) > 3:
        out.append('<nav class="prompt-jumper">Jump to: ')
        out.append(" · ".join(f'<a href="#p{escape(pid)}">P{escape(pid)}</a>' for pid in prompts))
        out.append('</nav>')

    cols = ("Rank", "Model", "Device", "NAQ", "TTFA warm", "Audio")
    num_cols = {"Rank", "NAQ", "TTFA warm"}
    for pid in prompts:
        items = ctx["per_prompt"].get(pid, [])
        out.append(f'<div class="prompt" id="p{escape(pid)}"><h2>Prompt {escape(pid)}</h2>')
        ptext = PROMPT_INFO.get(pid)
        if ptext:
            lang, text = ptext
            out.append(f'<span class="prompt-text"><span class="lang">[{escape(lang)}]</span>'
                       f'"{escape(text)}"</span>')
        if not items:
            out.append('<div class="meta">No successful samples for this prompt.</div></div>')
            continue
        out.append('<table><thead><tr>')
        for c in cols:
            cls = ' class="num"' if c in num_cols else ''
            out.append(f'<th{cls}>{c}</th>')
        out.append('</tr></thead><tbody>')
        for rank, it in enumerate(items, 1):
            row_id = f"sample-p{pid}-{it['model']}-{it['device']}".lower().replace("/", "-")
            dev_class = f"dev-{it['device']}"
            audio_html = (
                f'<audio controls preload="none" src="{escape(it["wav"])}"></audio>'
                if it["wav_exists"] else '<span class="muted">missing</span>'
            )
            out.append(f'<tr id="{escape(row_id)}">')
            out.append(f'<td class="num" data-sort="{rank}">{rank}</td>')
            out.append(f'<td>{escape(_display_name(it["model"]))}</td>')
            out.append(f'<td class="{dev_class}">{escape(it["device"])}</td>')
            out.append(f'<td class="num"{_ds(it["naq"])}>{_fmt_naq(it["naq"])}</td>')
            out.append(f'<td class="num pill"{_ds(it["ttfa_warm"])}>{_fmt_ttfa(it["ttfa_warm"])}</td>')
            out.append(f'<td>{audio_html}</td>')
            out.append('</tr>')
        out.append('</tbody></table></div>')

    out.append(SCRIPT)
    out.append('</body></html>')
    return "\n".join(out)


def build_report(run_dir: Path) -> Path:
    """Emit index.html + speed.html + quality.html + samples.html from results.csv.
    Returns the path to index.html (the lens picker)."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise SystemExit(f"No results.csv in {run_dir}")
    rows = _read_csv(csv_path)
    if not rows:
        raise SystemExit(f"Empty results.csv in {run_dir}")

    meta = _read_meta(run_dir)
    ctx = _build_context(rows, run_dir, meta)

    (run_dir / "index.html").write_text(_render_lens_picker(ctx), encoding="utf-8")
    (run_dir / "speed.html").write_text(_render_speed(ctx), encoding="utf-8")
    (run_dir / "quality.html").write_text(_render_quality(ctx), encoding="utf-8")
    (run_dir / "samples.html").write_text(_render_samples(ctx), encoding="utf-8")

    # Legacy report.html — emit a small redirect stub so any external link still works.
    legacy = run_dir / "report.html"
    legacy.write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0; url=index.html">'
        '<title>Redirecting…</title></head>'
        '<body><a href="index.html">Lens picker</a></body></html>',
        encoding="utf-8",
    )
    return run_dir / "index.html"


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
            "has_index": (d / "index.html").exists(),
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
    num_cols = {"Prompts", "Rows", "OK"}
    for col in ("Run", "Label", "Rig", "Models", "Devices", "Prompts", "Rows", "OK", "Report"):
        cls = ' class="num"' if col in num_cols else ''
        out.append(f'<th{cls}>{col}</th>')
    out.append('</tr></thead><tbody>')

    for r in runs:
        models = (", ".join(r["models"])
                  if len(r["models"]) <= 5
                  else f"{len(r['models'])} models")
        if r["has_index"]:
            link = (
                f'<a href="{escape(r["name"])}/speed.html">speed</a> · '
                f'<a href="{escape(r["name"])}/quality.html">quality</a> · '
                f'<a href="{escape(r["name"])}/samples.html">samples</a>'
            )
        elif (RESULTS / r["name"] / "report.html").exists():
            link = f'<a href="{escape(r["name"])}/report.html">view (legacy)</a>'
        else:
            link = '<span class="muted">no report</span>'
        rig_cell = (f'<code title="{escape(r["rig_full"])}">{escape(r["rig"])}</code>'
                    if r["rig"] else '<span class="muted">—</span>')
        label_cell = (escape(r["label"])
                      if r["label"]
                      else '<span class="muted">—</span>')
        out.append('<tr>')
        if r["has_index"]:
            out.append(f"<td><a href='{escape(r['name'])}/index.html'>{escape(r['name'])}</a></td>")
        else:
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
            print(f"  wrote {html.relative_to(REPO)} (+ 3 lens pages)")
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
    print(f"wrote {html.relative_to(REPO)} (+ speed.html, quality.html, samples.html)")
    build_index()
    if args.open:
        webbrowser.open(html.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
