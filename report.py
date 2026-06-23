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

# NAQ left the harness — it's now private lab-only R&D (see naq_lab/, gitignored).
# The bench no longer computes it and results.csv carries no naq columns, so the
# report has no quality lens / NAQ column. All NAQ scoring happens post-hoc in
# naq_lab over the saved wavs.

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

  /* TLDR callout used by speed.html */
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
  .lens-tabs { display: inline-flex; gap: 0.3rem; margin-right: 0.6rem;
               align-items: center; }
  .lens-tab { display: inline-block; padding: 4px 12px; border-radius: 6px;
              border: 1px solid var(--border); color: var(--text);
              text-decoration: none; font-size: 0.92em;
              transition: border-color 0.15s, background 0.15s; }
  .lens-tab:hover { border-color: var(--accent); text-decoration: none; }
  .lens-tab.active { background: var(--accent); color: var(--bg);
                     border-color: var(--accent); }
  .lens-arrow { display: inline-block; padding: 4px 10px; border-radius: 6px;
                border: 1px solid var(--border); color: var(--muted);
                text-decoration: none; font-size: 0.92em;
                transition: border-color 0.15s, color 0.15s; }
  .lens-arrow:hover { border-color: var(--accent); color: var(--accent);
                      text-decoration: none; }
  .lens-arrow-disabled { opacity: 0.35; cursor: default;
                         border-style: dashed; }
  .lens-arrow-disabled:hover { border-color: var(--border); color: var(--muted); }

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


def _humanize_error(err):
    """Turn a raw runner error into a reader-friendly reason so a failed cell
    doesn't read like a hardware fault. An OOM means the model is too big for
    this device's memory, not that the rig broke."""
    e = (err or "").strip()
    low = e.lower()
    if not e or e == "no successful run":
        return "no successful run"
    # "invalid buffer size" is how MPS reports an allocation it can't satisfy —
    # an out-of-memory by another name.
    if "outofmemory" in low or "out of memory" in low or "invalid buffer size" in low:
        if "cuda" in low or "gpu" in low:
            return "Skipped — out of GPU memory (model exceeds this GPU's VRAM)"
        return "Skipped — out of memory (model exceeds available RAM)"
    if "timeout" in low and "s" in low:
        # "timeout 600s" → 600
        import re
        m = re.search(r"timeout\s+(\d+)s", low)
        if m:
            secs = int(m.group(1))
            return f"Timed out after {secs // 60} min — model too slow at this prompt length"
        return "Timed out — model too slow at this prompt length"
    if "torchcodec" in low or "libtorchcodec" in low:
        return "Audio loader needs FFmpeg DLLs (not present on this platform)"
    if "flash_attn" in low and ("not installed" in low or "no module" in low):
        return "Needs flash_attn (no wheel for this Python/torch/OS combo)"
    if "no module named" in low and "luxtts" in low or "no module named 'zipvoice'" in low:
        return "LuxTTS install failed (piper-phonemize has no Windows wheels)"
    if "no module named" in low:
        import re
        m = re.search(r"no module named ['\"]?([^'\"\s]+)['\"]?", low)
        if m:
            return f"Missing dependency '{m.group(1)}' — install incomplete on this platform"
        return "Missing dependency — install incomplete on this platform"
    if "repo id must use alphanumeric" in low:
        return "transformers/Windows repo-id path bug (workaround pending)"
    if "mutable default" in low and "override" in low:
        return "hydra-core/omegaconf too old for Python 3.11 (deps need bump)"
    if ("typeerror" in low and "'nonetype' object cannot be interpreted as an integer" in low):
        return "Upstream config field missing — dep version mismatch"
    if "cuda error" in low or "cuda runtime" in low:
        return "CUDA runtime error — likely driver/version mismatch"
    if "cuda requested but not available" in low:
        return "CUDA not available on this rig"
    if "mps requested but not available" in low:
        return "MPS not available on this rig"
    if "cuda only" in low or "cuda-only" in low:
        return "Model is GPU-only — CPU/MPS not supported"
    if "reference" in low and ("not found" in low or "missing" in low):
        return "Reference wav missing for voice cloning"
    # Fallback: surface the first line, trimmed.
    return f"Failed: {e.splitlines()[0][:140]}"


# Display sizes for the Size column. Numbers are the model's weight count
# (params), not weights-on-disk. Hand-curated — keep in sync with README.
# Display name shown in the rendered HTML for each venv-keyed model.
# CSV column still uses the raw venv key; this is presentation-only.
# Keys not present here fall through to the raw key (Title-cased by convention).
MODEL_DISPLAY_NAMES = {
    "sesame":        "Sesame CSM-1B",
    "miso":          "Miso TTS 8B",
    "longcat_1b":    "LongCat-AudioDiT 1B",
    "longcat_3p5b":  "LongCat-AudioDiT 3.5B",
    "orpheus":       "Orpheus-TTS 3B",
    "cosyvoice":     "CosyVoice 3 0.5B",
    "lfm2_audio":    "LFM2.5-Audio 1.5B",
    "miotts_01b":    "MioTTS 0.1B",
    "miotts_06b":    "MioTTS 0.6B",
    "coqui":         "Coqui XTTS-v2",
    "vibevoice":     "VibeVoice Realtime 0.5B",
    "vibevoice_15b": "VibeVoice 1.5B",
    "qwentts":       "Qwen3-TTS 1.7B Base",
    "qwentts_fast":  "Qwen3-TTS 1.7B (CUDA-graph)",
    "neutts_air":    "NeuTTS Air",
    "neutts_nano":   "NeuTTS Nano",
    "voxcpm":        "VoxCPM2 2B",
    "mars5":         "Mars5-TTS",
    "pocket":        "Pocket-TTS",
    "magpie":        "Magpie-TTS",
    "indextts":      "IndexTTS-2",
    "f5tts":         "F5-TTS v1",
    "chatterbox":       "Chatterbox",
    "chatterbox_turbo": "Chatterbox Turbo",
    "dia":           "Dia 1.6B-0626",
    "omnivoice":     "OmniVoice",
    "zipvoice":      "ZipVoice 123M",
    "piper":         "Piper",
    "kokoro":        "Kokoro",
    "kittentts":     "KittenTTS Nano 0.1",
    "soprano":       "Soprano 1.1 80M",
    "moss_tts_nano": "MOSS-TTS-Nano",
    "moss_tts":      "MOSS-TTS v1.0",
    "moss_tts_v15":  "MOSS-TTS v1.5",
    "supertonic":    "Supertonic 3",
    "luxtts":        "LuxTTS",
    "maya1":         "Maya1",
    "voxtral":       "Voxtral 4B TTS",
    "fish_15":       "Fish Speech 1.5",
    "fish_s2":       "Fish Speech S2-Pro",
    "echo":          "Echo-TTS",
    "zonos":         "Zonos v0.1",
    "openvoice":     "OpenVoice v2",
    "styletts2":     "StyleTTS 2",
    "vibevoice_7b":  "VibeVoice 7B",
    "metavoice":     "MetaVoice-1B",
    "step_editx":    "Step-Audio-EditX",
    "echo":          "Echo-TTS",
    "miratts":       "MiraTTS",
    "outetts":       "OuteTTS 1.0 1B",
    "parler":        "Parler-TTS Mini v1",
    "melotts":       "MeloTTS",
    "higgs":         "Higgs Audio v2 3B",
    "higgs_v3":      "Higgs Audio v3 TTS",
    "dramabox":      "DramaBox",
    "dots_tts":      "dots.tts (soar)",
}


def _display_name(model):
    """Return display name for a model venv key; falls back to raw key."""
    return MODEL_DISPLAY_NAMES.get(model, model)


MODEL_SIZE = {
    "pocket":        "100M",
    "neutts_air":    "748M",
    "neutts_nano":   "229M",
    "luxtts":        "123M",
    "chatterbox":       "1.2B",
    "chatterbox_turbo": "744M",
    "f5tts":         "330M",
    "coqui":         "750M",
    "vibevoice":     "0.5B",
    "vibevoice_15b": "1.5B",
    "omnivoice":     "~1B",
    "zipvoice":      "123M",
    "voxcpm":        "2B",
    "magpie":        "357M",
    "qwentts":       "1.7B",
    "qwentts_fast":  "1.7B",
    "indextts":      "1.5B",
    "sesame":        "1B",
    "miso":          "8.2B",
    "longcat_1b":    "1.42B",
    "longcat_3p5b":  "3.83B",
    "orpheus":       "3.3B",   # Llama-3.2-3B backbone + SNAC decoder
    "cosyvoice":     "0.5B",   # Fun-CosyVoice3-0.5B-2512
    "lfm2_audio":    "1.5B",   # 1.2B LFM2.5 LM + 115M FastConformer encoder
    "miotts_01b":    "0.1B",   # Falcon-H1-Tiny-Multilingual-100M backbone
    "miotts_06b":    "0.6B",   # Qwen3-0.6B-Base backbone
    "mars5":         "1.2B",
    "dia":           "1.6B",
    "kokoro":        "82M",
    "kittentts":     "<100M",
    "piper":         "~25MB",
    "soprano":       "80M",
    "moss_tts_nano": "100M",
    "moss_tts":      "8B",
    "moss_tts_v15":  "8B",
    "supertonic":    "99M",
    "maya1":         "3B",
    "voxtral":       "4B",
    "fish_15":       "~500M",
    "fish_s2":       "4B",
    "echo":          "~2.8B",
    "zonos":         "1.6B",
    "openvoice":     "~100M",
    "styletts2":     "~148M",
    "vibevoice_7b":  "7B",
    "metavoice":     "1.2B",
    "step_editx":    "3B",
    "echo":          "2.8B",   # DiT generative model (safetensors count); +695M S1-DAC codec not counted (matches fish_15 convention)
    "miratts":       "0.5B",   # HF card "Model size 0.5B params" (BF16); FastBiCodec + FlashSR not counted
    "outetts":       "1B",     # Llama-3.2-1B backbone
    "parler":        "878M",   # parler-tts-mini-v1 safetensors total; large variant = 2.33B
    "melotts":       "~52M",   # MeloTTS-English checkpoint.pth (~208 MB fp32 / 4)
    "higgs":         "3.6B",   # generation LLM; +2.2B audio adapter (DualFFN) not counted (echo/fish convention)
    "higgs_v3":      "4B",     # Qwen3 ~4B backbone (HF card); Higgs audio tokenizer not counted
    "dramabox":      "3.3B",   # LTX-2.3 audio-only DiT (IC-LoRA merged); 12B 4-bit text encoder not counted
    "dots_tts":      "2B",     # semantic encoder + LLM + flow-matching acoustic head (soar/SCA checkpoint)
}

# Canonical URL of the EXACT checkpoint each runner loads (HF model card, or the
# GitHub repo when that's the pinned source). Extracted from runners/*.py and the
# installed venv packages — keep in sync with the README model tables, which carry
# the same links. Consumed by arena/build_manifest.py for the post-vote reveal.
_HF = "https://huggingface.co/"
MODEL_URL = {
    "pocket":        "https://github.com/kyutai-labs/pocket-tts",
    "neutts_air":    _HF + "neuphonic/neutts-air",
    "neutts_nano":   _HF + "neuphonic/neutts-nano-q4-gguf",
    "luxtts":        "https://github.com/ysharma3501/LuxTTS",
    "chatterbox":       _HF + "ResembleAI/chatterbox",
    "chatterbox_turbo": _HF + "ResembleAI/chatterbox-turbo",
    "f5tts":         _HF + "SWivid/F5-TTS",
    "coqui":         _HF + "coqui/XTTS-v2",
    "vibevoice":     _HF + "microsoft/VibeVoice-Realtime-0.5B",
    "vibevoice_15b": _HF + "microsoft/VibeVoice-1.5B",
    "vibevoice_7b":  _HF + "vibevoice/VibeVoice-7B",
    "omnivoice":     _HF + "k2-fsa/OmniVoice",
    "zipvoice":      _HF + "k2-fsa/ZipVoice",
    "voxcpm":        _HF + "openbmb/VoxCPM2",
    "magpie":        _HF + "nvidia/magpie_tts_multilingual_357m",
    "qwentts":       _HF + "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "qwentts_fast":  _HF + "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "indextts":      _HF + "IndexTeam/IndexTTS-2",
    "sesame":        _HF + "sesame/csm-1b",
    "miso":          _HF + "MisoLabs/MisoTTS",
    "longcat_1b":    _HF + "meituan-longcat/LongCat-AudioDiT-1B",
    "longcat_3p5b":  _HF + "meituan-longcat/LongCat-AudioDiT-3.5B",
    "orpheus":       _HF + "canopylabs/orpheus-3b-0.1-ft",
    "cosyvoice":     _HF + "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "lfm2_audio":    _HF + "LiquidAI/LFM2.5-Audio-1.5B",
    "miotts_01b":    _HF + "Aratako/MioTTS-0.1B",
    "miotts_06b":    _HF + "Aratako/MioTTS-0.6B",
    "mars5":         _HF + "Camb-ai/mars5-tts",
    "dia":           _HF + "nari-labs/Dia-1.6B-0626",
    "kokoro":        _HF + "hexgrad/Kokoro-82M",
    "kittentts":     _HF + "KittenML/kitten-tts-nano-0.1",
    "piper":         "https://github.com/OHF-Voice/piper1-gpl",
    "soprano":       _HF + "ekwek/Soprano-1.1-80M",
    "moss_tts_nano": _HF + "OpenMOSS-Team/MOSS-TTS-Nano",
    "moss_tts":      _HF + "OpenMOSS-Team/MOSS-TTS",
    "moss_tts_v15":  _HF + "OpenMOSS-Team/MOSS-TTS-v1.5",
    "supertonic":    _HF + "Supertone/supertonic-3",
    "maya1":         _HF + "maya-research/maya1",
    "voxtral":       _HF + "mistralai/Voxtral-4B-TTS-2603",
    "fish_15":       _HF + "fishaudio/fish-speech-1.5",
    "fish_s2":       _HF + "fishaudio/s2-pro",
    "echo":          _HF + "jordand/echo-tts-base",
    "zonos":         _HF + "Zyphra/Zonos-v0.1-transformer",
    "openvoice":     _HF + "myshell-ai/OpenVoiceV2",
    "styletts2":     "https://github.com/yl4579/StyleTTS2",
    "metavoice":     _HF + "metavoiceio/metavoice-1B-v0.1",
    "step_editx":    _HF + "stepfun-ai/Step-Audio-EditX",
    "miratts":       _HF + "YatharthS/MiraTTS",
    "outetts":       _HF + "OuteAI/Llama-OuteTTS-1.0-1B",
    "parler":        _HF + "parler-tts/parler-tts-mini-v1",
    "melotts":       _HF + "myshell-ai/MeloTTS-English",
    "higgs":         _HF + "bosonai/higgs-audio-v2-generation-3B-base",
    "higgs_v3":      _HF + "bosonai/higgs-audio-v3-tts-4b",
    "dramabox":      "https://github.com/resemble-ai/DramaBox",
    "dots_tts":      _HF + "rednote-hilab/dots.tts-soar",
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
    "vibevoice_15b": "cloning",   # no preset voice — clones from a reference wav
    "soprano":       "predefined",
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
    "miso":          "cloning",
    "longcat_1b":    "cloning",
    "longcat_3p5b":  "cloning",
    "orpheus":       "predefined",   # named preset voices (tara, ...), no wav cloning
    "cosyvoice":     "cloning",
    "lfm2_audio":    "predefined",   # 4 preset voices (US/UK x m/f) via system prompt; no wav cloning
    "miotts_01b":    "cloning",      # zero-shot clone; no-reference run = bundled chris clip
    "miotts_06b":    "cloning",
    "mars5":         "cloning",
    "dia":           "cloning",
    "neutts_air":    "cloning",
    "neutts_nano":   "cloning",
    "moss_tts_nano": "cloning",
    "moss_tts":      "cloning",
    "moss_tts_v15":  "cloning",
    "maya1":         "predefined",   # voice-description preset, no wav cloning
    "voxtral":       "cloning",      # cuda/vLLM path clones; MLX path preset-only
    "fish_15":       "cloning",
    "fish_s2":       "cloning",
    "echo":          "cloning",
    "zonos":         "cloning",
    "openvoice":     "cloning",
    "styletts2":     "cloning",
    "vibevoice_7b":  "cloning",
    "metavoice":     "cloning",
    "step_editx":    "cloning",
    "echo":          "cloning",   # no preset voice — zero-shot clones from a reference wav
    "miratts":       "cloning",   # no preset voice — zero-shot clones from a reference wav
    "outetts":       "cloning",   # clones from a reference wav; ALSO has preset voices (default lens)
    "parler":        "predefined",   # voice set by a text description, no wav cloning
    "melotts":       "predefined",   # VITS preset speakers (EN-US), no wav cloning
    "higgs":         "cloning",   # in-context cloning from a reference wav + transcript
    "higgs_v3":      "cloning",   # zero-shot cloning; no-reference run = its own default voice
    "dramabox":      "cloning",   # 10s+ wav cloning; no-reference run = prompt-described voice
    "dots_tts":      "cloning",   # zero-shot cloning (+ ref transcript); no-reference run = bundled chris clip
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
                      "peak_mem_mb", "peak_vram_mb", "wall_s"):
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
      per_model:   {(model, device) -> aggregated dict with avg TTFA/RTF, peak mem/vram,
                    n_ok, n_fail, n_total}
      per_prompt:  {prompt_id -> [{model, device, ttfa_warm, rtf_warm, wav, wav_exists}
                    ranked by warm TTFA asc (fastest first), successful-cold rows only]}
      tldr_speed:  {"predefined": (model, device, rtf_warm, ttfa_warm) | None,
                    "cloning":    (model, device, rtf_warm, ttfa_warm) | None}
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
            "peak_mem": [], "peak_vram": [],
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
                "ttfa_warm": ttfa_warm,
                "rtf_warm": rtf_warm,
                "wav": f"{model}_{dev}_p{pid}.wav",
                "wav_exists": (run_dir / f"{model}_{dev}_p{pid}.wav").exists(),
            })
        # Sort by warm TTFA asc (fastest first); missing values sink to the bottom.
        items.sort(key=lambda it: (it["ttfa_warm"] is None,
                                   it["ttfa_warm"] or float("inf")))
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

    return {
        "per_cell": cells,
        "per_model": agg,
        "per_prompt": per_prompt,
        "tldr_speed": {
            "predefined": _pick_best_speed("predefined"),
            "cloning":    _pick_best_speed("cloning"),
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


_LENSES = (("speed", "Speed"), ("samples", "Samples"))

_READING_GUIDE = {
    "speed": (
        '<div class="reading-guide">'
        '<strong>TTFA</strong> = time to first audio (ms; lower is better). '
        '<strong>RTF</strong> = real-time factor (× realtime; higher is better; e.g. 10× means '
        '10 sec of audio generated per 1 sec of compute). '
        '<strong>Cold</strong> = first run after process start; <strong>warm</strong> = subsequent runs.'
        '</div>'
    ),
    "samples": (
        '<div class="reading-guide">'
        'Each prompt section shows every model\'s audio output, ordered by '
        '<strong>warm TTFA</strong> (fastest first). '
        'Click any audio player to hear that model\'s rendering.'
        '</div>'
    ),
}


def _lens_nav(active):
    """Emit the nav strip with prev/next arrows + lens tabs + 'all runs' link.

    `active` is one of the slugs in _LENSES. Arrows are bounded — at the
    first lens the ← arrow renders disabled (no link); at the last lens
    the → arrow renders disabled. No wraparound, so the report tour ends
    at the natural edges instead of cycling infinitely.
    """
    slugs = [s for s, _ in _LENSES]
    labels = {s: l for s, l in _LENSES}
    idx = slugs.index(active)
    prev_slug = slugs[idx - 1] if idx > 0 else None
    next_slug = slugs[idx + 1] if idx < len(slugs) - 1 else None

    parts = ['<div class="nav"><span class="lens-tabs">']
    if prev_slug is not None:
        parts.append(
            f'<a class="lens-arrow" href="{prev_slug}.html" '
            f'title="Previous lens: {labels[prev_slug]}">← {labels[prev_slug]}</a>'
        )
    else:
        parts.append('<span class="lens-arrow lens-arrow-disabled">← </span>')
    for slug, label in _LENSES:
        cls = "lens-tab active" if slug == active else "lens-tab"
        parts.append(f'<a class="{cls}" href="{slug}.html">{label}</a>')
    if next_slug is not None:
        parts.append(
            f'<a class="lens-arrow" href="{next_slug}.html" '
            f'title="Next lens: {labels[next_slug]}">{labels[next_slug]} →</a>'
        )
    else:
        parts.append('<span class="lens-arrow lens-arrow-disabled"> →</span>')
    parts.append('</span> · <a href="../index.html">all runs</a></div>')
    return "".join(parts)


def _speed_table_html(ctx):
    """Return (table_html, rtf_warm_col_idx) for one run's speed view.

    Just the <table> (aggregated per model/device, no audio) — shared by the
    per-rig speed.html and the cross-rig speed hub on the published landing.
    The caller decides where to place it and whether to set window.__defaultSort
    (rtf_warm_col_idx is returned so the caller can default-sort by warm RTF)."""
    cols = ("Model", "Device", "TTFA cold", "TTFA warm",
            "RTF cold", "RTF warm", "Peak RAM", "Peak VRAM", "Size")
    num_cols = {"TTFA cold", "TTFA warm", "RTF cold", "RTF warm",
                "Peak RAM", "Peak VRAM"}
    rtf_warm_idx = cols.index("RTF warm")
    out = ['<table><thead><tr>']
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
                       f'<td colspan="{len(cols)-2}" class="fail">{escape(_humanize_error(err))}</td>'
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
        out.append('</tr>')
    out.append('</tbody></table>')
    return "".join(out), rtf_warm_idx


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

    # Table (shared builder — same markup the cross-rig speed hub uses)
    table_html, rtf_warm_idx = _speed_table_html(ctx)
    out.append(table_html)

    # Default sort: RTF warm desc
    out.append(f'<script>window.__defaultSort = {{colIdx: {rtf_warm_idx}, dir: -1}};</script>')
    out.append(SCRIPT)
    out.append('</body></html>')
    return "\n".join(out)


def _render_samples(ctx):
    """Render samples.html — by-prompt gallery, successful rows only, ordered by warm TTFA."""
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
    rank_basis = "warm TTFA (fastest first)"
    out.append(f'<div class="meta">{len(prompts)} prompt(s) · '
               f'one section per prompt · all models ranked by {rank_basis} within each</div>')

    out.append(_READING_GUIDE["samples"])

    # Reference-voice player for cloning runs: each model in the table below
    # is trying to imitate this voice, so the listener needs to hear it to
    # judge fidelity. publish.py copies the source wav into the slug dir as
    # `_reference.wav`; if absent (older runs / non-cloning runs) we skip.
    ref_meta = (meta or {}).get("reference") if meta else None
    if ref_meta and (ctx["run_dir"] / "_reference.wav").exists():
        out.append(
            '<div class="prompt" style="border-left: 3px solid var(--accent);">'
            '<h2>Reference voice</h2>'
            f'<div class="meta">Each model below was given this clip + transcript as the voice to imitate. '
            f'Source: <code>{escape(ref_meta)}</code></div>'
            '<audio controls preload="metadata" src="_reference.wav" '
            'style="width: 100%; max-width: 480px;"></audio>'
            '</div>'
        )

    if len(prompts) > 3:
        out.append('<nav class="prompt-jumper">Jump to: ')
        out.append(" · ".join(f'<a href="#p{escape(pid)}">P{escape(pid)}</a>' for pid in prompts))
        out.append('</nav>')

    cols = ("Rank", "Model", "Device", "TTFA warm", "Audio")
    num_cols = {"Rank", "TTFA warm"}
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
            out.append(f'<td class="num pill"{_ds(it["ttfa_warm"])}>{_fmt_ttfa(it["ttfa_warm"])}</td>')
            out.append(f'<td>{audio_html}</td>')
            out.append('</tr>')
        out.append('</tbody></table></div>')

    out.append(SCRIPT)
    out.append('</body></html>')
    return "\n".join(out)


def build_report(run_dir: Path) -> Path:
    """Emit index.html + speed.html + samples.html from results.csv.
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
    # NAQ left the harness — drop any quality.html left over from an older run.
    (run_dir / "quality.html").unlink(missing_ok=True)
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
            parts = [f'<a href="{escape(r["name"])}/speed.html">speed</a>']
            parts.append(f'<a href="{escape(r["name"])}/samples.html">samples</a>')
            link = " · ".join(parts)
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
    print(f"wrote {html.relative_to(REPO)} (+ speed.html, samples.html)")
    build_index()
    if args.open:
        webbrowser.open(html.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
