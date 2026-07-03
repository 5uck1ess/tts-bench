"""Publish a bench run to the gh-pages branch for GitHub Pages hosting.

Copies a chosen results/<date>/ subdir (report.html + wavs + CSV) to a
worktree on the gh-pages branch, rebuilds a top-level index of published
runs, commits, and pushes. After GitHub Pages is enabled in repo settings
(branch: gh-pages, folder: /):

    https://<user>.github.io/<repo>/                       <- index
    https://<user>.github.io/<repo>/<run-name>/report.html <- one run

The master branch is never touched — gh-pages is managed via a separate
git worktree at _gh-pages/.

Usage:
    python publish.py results/2026-05-23_2203          # publish + push
    python publish.py results/2026-05-23_2203 --no-push # commit only
    python publish.py --list                            # list published runs
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from html import escape
from pathlib import Path

# Windows console defaults to cp1252; force UTF-8 so em-dashes / arrows in
# print() messages don't crash mid-run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from report import (
    STYLE, CONTROLS, SCRIPT, PROMPT_INFO, MODEL_SIZE, MODEL_KIND, MODEL_URL,
    MODEL_SR, MODEL_EXPRESSIVE, MODEL_LICENSE, MODEL_LANGS,
    _ds, _read_csv, _read_meta, _rig_summary, _sort_prompt_ids, build_report,
    _build_context, _speed_table_html, _display_name, _release_label, _release_td,
    _is_multilingual, _is_commercial, _sr_hz, _is_crosslingual,
)
from harness import MODELS as _HARNESS_MODELS

# Models whose runner advertises a CPU device (harness MODELS tuple, devices at
# index 4) — drives the "runs on CPU" capability flag/filter.
_CPU_OK = frozenset(row[0] for row in _HARNESS_MODELS if "cpu" in row[4])

# Cloning models that ALSO ship selectable preset voices (README: both Predefined
# and Cloning are ✓). Every other model is clone-only or preset-only via MODEL_KIND.
_PRESET_AND_CLONE = {"outetts", "voxtral"}

# Models that synthesize more than English (from the README capability table) —
# used only for the at-a-glance "multilingual" badge in the Listen by-model view.
MODEL_MULTILINGUAL = {
    "piper", "kokoro", "magpie", "supertonic", "f5tts", "indextts", "omnivoice",
    "zipvoice", "voxcpm", "coqui", "qwentts", "qwentts_fast", "moss_tts_nano",
    "moss_tts", "moss_tts_v15", "voxtral", "fish_15", "zonos", "openvoice", "dots_tts",
}

# Cloning models with NO preset/built-in voice: their no-`--reference` "default"
# run falls back to cloning the bundled reference clip, so the default sample is
# just a (different, non-deterministic take of a) Chris clone — not a real default
# voice. The Listen gallery shows these only under Cloning, never in the
# Default-voice section. (styletts2 → bundled LibriVox speaker, and VibeVoice 0.5B
# → the en-Emma_woman preset, DO have genuine defaults, so they're not here.)
NO_PRESET_VOICE = {
    "moss_tts", "moss_tts_v15", "moss_tts_nano", "fish_15", "fish_s2", "metavoice",
    "openvoice", "zipvoice", "zonos", "vibevoice_15b", "vibevoice_7b",
    "echo",
    # cosyvoice: pure zero-shot cloning, no model-native preset — its "default"
    # lens just clones the house ref (chris_hemsworth), like moss_tts.
    "cosyvoice",
    # miotts: pure zero-shot cloner (base64 ref), no model-native preset voice —
    # cloning board only, like cosyvoice. (Keep in sync with arena/build_manifest.py.)
    "miotts_01b", "miotts_06b",
    # wavtts: pure zero-shot cloner (ref wav + sibling .txt), no model-native preset —
    # its "default" lens clones the house ref, like cosyvoice. Cloning board only.
    "wavtts",
}

# Speed-only models: they carry a real speed row on the per-rig leaderboard but
# must NOT appear on Listen / Scores / Arena, because their audio is identical to
# another tracked model (same weights, different runtime). kokoro_mlx is the MLX
# twin of `kokoro` — published so the M4 speed board shows MLX vs PyTorch-MPS
# side by side, but its clips/scores would just duplicate PyTorch Kokoro's.
# Filtered out in _ok_models (Listen + Scores) and arena/build_manifest.py (Arena);
# the Speed hub reads CSV rows directly, so the row still shows there.
SPEED_ONLY = {"kokoro_mlx"}

# Curated per-(model, voice-mode) QA findings, surfaced as a small badge + tooltip on
# the model's row in the Listen gallery and recorded in docs/known-issues.md.
# Each value is (kind, label, note): kind "note" renders a neutral badge for a
# by-design behavior heads-up; kind "warn" renders a red badge for an actual defect.
KNOWN_ISSUES = {
    # qwentts_fast/cloning (runaway) is fixed via non_streaming_mode=True and re-benched;
    # fish_s2/cloning (wrong reference) was re-benched with Chris on Linux.
    ("higgs_v3", "default"): (
        "note", "⟳ voice varies",
        "Default voice is sampled fresh per generation (no fixed preset) — each prompt's "
        "clip is a different speaker. Cloning stays consistent with the reference."
    ),
    ("dots_tts", "default"): (
        "note", "⟳ voice varies",
        "Default voice is sampled fresh per generation (no fixed preset) — each prompt's "
        "clip is a different speaker. Cloning stays consistent with the reference."
    ),
    ("miso", "default"): (
        "warn", "⚠ artifacts",
        "Audible artifacts/glitches across generations (by-ear QA, 2026-06-10) — the "
        "default voice character is good but the texture is rough; UTMOS sits in the "
        "bench's bottom half and long prompts can run on past the text (high WER)."
    ),
    ("miso", "cloning"): (
        "warn", "⟳ clone unstable",
        "Voice retention from the reference is stochastic shot-to-shot: at upstream "
        "sampling (temp 0.9/topk 50) the clone usually loses the reference voice, so the "
        "bench runs cloning at temp 0.7/topk 30, which holds the voice most of the time "
        "but can still drift on a given generation (by-ear A/B, 2026-06-10; SIM ranges "
        "0.03-0.74 across clips). The artifact-prone texture flagged on the default lens "
        "applies here too."
    ),
    ("longcat_1b", "default"): (
        "note", "⟳ voice varies",
        "Default voice is sampled fresh per generation (no fixed preset) — each prompt's "
        "clip is a different speaker. Cloning stays consistent with the reference."
    ),
    ("longcat_3p5b", "default"): (
        "note", "⟳ voice varies",
        "Default voice is sampled fresh per generation (no fixed preset) — each prompt's "
        "clip is a different speaker. Cloning stays consistent with the reference."
    ),
    ("longcat_3p5b", "cloning"): (
        "warn", "⟳ clone less reliable",
        "The 3.5B clone is weaker than the 1B despite the larger model: it can lose the "
        "reference voice on a generation (p1 SIM 0.63 vs the 1B's ~0.87 mean) and carries "
        "the same texture issues — by-ear A/B + Seed-style SIM (2026-06-14) make the 1B the "
        "recommended LongCat variant for cloning."
    ),
    # cosyvoice is NO_PRESET_VOICE -> renders only on the cloning board (its default
    # clips fall back there), so a single cloning-lens note avoids a doubled badge.
    ("cosyvoice", "cloning"): (
        "note", "⟳ length varies",
        "Output length is unstable by design: short prompts can collapse to under a "
        "second (p1) and long prompts over-generate — the model's LLM decoder has no "
        "hard length cap. Inherent CosyVoice behavior at upstream defaults, not a bench defect."
    ),
    # miotts is NO_PRESET_VOICE -> cloning board only (like cosyvoice). Only the 0.6B
    # shows the instability; the 0.1B sibling is clean, so it carries no badge.
    ("miotts_06b", "cloning"): (
        "note", "⟳ length varies",
        "Output length is unstable: on the short prompt (p1) the LLM-codec decoder "
        "over-generated to ~17 s, dragging that clip to UTMOS 1.34 / WER 0.43. Inherent "
        "MioTTS-0.6B behavior at upstream defaults (same class as cosyvoice), not a bench "
        "defect — the 0.1B sibling stays clean."
    ),
    # Add new entries here as QA surfaces them — use "warn" only for genuine defects.
}


def _issue_badge(model, mode):
    """Badge for a known (model, voice-mode) note/issue, or '' if none. Neutral for a
    by-design behavior note ('note'), red for an actual defect ('warn')."""
    entry = KNOWN_ISSUES.get((model, mode))
    if not entry:
        return ""
    kind, label, note = entry
    klass = "badge warn" if kind == "warn" else "badge"
    return f' <span class="{escape(klass)}" title="{escape(note)}">{escape(label)}</span>'

REPO = Path(__file__).parent
WORKTREE = REPO / "_gh-pages"
BRANCH = "gh-pages"

SCORES_CSV = REPO / "scoring" / "scores.csv"

WER_FAIL_THRESHOLD = 0.5  # mean WER above this flags a model row as "broken"


def _plain_rows(path):
    """csv.DictReader rows without report._read_csv's numeric coercion."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_scores_csv():
    """Index scoring/scores.csv as {(dir, wav): {metric: float|None}}.
    Empty dict if the file is absent (Scores page then renders empty)."""
    look = {}
    if not SCORES_CSV.exists():
        return look
    for row in _plain_rows(SCORES_CSV):
        vals = {}
        for k in ("utmos", "wer", "sim"):
            v = row.get(k, "")
            vals[k] = float(v) if v not in ("", None) else None
        vals["health"] = (row.get("health", "") or "").strip()  # "" = clean / unscored
        look[(row["dir"], row["wav"])] = vals
    return look


def _model_scores(model, prompt_ids, dirs, look):
    """Mean of each metric over the canonical picked clip per prompt for one model.
    Returns {'utmos','wer','sim': float|None, 'n': int}. Blanks are skipped per
    metric; n = number of prompts that had a picked clip present in scores.csv."""
    acc = {"utmos": [], "wer": [], "sim": []}
    health_flags = []                         # [(prompt_id, "gap"), ...] over picked clips
    n = 0
    for pid in prompt_ids:
        picked = _pick_clip(dirs, model, pid)
        if not picked:
            continue
        rel = picked[0]                       # "<dir>/<wav>"
        dname, wav = rel.split("/", 1)
        row = look.get((dname, wav))
        if row is None:
            continue
        n += 1
        for k in acc:
            if row.get(k) is not None:
                acc[k].append(row[k])
        for flag in (row.get("health") or "").split(";"):
            if flag:
                health_flags.append((pid, flag))
    out = {k: (sum(v) / len(v) if v else None) for k, v in acc.items()}
    out["health_flags"] = health_flags
    out["n"] = n
    return out


# ---- Cross-rig consolidation for the Listen/Speed landing -------------------
# Canonical dir name = "<rig>-<mode>". LISTEN publishes ONE clip per
# (model, voice-mode): audio is rig-independent (same weights → same output),
# so we source a single sample from the highest-fidelity GREEN rig and tag it.
# Cloning sources skip Mac — the Mac cloning run used a different reference voice
# (jo.wav, not chris_hemsworth_15s), so its clips aren't comparable here.
LISTEN_DEFAULT_DIRS = ("windows-default", "linux-default", "mac-default")
LISTEN_CLONING_DIRS = ("windows-cloning", "linux-cloning")
LISTEN_DEVICE_PRIORITY = ("cuda", "mps", "cpu")  # prefer the GPU (fp16) path
SPEED_RIGS = (  # (rig slug, default dir, cloning dir)
    ("windows-5090", "windows-default", "windows-cloning"),
    ("linux-3090",   "linux-default",   "linux-cloning"),
    ("mac-m4",       "mac-default",     "mac-cloning"),
)
RIG_SHORT = {"windows-5090": "win", "linux-3090": "linux", "mac-m4": "mac"}


def _git(*args, cwd=REPO, check=True, capture=True):
    if capture:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    else:
        r = subprocess.run(["git", *args], cwd=cwd, text=True)
    if check and r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip() if capture else ""
        raise SystemExit(f"git {' '.join(args)} failed (cwd={cwd}):\n{msg}")
    return (r.stdout or "").strip()


def _branch_exists_local(branch):
    return bool(_git("branch", "--list", branch).strip())


def _branch_exists_remote(branch):
    return bool(_git("ls-remote", "--heads", "origin", branch, check=False).strip())


def _count_published():
    if not WORKTREE.exists():
        return 0
    return sum(
        1 for d in WORKTREE.iterdir()
        if d.is_dir() and not d.name.startswith(".") and (d / "index.html").exists()
    )


def ensure_worktree():
    """Ensure _gh-pages/ exists and is checked out to the gh-pages branch."""
    _git("worktree", "prune", check=False)

    if WORKTREE.exists() and (WORKTREE / ".git").exists():
        cur = _git("branch", "--show-current", cwd=WORKTREE)
        if cur != BRANCH:
            raise SystemExit(f"_gh-pages/ is on branch '{cur}', expected '{BRANCH}'.")
        return

    if _branch_exists_local(BRANCH):
        print(f"Adding worktree from local '{BRANCH}' branch...")
        _git("worktree", "add", str(WORKTREE), BRANCH)
        return

    if _branch_exists_remote(BRANCH):
        print(f"Adding worktree tracking origin/{BRANCH}...")
        _git("worktree", "add", "-b", BRANCH, str(WORKTREE), f"origin/{BRANCH}")
        return

    print(f"No '{BRANCH}' branch found — creating it as an orphan...")
    # git 2.42+ supports `git worktree add --orphan`; fall back if older.
    r = subprocess.run(
        ["git", "worktree", "add", "--orphan", "-b", BRANCH, str(WORKTREE)],
        cwd=REPO, capture_output=True, text=True,
    )
    if r.returncode != 0:
        # Older git path: detach worktree, then checkout --orphan, then wipe.
        _git("worktree", "add", "--detach", str(WORKTREE), "HEAD")
        _git("checkout", "--orphan", BRANCH, cwd=WORKTREE)
        for item in WORKTREE.iterdir():
            if item.name == ".git":
                continue
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        _git("rm", "-rf", "--cached", ".", cwd=WORKTREE, check=False)

    (WORKTREE / ".nojekyll").write_text("")
    (WORKTREE / "README.md").write_text(
        "# TTS Bench — Published Runs\n\n"
        "This branch hosts the static HTML reports + audio for "
        "[tts-bench](https://github.com/5uck1ess/tts-bench).\n"
        "Open `index.html` for the run list.\n",
        encoding="utf-8",
    )
    _git("add", "-A", cwd=WORKTREE)
    _git("commit", "-m", f"Initialize {BRANCH} branch", cwd=WORKTREE)


def _copy_branding_assets():
    """Copy logo + favicon assets from assets/ into _gh-pages/ so the
    deployed site can reference them with relative paths."""
    src_dir = REPO / "assets"
    if not src_dir.exists():
        return
    for name in ("logo-flat-dark.svg", "logo-flat-light.svg", "logo-mark.svg"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, WORKTREE / name)


LOGO_HEADER = (
    '<header class="site-header">'
    '<a href="index.html" class="site-logo-link" title="Home">'
    '<img class="site-logo site-logo--dark" src="logo-flat-dark.svg" '
    'alt="tts-bench" width="320" height="auto">'
    '<img class="site-logo site-logo--light" src="logo-flat-light.svg" '
    'alt="tts-bench" width="320" height="auto">'
    '</a>'
    '</header>'
)

LOGO_STYLE = (
    '<style>'
    '.site-header{margin:1rem 0 1.25rem;}'
    '.site-logo-link{display:inline-block;text-decoration:none;}'
    '.site-logo-link:hover{text-decoration:none;}'
    '.site-logo{display:block;height:auto;max-width:min(320px,80vw);}'
    '.site-logo--light{display:none;}'
    '[data-theme="light"] .site-logo--dark{display:none;}'
    '[data-theme="light"] .site-logo--light{display:block;}'
    '</style>'
)

FAVICON_LINK = (
    '<link rel="icon" type="image/svg+xml" href="logo-mark.svg">'
    '<link rel="icon" type="image/png" sizes="32x32" href="favicon.png">'
    '<link rel="apple-touch-icon" sizes="180x180" href="favicon-180.png">'
)


def _canonical_dir(name):
    """Return WORKTREE/name if it's a published run dir (has results.csv), else None."""
    d = WORKTREE / name
    return d if (d.is_dir() and (d / "results.csv").exists()) else None


def _ok_models(name):
    """Set of models with >=1 successful row in a canonical dir (empty if absent)."""
    d = _canonical_dir(name)
    if not d:
        return set()
    try:
        # SPEED_ONLY models are dropped here so Listen + Scores (which derive their
        # model sets from _ok_models) never show them; the Speed hub bypasses this.
        return {r["model"] for r in _read_csv(d / "results.csv") if r["ok"]} - SPEED_ONLY
    except Exception:
        return set()


def _all_prompt_ids(dirs):
    """Sorted union of prompt ids across the given canonical dirs."""
    ids = set()
    for name in dirs:
        d = _canonical_dir(name)
        if not d:
            continue
        try:
            ids |= {r["prompt_id"] for r in _read_csv(d / "results.csv")}
        except Exception:
            pass
    return _sort_prompt_ids(ids)


def _pick_clip(dirs, model, pid):
    """Highest-priority existing wav for (model, prompt) across dirs (in order).
    Within a dir, prefer the GPU path. Returns (relpath_from_root, rig_short, dev) or None."""
    for name in dirs:
        d = _canonical_dir(name)
        if not d:
            continue
        rig = (_read_meta(d) or {}).get("rig")
        for dev in LISTEN_DEVICE_PRIORITY:
            wav = d / f"{model}_{dev}_p{pid}.wav"
            if wav.exists():
                return (f"{name}/{wav.name}", RIG_SHORT.get(rig, rig or "?"), dev)
    return None


_LISTEN_GUIDE = (
    '<div class="reading-guide">One clip per model per voice mode. Audio is '
    '<strong>rig-independent</strong> (same weights → same output), so each sample is sourced '
    'once from the highest-fidelity available rig — Windows RTX 5090 where possible, else Linux, '
    'else Mac. The small tag on each row shows the source rig·device. '
    '<strong>Default voice</strong> = the model\'s own preset/built-in speaker; '
    '<strong>Cloning</strong> = the model imitating one reference voice '
    '(<code>chris_hemsworth_15s</code>). Switch <strong>By prompt</strong> (compare every '
    'model on one sentence) and <strong>By model</strong> (audition one model across prompts) '
    'below; only one clip plays at a time. Speed per rig is on the '
    '<a href="speed.html">Speed</a> page.</div>'
)

# Shared nesting style for the Default/Cloning sub-sections on both Listen and
# Speed: a slight left-indent + rule makes each a clear child of its parent
# (prompt on Listen, rig on Speed); cloning gets the accent rule + a reference
# player so each clone can be A/B'd against the target voice.
_SUBSECTION_STYLE = (
    '<style>'
    '.subsection{border-left:2px solid var(--border);padding:.1rem 0 .1rem 1.1rem;'
    'margin:.5rem 0 1.3rem .3rem;}'
    '.subsection.cloning{border-left-color:var(--accent);}'
    '.sub-head,.listen-group{margin:.3rem 0 .6rem;font-size:1.02em;color:var(--text);'
    'font-weight:600;}'
    '.ref-row{margin:.2rem 0 .9rem;font-size:.9em;color:var(--muted);'
    'display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;}'
    '.ref-row audio{width:260px;height:30px;vertical-align:middle;}'
    '.ref-row .ref-label{color:var(--accent);font-weight:600;}'
    '</style>'
)


_LISTEN_VIEW_STYLE = (
    '<style>'
    # collapsible prompt / model panels
    'details.panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;'
    'margin-bottom:1rem;padding:0 1.2rem;}'
    'details.panel>summary{cursor:pointer;padding:.85rem 0;font-weight:600;font-size:1.05em;'
    'color:var(--text);list-style:none;}'
    'details.panel>summary::-webkit-details-marker{display:none;}'
    'details.panel>summary::before{content:"\\25B8";color:var(--accent);margin-right:.55rem;'
    'display:inline-block;transition:transform .12s;}'
    'details.panel[open]>summary::before{transform:rotate(90deg);}'
    'details.panel>summary .summ-text{color:var(--prompt-text);font-style:italic;font-weight:400;}'
    # view toggle (By prompt / By model)
    '.view-toggle{display:inline-flex;gap:.3rem;margin:.2rem 0 1rem;}'
    '.view-toggle button{background:var(--input-bg);color:var(--text);'
    'border:1px solid var(--input-border);border-radius:6px;padding:6px 14px;'
    'font:inherit;cursor:pointer;}'
    '.view-toggle button.active{background:var(--accent);color:var(--bg);border-color:var(--accent);}'
    # at-a-glance metadata badges
    '.badge{display:inline-block;font-size:.72em;padding:1px 7px;border-radius:10px;'
    'border:1px solid var(--border);color:var(--muted);margin-left:.45rem;vertical-align:middle;}'
    '.badge.clone{color:var(--accent);border-color:var(--accent);}'
    '.badge.multi{color:var(--prompt-text);border-color:var(--prompt-text);}'
    '.badge.warn{color:var(--fail);border-color:var(--fail);}'
    '.summary-meta{color:var(--muted);font-weight:400;font-size:.85em;}'
    '</style>'
)


def _top_controls(active):
    """Sticky control bar for the top-level Listen/Speed pages: a persistent lens
    switcher + the filter/reset/theme controls SCRIPT (from report.py) wires to by id."""
    tabs = "".join(
        f'<a class="lens-tab{" active" if s == active else ""}" href="{s}.html">{l}</a>'
        for s, l in (("listen", "Listen"), ("speed", "Speed"), ("scores", "Scores"),
                     ("capabilities", "Capabilities"))
    )
    # Vote is an external link to the HF arena (not a local data view), so it
    # opens in a new tab and never takes the "active" state.
    tabs += ('<a class="lens-tab lens-tab-ext" href="https://5uck1ess-tts-arena.hf.space" '
             'target="_blank" rel="noopener">🗳 Vote ↗</a>')
    return ('<div class="controls">'
            f'<span class="lens-tabs">{tabs}</span>'
            '<input id="filter" type="search" placeholder="filter by model name…" autocomplete="off">'
            '<button type="button" id="reset-sort">reset sort</button>'
            '<span class="spacer"></span>'
            '<button type="button" id="theme-toggle" title="Toggle theme">☾ dark</button>'
            '</div>')


# Listen-page behaviour layered on top of the shared SCRIPT: single-play audio,
# the view toggle, filter-opens-collapsed-sections, and hash-jump expansion.
_LISTEN_SCRIPT = '''<script>
(function(){
  // Audio-comparison UX: only one clip plays at a time. 'play' doesn't bubble → capture.
  document.addEventListener('play', function(e){
    document.querySelectorAll('audio').forEach(function(a){ if(a !== e.target) a.pause(); });
  }, true);

  // View toggle: By prompt / By model.
  var byPrompt = document.getElementById('view-by-prompt');
  var byModel  = document.getElementById('view-by-model');
  document.querySelectorAll('.view-toggle button').forEach(function(b){
    b.addEventListener('click', function(){
      var v = b.dataset.view;
      document.querySelectorAll('.view-toggle button').forEach(function(x){ x.classList.toggle('active', x === b); });
      if(byPrompt) byPrompt.style.display = (v === 'by-prompt') ? '' : 'none';
      if(byModel)  byModel.style.display  = (v === 'by-model')  ? '' : 'none';
    });
  });

  // Filtering: open collapsed panels so matches inside are visible, and hide
  // whole model cards in the by-model view whose name doesn't match.
  var f = document.getElementById('filter');
  if(f) f.addEventListener('input', function(){
    var q = this.value.trim().toLowerCase();
    if(q){ document.querySelectorAll('#view-by-prompt details.panel').forEach(function(d){ d.open = true; }); }
    document.querySelectorAll('#view-by-model details.panel').forEach(function(d){
      var name = d.querySelector('summary').textContent.toLowerCase();
      var hit = !q || name.indexOf(q) !== -1;
      d.style.display = hit ? '' : 'none';
      if(hit && q) d.open = true;
    });
  });

  // Prompt-jumper anchors expand their target prompt.
  function openHash(){
    var h = location.hash.slice(1);
    if(!h) return;
    var d = document.getElementById(h);
    if(d && d.tagName === 'DETAILS') d.open = true;
  }
  window.addEventListener('hashchange', openHash);
  openHash();
})();
</script>'''


def build_listen():
    """Build listen.html — one consolidated gallery: every model on every prompt,
    default voice + cloning, one clip per model sourced from the best rig."""
    prompts = _all_prompt_ids(LISTEN_DEFAULT_DIRS) or _all_prompt_ids(LISTEN_CLONING_DIRS)
    raw_default = set().union(*(_ok_models(n) for n in LISTEN_DEFAULT_DIRS)) if LISTEN_DEFAULT_DIRS else set()
    raw_cloning = set().union(*(_ok_models(n) for n in LISTEN_CLONING_DIRS)) if LISTEN_CLONING_DIRS else set()
    # No-preset models clone the reference even in their "default" run, so their
    # default sample is a Chris clone, not a real default voice — show them only
    # under Cloning. Drop them from Default; fold them into the cloning set.
    default_models = raw_default - NO_PRESET_VOICE
    cloning_models = raw_cloning | (NO_PRESET_VOICE & (raw_default | raw_cloning))

    # The target voice every cloning model imitates. publish() stages the source
    # clip into each cloning run dir as _reference.wav; embed it at the top of the
    # Cloning group so each clone can be A/B'd against what it was trying to match.
    cloning_ref = None
    for n in LISTEN_CLONING_DIRS:
        cd = _canonical_dir(n)
        if cd and (cd / "_reference.wav").exists():
            cloning_ref = f"{n}/_reference.wav"
            break

    def _by_name(models):
        return sorted(models, key=lambda m: _display_name(m).lower())

    # Pick each clip once, then render it into both the by-prompt and by-model views.
    def _clips_for(dirs, models):
        picked = {}  # model -> {pid: (src, rig_short, dev)}
        for m in models:
            mp = {pid: c for pid in prompts if (c := _pick_clip(dirs, m, pid))}
            if mp:
                picked[m] = mp
        return picked

    def _cloning_clips(models):
        """Cloning clips, from the cloning dirs. A no-preset model benched only in
        default mode (e.g. vibevoice_15b) cloned the reference there, so fall back
        to its default-dir clip so it still appears under Cloning."""
        picked = {}
        for m in models:
            mp = {pid: c for pid in prompts if (c := _pick_clip(LISTEN_CLONING_DIRS, m, pid))}
            if not mp and m in NO_PRESET_VOICE:
                mp = {pid: c for pid in prompts if (c := _pick_clip(LISTEN_DEFAULT_DIRS, m, pid))}
            if mp:
                picked[m] = mp
        return picked

    default_clips = _clips_for(LISTEN_DEFAULT_DIRS, default_models)
    cloning_clips = _cloning_clips(cloning_models)

    def _audio(src):
        return f'<audio controls preload="none" src="{escape(src)}"></audio>'

    def _src_cell(rig_short, dev):
        return f'<td class="pill dev-{escape(dev)}">{escape(rig_short)}·{escape(dev)}</td>'

    def _badges(m):
        bs = []
        kind = MODEL_KIND.get(m)
        if kind == "cloning":
            bs.append('<span class="badge clone">clones</span>')
        elif kind == "predefined":
            bs.append('<span class="badge">preset</span>')
        if m in MODEL_MULTILINGUAL:
            bs.append('<span class="badge multi">multilingual</span>')
        return "".join(bs)

    # ---- VIEW 1: by prompt (compare every model on one sentence) ----
    def _prompt_section(title, clips_by_model, pid, mode, accent=False, ref_src=None):
        rows = [(m, clips_by_model[m][pid]) for m in _by_name(clips_by_model)
                if pid in clips_by_model[m]]
        if not rows:
            return ""
        b = [f'<div class="subsection{" cloning" if accent else ""}">',
             f'<div class="listen-group">{escape(title)} '
             f'<span class="muted">({len(rows)})</span></div>']
        if ref_src:
            b.append('<div class="ref-row"><span class="ref-label">▶ Reference voice — '
                     f'the target each clone imitates:</span>{_audio(ref_src)}</div>')
        b.append('<table><thead><tr><th>Model</th><th>Size</th>'
                 '<th>Source</th><th>Audio</th></tr></thead><tbody>')
        for (m, (src, rig_short, dev)) in rows:
            b.append('<tr>'
                     f'<td>{escape(_display_name(m))}{_issue_badge(m, mode)}</td>'
                     f'<td class="muted">{escape(MODEL_SIZE.get(m, "—"))}</td>'
                     f'{_src_cell(rig_short, dev)}'
                     f'<td>{_audio(src)}</td></tr>')
        b.append('</tbody></table></div>')
        return "".join(b)

    bp = ['<div id="view-by-prompt">']
    if len(prompts) > 3:
        bp.append('<nav class="prompt-jumper">Jump to: '
                  + " · ".join(f'<a href="#p{escape(pid)}">P{escape(pid)}</a>' for pid in prompts)
                  + '</nav>')
    for i, pid in enumerate(prompts):
        summ = f'Prompt {escape(pid)}'
        ptext = PROMPT_INFO.get(pid)
        if ptext:
            lang, text = ptext
            summ += f' <span class="summ-text">[{escape(lang)}] "{escape(text)}"</span>'
        bp.append(f'<details class="panel" id="p{escape(pid)}"{" open" if i == 0 else ""}>'
                  f'<summary>{summ}</summary>')
        bp.append(_prompt_section("Default voice", default_clips, pid, "default"))
        bp.append(_prompt_section("Cloning — chris_hemsworth", cloning_clips, pid, "cloning",
                                  accent=True, ref_src=cloning_ref))
        bp.append('</details>')
    bp.append('</div>')

    # ---- VIEW 2: by model (audition one model across prompts) ----
    def _model_section(title, pid_clips, model, mode, accent=False, ref_src=None):
        if not pid_clips:
            return ""
        b = [f'<div class="subsection{" cloning" if accent else ""}">',
             f'<div class="listen-group">{escape(title)}{_issue_badge(model, mode)}</div>']
        if ref_src:
            b.append(f'<div class="ref-row"><span class="ref-label">▶ Reference voice:</span>'
                     f'{_audio(ref_src)}</div>')
        b.append('<table><thead><tr><th>Prompt</th><th>Source</th>'
                 '<th>Audio</th></tr></thead><tbody>')
        for pid in prompts:
            if pid not in pid_clips:
                continue
            src, rig_short, dev = pid_clips[pid]
            # Hidden model name so the shared text filter matches by-model rows.
            b.append(f'<tr><td>P{escape(pid)}<span hidden> {escape(model)} '
                     f'{escape(_display_name(model))}</span></td>'
                     f'{_src_cell(rig_short, dev)}<td>{_audio(src)}</td></tr>')
        b.append('</tbody></table></div>')
        return "".join(b)

    bm = ['<div id="view-by-model" style="display:none">']
    for m in _by_name(set(default_clips) | set(cloning_clips)):
        rel = _release_label(m)
        meta = escape(MODEL_SIZE.get(m, "—")) + (f' · {escape(rel)}' if rel else "")
        bm.append(f'<details class="panel" id="m-{escape(m)}"><summary>{escape(_display_name(m))} '
                  f'<span class="summary-meta">{meta}</span> '
                  f'{_badges(m)}{_issue_badge(m, "default")}{_issue_badge(m, "cloning")}</summary>')
        bm.append(_model_section("Default voice", default_clips.get(m, {}), m, "default"))
        bm.append(_model_section("Cloning — chris_hemsworth", cloning_clips.get(m, {}), m, "cloning",
                                 accent=True, ref_src=cloning_ref))
        bm.append('</details>')
    bm.append('</div>')

    out = ['<!doctype html>',
           '<html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           '<title>tts-bench — Listen</title>',
           FAVICON_LINK, STYLE, LOGO_STYLE, _SUBSECTION_STYLE, _LISTEN_VIEW_STYLE,
           '</head><body>', _top_controls("listen"), LOGO_HEADER,
           '<h1>Listen</h1>', _LISTEN_GUIDE,
           '<div class="view-toggle">'
           '<button type="button" data-view="by-prompt" class="active">By prompt</button>'
           '<button type="button" data-view="by-model">By model</button></div>',
           "".join(bp), "".join(bm),
           SCRIPT, _LISTEN_SCRIPT, '</body></html>']
    (WORKTREE / "listen.html").write_text("\n".join(out), encoding="utf-8")


_SPEED_GUIDE = (
    '<div class="reading-guide"><strong>TTFA</strong> = time to first audio (lower is better). '
    '<strong>RTF</strong> = real-time factor (× realtime; higher is better). '
    '<strong>Cold</strong> = first run after load; <strong>warm</strong> = subsequent runs. '
    'Pick a rig below — each shows its default-voice and cloning runs. Tables default to '
    '<strong>warm RTF, fastest first</strong>; click any column header to re-sort. '
    'Audio is on the <a href="listen.html">Listen</a> page.</div>'
)

_SPEED_HUB_STYLE = (
    '<style>'
    '.rig-select{display:flex;align-items:center;gap:.7rem;flex-wrap:wrap;'
    'margin:1.4rem 0 1.6rem;}'
    '.rig-select .rig-select-label{color:var(--muted);font-size:.9em;}'
    '#rig-tabs{display:inline-flex;gap:.5rem;flex-wrap:wrap;}'
    '#rig-tabs .lens-tab{padding:7px 18px;}'
    '.rig-panel>.meta{margin:.2rem 0 1.4rem;padding:.7rem .95rem;background:var(--panel);'
    'border:1px solid var(--border);border-radius:8px;line-height:1.55;}'
    '.rig-panel .subsection{margin-top:.7rem;}'
    '.rig-panel .sub-head{margin:.2rem 0 .7rem;}'
    '.mode-select{display:flex;align-items:center;gap:.7rem;flex-wrap:wrap;'
    'margin:1.4rem 0 .5rem;}'
    '#mode-tabs{display:inline-flex;gap:.5rem;flex-wrap:wrap;}'
    '#mode-tabs .lens-tab{padding:7px 18px;}'
    '.mode-empty{color:var(--muted);}'
    '</style>'
)

_RIG_TAB_SCRIPT = '''<script>
(function(){
  var tabs = document.querySelectorAll('#rig-tabs .lens-tab');
  var panels = document.querySelectorAll('.rig-panel');
  function show(rig){
    panels.forEach(function(p){ p.style.display = (p.dataset.rig === rig) ? '' : 'none'; });
    tabs.forEach(function(t){ t.classList.toggle('active', t.dataset.rig === rig); });
  }
  tabs.forEach(function(t){
    t.addEventListener('click', function(e){ e.preventDefault(); show(t.dataset.rig); });
  });
})();
</script>'''

# Default/Cloning toggle. Composes with the rig toggle: a subsection is on screen
# only when its rig-panel is the active rig AND its data-mode is the active mode.
# This script governs data-mode visibility across every panel at once, so the
# already-correct subsection is showing the moment a rig becomes visible.
_MODE_TAB_SCRIPT = '''<script>
(function(){
  var tabs = document.querySelectorAll('#mode-tabs .lens-tab');
  var subs = document.querySelectorAll('.subsection[data-mode]');
  function show(mode){
    subs.forEach(function(s){ s.style.display = (s.dataset.mode === mode) ? '' : 'none'; });
    tabs.forEach(function(t){ t.classList.toggle('active', t.dataset.mode === mode); });
  }
  tabs.forEach(function(t){
    t.addEventListener('click', function(e){ e.preventDefault(); show(t.dataset.mode); });
  });
})();
</script>'''


def build_speed_hub():
    """Build speed.html — speed leaderboard with a Default/Cloning mode toggle and
    rig tabs. Mode is the primary axis: pick Cloning and the top row of the
    (RTF-sorted) table is the fastest cloning model on that rig. Each rig panel
    holds both mode subsections; the mode toggle shows one at a time across rigs."""
    rig_slugs, panels, rtf_idx = [], [], 5
    has_cloning = False  # only render the mode toggle if some rig actually has cloning data
    for (rig, dname, cname) in SPEED_RIGS:
        ddir, cdir = _canonical_dir(dname), _canonical_dir(cname)
        if not ddir and not cdir:
            continue
        hidden = "" if not rig_slugs else ' style="display:none"'
        block = [f'<section class="rig-panel" data-rig="{escape(rig)}"{hidden}>']
        block.append(f'<div class="meta"><strong>Rig:</strong> <code>{escape(rig)}</code> — '
                     f'{escape(_rig_summary(_read_meta(ddir or cdir)))}</div>')
        # One subsection per mode, tagged data-mode so the mode toggle can show
        # exactly one at a time. Default visible; cloning hidden until toggled.
        for (mkey, label, d) in (("default", "Default voice", ddir),
                                 ("cloning", "Cloning", cdir)):
            accent = " cloning" if mkey == "cloning" else ""
            vis = "" if mkey == "default" else ' style="display:none"'
            rows = None
            if d:
                try:
                    rows = _read_csv(d / "results.csv")
                except Exception:
                    rows = None
            if not rows:
                # Keep the slot so switching modes never lands on a blank page.
                block.append(f'<div class="subsection mode-empty{accent}" data-mode="{mkey}"{vis}>'
                             f'<h3 class="sub-head">{escape(label)} '
                             f'<span class="muted">· no runs on this rig</span></h3></div>')
                continue
            if mkey == "cloning":
                has_cloning = True
            ctx = _build_context(rows, d, _read_meta(d))
            table_html, rtf_idx = _speed_table_html(ctx)
            block.append(f'<div class="subsection{accent}" data-mode="{mkey}"{vis}>')
            block.append(f'<h3 class="sub-head">{escape(label)} '
                         f'<span class="muted">· {len(ctx["models_seen"])} models · '
                         f'sorted fastest first · '
                         f'<a href="{escape(d.name)}/index.html">full report ↗</a></span></h3>')
            block.append(table_html)
            block.append('</div>')
        block.append('</section>')
        rig_slugs.append(rig)
        panels.append("".join(block))

    out = ['<!doctype html>',
           '<html lang="en"><head><meta charset="utf-8">',
           '<title>tts-bench — Speed</title>',
           FAVICON_LINK, STYLE, LOGO_STYLE, _SUBSECTION_STYLE, _SPEED_HUB_STYLE,
           '</head><body>', _top_controls("speed"), LOGO_HEADER,
           '<h1>Speed</h1>',
           _SPEED_GUIDE]
    if has_cloning:
        out.append('<div class="mode-select"><span class="rig-select-label">Voice:</span>'
                   '<div class="lens-tabs" id="mode-tabs">'
                   '<a class="lens-tab active" data-mode="default" href="#">Default voice</a>'
                   '<a class="lens-tab" data-mode="cloning" href="#">Cloning</a>'
                   '</div></div>')
    out.append('<div class="rig-select"><span class="rig-select-label">Rig:</span>'
               '<div class="lens-tabs" id="rig-tabs">')
    for rig in rig_slugs:
        cls = "lens-tab active" if rig == rig_slugs[0] else "lens-tab"
        out.append(f'<a class="{cls}" data-rig="{escape(rig)}" href="#">{escape(rig)}</a>')
    out.append('</div></div>')
    out.extend(panels)
    out.append(f'<script>window.__defaultSort = {{colIdx: {rtf_idx}, dir: -1}};</script>')
    out.append(SCRIPT)
    out.append(_RIG_TAB_SCRIPT)
    if has_cloning:
        out.append(_MODE_TAB_SCRIPT)
    out.append('</body></html>')
    (WORKTREE / "speed.html").write_text("\n".join(out), encoding="utf-8")


_SCORES_GUIDE = (
    '<div class="reading-guide">Objective scores over the same 5 prompts. '
    '<strong>UTMOS</strong> = predicted naturalness (higher better); '
    '<strong>WER</strong> = ASR word-error rate vs the intended text — a '
    '<em>failure-detector</em>, not a fine ranking (lower better); '
    '<strong>SIM</strong> = speaker similarity to the cloned reference '
    '(<code>chris_hemsworth_15s</code>, higher better); '
    '<strong>Health</strong> = deterministic defect triage of the published clip '
    '(⚠ flags long internal silence / clipping / dead audio — a "go listen" cue, '
    'not a score). Switch '
    '<strong>Default</strong> / <strong>Cloning</strong> below; click any header to '
    're-sort. Each score is the mean over the exact clips shown on '
    '<a href="listen.html">Listen</a>. Human votes are the preference ground truth; '
    'these objective metrics are backstops.</div>'
)


def _health_cell(flags):
    """Render the Health column cell from [(prompt_id, flag), ...] over a model's
    picked clips. "" of flags ⇒ a muted ✓; otherwise a ⚠ badge naming the flag(s)
    and which prompt(s). data-sort = flag count so the column sorts clean-first."""
    if not flags:
        return '<td class="num muted" data-sort="0">✓</td>'
    by_flag = {}
    for pid, f in flags:
        by_flag.setdefault(f, []).append(pid)
    label = "; ".join(f'{f} ({",".join(sorted(set(pids)))})'
                      for f, pids in sorted(by_flag.items()))
    return (f'<td class="num" data-sort="{len(flags)}">'
            f'<span class="health-flag" title="mechanical defect in the published clip '
            f'(long internal silence / clipping / dead audio) — listen before trusting">'
            f'⚠ {escape(label)}</span></td>')


def _scores_table(models, dirs, look, columns, fallback_dirs=None, fallback_models=frozenset(),
                  show_health=True):
    """One sortable table. columns = list of (metric_key, header, higher_better).
    Rows aggregated per model; WER>threshold rows get the 'flagged' class.
    show_health appends a non-ranking Health triage column (clip/silent/gap).
    fallback_dirs/fallback_models: for NO_PRESET_VOICE models on the cloning board,
    append default dirs so a model benched only in default mode still appears."""
    # Numeric headers get class="num" so they right-align over their right-aligned
    # number cells (td.num); a plain <th> is left-aligned and visibly drifts off its
    # column on wide screens where each column is stretched.
    head = '<th>Model</th><th>Size</th><th>Released</th>' + "".join(
        f'<th class="num">{escape(h)} {"↑" if up else "↓"}</th>' for (_k, h, up) in columns)
    if show_health:
        head += '<th class="num">Health</th>'
    body = []
    aggs = []
    for m in sorted(models, key=lambda x: _display_name(x).lower()):
        eff_dirs = dirs
        if fallback_dirs and m in fallback_models:
            eff_dirs = tuple(dirs) + tuple(fallback_dirs)
        agg = _model_scores(m, _all_prompt_ids(eff_dirs), eff_dirs, look)
        if agg["n"] == 0:
            continue
        aggs.append((m, agg))
    for (m, agg) in aggs:
        flagged = agg.get("wer") is not None and agg["wer"] > WER_FAIL_THRESHOLD
        cls = ' class="flagged"' if flagged else ""
        cells = [f'<td>{escape(_display_name(m))}</td>',
                 f'<td class="muted">{escape(MODEL_SIZE.get(m, "—"))}</td>',
                 _release_td(m)]
        for (k, _h, _up) in columns:
            v = agg.get(k)
            if v is None:
                cells.append('<td class="num muted" data-sort="">—</td>')
            else:
                cells.append(f'<td class="num" data-sort="{v:.4f}">{v:.3f}</td>')
        if show_health:
            cells.append(_health_cell(agg.get("health_flags", [])))
        body.append(f'<tr{cls}>' + "".join(cells) + '</tr>')
    if not body:
        return '<p class="mode-empty muted">No scored models yet.</p>'
    return (f'<table><thead><tr>{head}</tr></thead><tbody>'
            + "".join(body) + '</tbody></table>')


def build_scores():
    """Build scores.html — objective metric leaderboard with a Default/Cloning
    toggle. Default = UTMOS+WER; Cloning = SIM+UTMOS+WER. Reuses _pick_clip so each
    score matches the clip shown on Listen."""
    look = _read_scores_csv()
    raw_default = set().union(*(_ok_models(n) for n in LISTEN_DEFAULT_DIRS)) if LISTEN_DEFAULT_DIRS else set()
    raw_cloning = set().union(*(_ok_models(n) for n in LISTEN_CLONING_DIRS)) if LISTEN_CLONING_DIRS else set()
    default_models = raw_default - NO_PRESET_VOICE
    cloning_models = raw_cloning | (NO_PRESET_VOICE & (raw_default | raw_cloning))

    default_cols = [("utmos", "UTMOS", True), ("wer", "WER", False)]
    cloning_cols = [("sim", "SIM", True), ("utmos", "UTMOS", True), ("wer", "WER", False)]

    default_tbl = _scores_table(default_models, LISTEN_DEFAULT_DIRS, look, default_cols)
    # Cloning board: like build_listen, a NO_PRESET_VOICE model benched only in
    # default mode cloned the reference there, so fall back to its default-dir clip
    # (SIM will be blank for it since that clip was scored in default mode).
    cloning_tbl = _scores_table(cloning_models, LISTEN_CLONING_DIRS, look, cloning_cols,
                                fallback_dirs=LISTEN_DEFAULT_DIRS, fallback_models=NO_PRESET_VOICE)

    flagged_style = (
        '<style>tr.flagged>td{opacity:.55;}'
        'tr.flagged>td:first-child::after{content:" \\26A0";color:var(--fail);}'
        '.health-flag{color:var(--fail);white-space:nowrap;}'
        '.scores-foot{color:var(--muted);font-size:.85em;margin-top:1.4rem;'
        'border-top:1px solid var(--border);padding-top:.8rem;line-height:1.5;}'
        '.scores-foot a{color:var(--accent);}</style>')
    foot = (
        '<div class="scores-foot">Scored over the 5 bench prompts (thin — WER is a '
        'failure-detector, not a fine ranking). Checkpoints: UTMOS '
        '<code>utmos22_strong</code> (SpeechMOS), SIM canonical UniSpeech-SAT '
        '<code>wavlm_large_finetune</code>, WER Whisper-large-v3. Method follows '
        '<a href="https://github.com/BytedanceSpeech/seed-tts-eval">seed-tts-eval</a>. '
        'Human votes are the preference ground truth; these are objective backstops.</div>')

    out = ['<!doctype html>',
           '<html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           '<title>tts-bench — Scores</title>',
           FAVICON_LINK, STYLE, LOGO_STYLE, _SUBSECTION_STYLE, _SPEED_HUB_STYLE,
           flagged_style,
           '</head><body>', _top_controls("scores"), LOGO_HEADER,
           '<h1>Scores</h1>', _SCORES_GUIDE,
           '<div class="mode-select"><span class="rig-select-label">Voice:</span>'
           '<div class="lens-tabs" id="mode-tabs">'
           '<a class="lens-tab active" data-mode="default" href="#">Default voice</a>'
           '<a class="lens-tab" data-mode="cloning" href="#">Cloning</a>'
           '</div></div>',
           f'<div class="subsection" data-mode="default"><h3 class="sub-head">Default voice '
           f'<span class="muted">· naturalness + intelligibility</span></h3>{default_tbl}</div>',
           f'<div class="subsection cloning" data-mode="cloning" style="display:none">'
           f'<h3 class="sub-head">Cloning <span class="muted">· fidelity + naturalness + '
           f'intelligibility</span></h3>{cloning_tbl}</div>',
           foot,
           # colIdx 3 = first metric column (after Model/Size/Released): UTMOS on
           # the default board, SIM on the cloning board — keep the metric-led sort.
           '<script>window.__defaultSort = {colIdx: 3, dir: -1};</script>',
           SCRIPT, _MODE_TAB_SCRIPT, '</body></html>']
    (WORKTREE / "scores.html").write_text("\n".join(out), encoding="utf-8")


def _params_num(m):
    """Rough param/size magnitude from the MODEL_SIZE cell, for a sortable Params
    column. Best-effort: "1.5B"->1.5e9, "<100M"/"~500M"->Ne6, "~25MB"->disk proxy."""
    s = MODEL_SIZE.get(m, "").lstrip("~<").strip()
    num = ""
    for ch in s:
        if ch.isdigit() or ch == ".":
            num += ch
        else:
            break
    if not num:
        return None
    val = float(num)
    unit = s[len(num):].strip().upper()
    if unit.startswith("B"):
        return val * 1e9
    return val * 1e6  # "M" params or "MB" disk — both small relative to B


def _langs_num(m):
    """Sort key for the Languages column: 0 = English-only, else the language
    count (parsed from "(31)"/"(600+)"), token count for "(zh+en)", or 1 for a
    bare ✓ with unstated count."""
    cell = MODEL_LANGS.get(m, "")
    if not cell.startswith("✓"):
        return 0
    inside = cell[cell.find("(") + 1:cell.find(")")] if "(" in cell else ""
    digits = "".join(c for c in inside if c.isdigit())
    if digits:
        return int(digits)
    if "+" in inside:
        return inside.count("+") + 1
    return 1


_CAPS_GUIDE = (
    '<div class="reading-guide">Every tracked model and what it can do. The '
    'capability toggles <strong>combine</strong> — check two and you see only models '
    'with both; the search box filters by name; click any column header to sort. '
    '<strong>Cross-lingual clone</strong> = clones a voice from a reference in one '
    'language and speaks a <em>different</em> one in that same voice (verified '
    'per-model from each model card/paper). The License column shows the exact '
    'license; the <em>commercial</em> toggle is a coarse "no NC / research-only '
    'clause" filter — check the license before relying on it.</div>'
)

_CAPS_STYLE = (
    '<style>'
    '.cap-filters{display:flex;flex-wrap:wrap;gap:.5rem;margin:1rem 0 1.4rem;}'
    '.cap-chip{display:inline-flex;align-items:center;gap:.4rem;background:var(--input-bg);'
    'border:1px solid var(--input-border);border-radius:999px;padding:5px 13px;'
    'font-size:.9em;cursor:pointer;user-select:none;}'
    '.cap-chip input{accent-color:var(--accent);cursor:pointer;margin:0;}'
    '.cap-chip:has(input:checked){border-color:var(--accent);color:var(--accent);'
    'background:color-mix(in srgb,var(--accent) 12%,transparent);}'
    '.cap-yes{color:var(--accent);font-weight:600;}'
    'td.cap-type{font-variant:small-caps;letter-spacing:.03em;}'
    '#caps-table td a{color:var(--text);text-decoration:none;border-bottom:1px dotted var(--muted);}'
    '#caps-table td a:hover{color:var(--accent);}'
    '</style>'
)

_CAPS_SCRIPT = '''<script>
(function(){
  var chips = Array.prototype.slice.call(document.querySelectorAll('.cap-chip input'));
  var nameInput = document.getElementById('filter');
  var rows = Array.prototype.slice.call(document.querySelectorAll('#caps-table tbody tr'));
  function apply(){
    var q = ((nameInput && nameInput.value) || '').toLowerCase().trim();
    var active = chips.filter(function(c){ return c.checked; })
                      .map(function(c){ return c.getAttribute('data-cap'); });
    rows.forEach(function(r){
      var nameHit = !q || r.textContent.toLowerCase().indexOf(q) !== -1;
      var capHit = active.every(function(cap){ return r.getAttribute('data-' + cap) === '1'; });
      r.style.display = (nameHit && capHit) ? '' : 'none';
    });
  }
  // Registered AFTER the shared SCRIPT, so on a name-filter input this runs last
  // and is the final authority on row visibility (name AND active chips).
  chips.forEach(function(c){ c.addEventListener('change', apply); });
  if(nameInput) nameInput.addEventListener('input', apply);
  var reset = document.getElementById('reset-sort');
  if(reset) reset.addEventListener('click', function(){
    chips.forEach(function(c){ c.checked = false; });
    setTimeout(apply, 0);
  });
  apply();
})();
</script>'''


def build_capabilities():
    """Build capabilities.html — one sortable capability matrix over every tracked
    model, with toggle-chip filters (clone/preset/multilingual/cross-lingual/
    expressive/commercial/CPU) that AND together on top of the shared name filter."""
    raw_default = set().union(*(_ok_models(n) for n in LISTEN_DEFAULT_DIRS)) if LISTEN_DEFAULT_DIRS else set()
    raw_cloning = set().union(*(_ok_models(n) for n in LISTEN_CLONING_DIRS)) if LISTEN_CLONING_DIRS else set()
    models = sorted(raw_default | raw_cloning, key=lambda m: _display_name(m).lower())

    def yn(b):
        return '<span class="cap-yes">✓</span>' if b else '<span class="muted">—</span>'

    def _row(m):
        clone = MODEL_KIND.get(m) == "cloning"
        preset = MODEL_KIND.get(m) == "predefined" or m in _PRESET_AND_CLONE
        multi = _is_multilingual(m)
        xling = _is_crosslingual(m)
        expr = MODEL_EXPRESSIVE.get(m, "—")
        has_expr = expr not in ("", "—")
        comm = _is_commercial(m)
        cpu = m in _CPU_OK
        typ = "both" if (clone and preset) else ("clone" if clone else "preset")
        url = MODEL_URL.get(m)
        name = escape(_display_name(m))
        name_cell = (f'<a href="{escape(url)}" target="_blank" rel="noopener">{name}</a>'
                     if url else name)
        # Cross-lingual is only meaningful for cloners; preset models show n/a and
        # sort to the bottom (data-sort -1 < the 0/1 cloners use).
        xling_cell = (f'<td class="num" data-sort="{1 if xling else 0}">{yn(xling)}</td>'
                      if clone else '<td class="num muted" data-sort="-1">n/a</td>')
        return (
            f'<tr data-clone="{1 if clone else 0}" data-preset="{1 if preset else 0}" '
            f'data-multi="{1 if multi else 0}" data-xling="{1 if xling else 0}" '
            f'data-expr="{1 if has_expr else 0}" data-comm="{1 if comm else 0}" '
            f'data-cpu="{1 if cpu else 0}">'
            f'<td>{name_cell}</td>'
            f'<td class="num"{_ds(_params_num(m))}>{escape(MODEL_SIZE.get(m, "—"))}</td>'
            f'{_release_td(m)}'
            f'<td class="cap-type">{typ}</td>'
            f'<td data-sort="{_langs_num(m)}">{escape(MODEL_LANGS.get(m, "—"))}</td>'
            f'{xling_cell}'
            f'<td data-sort="{1 if has_expr else 0}">{escape(expr)}</td>'
            f'<td class="num"{_ds(_sr_hz(m))}>{escape(MODEL_SR.get(m, "—"))}</td>'
            f'<td>{escape(MODEL_LICENSE.get(m, "—"))}</td>'
            f'<td class="num" data-sort="{1 if cpu else 0}">{yn(cpu)}</td>'
            '</tr>'
        )

    head = ('<th>Model</th><th class="num">Params</th><th>Released</th><th>Type</th>'
            '<th>Languages</th><th class="num">Cross-lingual</th><th>Expressive</th>'
            '<th class="num">Sample rate</th><th>License</th><th class="num">CPU</th>')
    table = (f'<table id="caps-table"><thead><tr>{head}</tr></thead><tbody>'
             + "".join(_row(m) for m in models) + '</tbody></table>')

    chips = [("clone", "clones"), ("preset", "presets"), ("multi", "multilingual"),
             ("xling", "cross-lingual clone"), ("expr", "expressive"),
             ("comm", "commercial license"), ("cpu", "runs on CPU")]
    chip_html = "".join(
        f'<label class="cap-chip"><input type="checkbox" data-cap="{c}"> {l}</label>'
        for c, l in chips)

    out = ['<!doctype html>', '<html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           '<title>tts-bench — Capabilities</title>',
           FAVICON_LINK, STYLE, LOGO_STYLE, _CAPS_STYLE,
           '</head><body>', _top_controls("capabilities"), LOGO_HEADER,
           f'<h1>Capabilities <span class="muted" style="font-size:.5em;font-weight:400;">'
           f'· {len(models)} models</span></h1>',
           _CAPS_GUIDE,
           f'<div class="cap-filters">{chip_html}</div>',
           table, SCRIPT, _CAPS_SCRIPT, '</body></html>']
    (WORKTREE / "capabilities.html").write_text("\n".join(out), encoding="utf-8")


_HOME_STYLE = (
    '<style>'
    '.home-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:1.2rem;margin:1.8rem 0;}'
    '@media(max-width:680px){.home-grid{grid-template-columns:1fr;}}'
    '.home-card{background:var(--panel);border:1px solid var(--border);border-radius:14px;'
    'transition:border-color .15s,transform .15s;}'
    '.home-card:hover{border-color:var(--accent);transform:translateY(-2px);}'
    '.home-card a{display:block;text-decoration:none;color:var(--text);padding:1.8rem 1.6rem;}'
    '.home-card h2{color:var(--accent);font-size:1.5em;margin:0 0 .5rem;}'
    '.home-card p{color:var(--muted);margin:0;font-size:.95em;line-height:1.5;}'
    '.home-foot{color:var(--muted);font-size:.9em;margin-top:1.6rem;'
    'border-top:1px solid var(--border);padding-top:1rem;}'
    '.home-foot a{color:var(--accent);}'
    '.home-controls{display:flex;justify-content:flex-end;padding:.6rem 0;}'
    '.home-controls button{background:var(--input-bg);color:var(--text);'
    'border:1px solid var(--input-border);border-radius:6px;padding:6px 12px;'
    'font:inherit;cursor:pointer;}'
    '</style>'
)

_THEME_ONLY_SCRIPT = '''<script>
(function(){
  var b = document.getElementById('theme-toggle');
  function apply(n){ document.documentElement.setAttribute('data-theme', n);
    b.textContent = n === 'light' ? '☀ light' : '☾ dark';
    try { localStorage.setItem('tts-bench-theme', n); } catch (e) {} }
  var s = null; try { s = localStorage.getItem('tts-bench-theme'); } catch (e) {}
  apply(s === 'light' ? 'light' : 'dark');
  b.addEventListener('click', function(){
    var c = document.documentElement.getAttribute('data-theme') || 'dark';
    apply(c === 'dark' ? 'light' : 'dark'); });
})();
</script>'''


def build_landing():
    """Build index.html — the landing cards (Listen / Speed / Scores / Vote) + archive/GitHub footer."""
    _copy_branding_assets()
    all_dirs = [n for (_r, dn, cn) in SPEED_RIGS for n in (dn, cn)]
    n_models = len(set().union(*(_ok_models(n) for n in all_dirs))) if all_dirs else 0
    n_rigs = sum(1 for (_r, dn, cn) in SPEED_RIGS if _canonical_dir(dn) or _canonical_dir(cn))

    out = ['<!doctype html>',
           '<html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           '<title>tts-bench — local TTS benchmark</title>',
           FAVICON_LINK, STYLE, LOGO_STYLE, _HOME_STYLE,
           '</head><body>',
           '<div class="home-controls"><button type="button" id="theme-toggle">☾ dark</button></div>',
           LOGO_HEADER,
           '<div class="meta">Open-source TTS models benchmarked side-by-side. '
           'Three lenses plus a blind community vote — pick one:</div>',
           '<div class="home-grid">',
           '<div class="home-card"><a href="listen.html"><h2>▶ Listen</h2>'
           '<p>Hear every model on the same prompts — default voice and voice cloning, '
           'side by side. One clip per model, sourced from the highest-fidelity rig. '
           'Quality and prosody are obvious in seconds.</p></a></div>',
           '<div class="home-card"><a href="speed.html"><h2>▶ Speed</h2>'
           '<p>Time-to-first-audio, real-time factor, and memory for every model — '
           'per rig (windows-5090 · linux-3090 · mac-m4), cold and warm. Sortable.</p></a></div>',
           '<div class="home-card"><a href="scores.html"><h2>▶ Scores</h2>'
           '<p>Objective metrics for every model — UTMOS naturalness, WER '
           'intelligibility, and cloning-fidelity SIM. Sortable, default voice '
           'and cloning. Human votes remain the ground truth; these are backstops.</p></a></div>',
           '<div class="home-card"><a href="capabilities.html"><h2>▶ Capabilities</h2>'
           '<p>What each model can actually do — voice cloning, cross-lingual cloning, '
           'languages, expressive control, sample rate, license, CPU support. Toggle '
           'filters that combine, plus sort by any column.</p></a></div>',
           '<div class="home-card"><a href="https://5uck1ess-tts-arena.hf.space" '
           'target="_blank" rel="noopener"><h2>🗳 Vote ↗</h2>'
           '<p>Hear two clips blind and pick the better one — default voice or voice '
           'cloning. No install, ~5 seconds a vote, feeding a live human-preference '
           'Elo leaderboard. This is the ground truth the Scores lens is measured against.</p></a></div>',
           '</div>',
           f'<div class="home-foot">{n_models} models · {n_rigs} rigs · '
           '<a href="archive.html">Archive — full per-rig reports</a> · '
           '<a href="https://github.com/5uck1ess/tts-bench">GitHub →</a></div>',
           _THEME_ONLY_SCRIPT,
           '</body></html>']
    (WORKTREE / "index.html").write_text("\n".join(out), encoding="utf-8")


def build_archive_index():
    """Build archive.html — the full per-rig report table (deep speed/samples pages
    for each canonical run). Linked from the landing footer; preserves every run."""
    runs = []
    for d in sorted(WORKTREE.iterdir(), reverse=True):
        if not d.is_dir() or d.name.startswith("."):
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
           '<title>tts-bench — Archive</title>',
           FAVICON_LINK,
           STYLE,
           LOGO_STYLE,
           '</head><body>',
           CONTROLS,
           LOGO_HEADER,
           '<div class="nav"><a href="index.html">← home</a></div>',
           '<h1>Archive — per-rig reports</h1>',
           '<div class="meta">The full per-rig run reports behind the consolidated '
           '<a href="listen.html">Listen</a> / <a href="speed.html">Speed</a> pages. '
           'Each row is one rig × voice-mode, with its own speed table and by-prompt '
           'samples gallery.</div>',
           f'<div class="meta">{len(runs)} run(s)</div>',
           '<table><thead><tr>']
    for col in ("Run", "Label", "Rig", "Models", "Devices", "Prompts", "Rows", "OK", "Report"):
        out.append(f'<th>{col}</th>')
    out.append('</tr></thead><tbody>')

    for r in runs:
        models = (", ".join(r["models"])
                  if len(r["models"]) <= 5
                  else f"{len(r['models'])} models")
        rig_cell = (f'<code title="{escape(r["rig_full"])}">{escape(r["rig"])}</code>'
                    if r["rig"] else '<span class="muted">—</span>')
        label_cell = (escape(r["label"])
                      if r["label"]
                      else '<span class="muted">—</span>')
        out.append('<tr>')
        if r.get("has_index"):
            out.append(f"<td><a href='{escape(r['name'])}/index.html'>{escape(r['name'])}</a></td>")
        else:
            out.append(f"<td><a href='{escape(r['name'])}/report.html'>{escape(r['name'])}</a></td>")
        out.append(f"<td>{label_cell}</td>")
        out.append(f"<td>{rig_cell}</td>")
        out.append(f"<td{_ds(len(r['models']))}>{escape(models)}</td>")
        out.append(f"<td>{escape(', '.join(r['devices']))}</td>")
        out.append(f"<td class='num'{_ds(len(r['prompts']))}>{len(r['prompts'])}</td>")
        out.append(f"<td class='num'{_ds(r['rows'])}>{r['rows']}</td>")
        out.append(f"<td class='num'{_ds(r['ok'])}>{r['ok']}/{r['rows']}</td>")
        if r.get("has_index"):
            parts = [f'<a href="{escape(r["name"])}/speed.html">speed</a>']
            parts.append(f'<a href="{escape(r["name"])}/samples.html">samples</a>')
            out.append(f'<td>{" · ".join(parts)}</td>')
        else:
            out.append(f'<td><a href="{escape(r["name"])}/report.html">view (legacy)</a></td>')
        out.append('</tr>')

    out.append('</tbody></table>')
    out.append(SCRIPT)
    out.append('</body></html>')

    (WORKTREE / "archive.html").write_text("\n".join(out), encoding="utf-8")


def build_top_level():
    """Regenerate every top-level landing page from the canonical dirs in the worktree."""
    _copy_branding_assets()
    build_listen()
    build_speed_hub()
    build_scores()
    build_capabilities()
    build_archive_index()
    build_landing()  # last: counts depend on the dirs, not the other pages


def _pages_url():
    try:
        remote = _git("config", "--get", "remote.origin.url")
    except SystemExit:
        return None
    s = remote.rstrip("/").removesuffix(".git").replace("git@github.com:", "github.com/")
    if s.startswith("https://"):
        s = s[len("https://"):]
    parts = s.split("/")
    if len(parts) < 3 or "github.com" not in parts[0]:
        return None
    return f"https://{parts[1]}.github.io/{parts[2]}/"


def list_published():
    if not WORKTREE.exists():
        print("No _gh-pages/ worktree yet — nothing has been published.")
        return
    runs = sorted(
        [d.name for d in WORKTREE.iterdir()
         if d.is_dir() and not d.name.startswith(".") and (d / "index.html").exists()],
        reverse=True,
    )
    if not runs:
        print("No published runs.")
        return
    print(f"{len(runs)} published run(s):")
    for name in runs:
        print(f"  {name}")
    url = _pages_url()
    if url:
        print(f"\nIndex (once Pages is enabled): {url}")


def _publish_csv(src: Path, dest_dir: Path) -> None:
    """Copy results.csv to the published dir, always dropping any NAQ columns.
    NAQ is private lab-only R&D (possible paper), so the public CSV must never
    carry a 'naq*' column. New runs already have none, but older canonical CSVs
    still do — so the scrub stays here at the copy boundary as a safety net.
    The local results.csv is left intact."""
    with src.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        shutil.copy2(src, dest_dir / src.name)
        return
    keep = [i for i, col in enumerate(rows[0]) if "naq" not in col.lower()]
    with (dest_dir / src.name).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow([row[i] for i in keep if i < len(row)])


def publish(run_dir: Path, no_push: bool = False) -> None:
    if not run_dir.exists():
        raise SystemExit(f"Not found: {run_dir}")
    if not (run_dir / "results.csv").exists():
        raise SystemExit(f"No results.csv in {run_dir}")

    meta_src = run_dir / "meta.json"

    # Cloning runs: stage the reference clip into the run dir as `_reference.wav`
    # BEFORE building, so build_report's samples page can embed a "voice we
    # cloned" player (its render checks run_dir/_reference.wav). meta.json stores
    # only the basename (bench.py: Path(reference).name) and clips live in
    # reference/, so check there too — not just REPO.
    if meta_src.exists():
        try:
            meta_data = json.loads(meta_src.read_text(encoding="utf-8"))
        except Exception:
            meta_data = {}
        ref_rel = meta_data.get("reference")
        if ref_rel:
            p = Path(ref_rel)
            candidates = ([p] if p.is_absolute()
                          else [REPO / ref_rel, REPO / "reference" / p.name])
            ref_path = next((c for c in candidates if c.exists()), None)
            if ref_path:
                shutil.copy2(ref_path, run_dir / "_reference.wav")

    # Always (re)build so the published HTML reflects the current CSV, reference,
    # and report.py — never ship a stale index/speed/samples page.
    build_report(run_dir)

    ensure_worktree()

    dest = WORKTREE / run_dir.name
    if dest.exists():
        print(f"Overwriting existing {dest.relative_to(REPO)}")
        shutil.rmtree(dest)
    dest.mkdir()

    html_files = ("index.html", "speed.html", "samples.html",
                  "report.html", "results.csv")
    for fname in html_files:
        src = run_dir / fname
        if src.exists():
            if fname == "results.csv":
                _publish_csv(src, dest)
            else:
                shutil.copy2(src, dest)
    if meta_src.exists():
        shutil.copy2(meta_src, dest)
    else:
        print(f"  warning: no meta.json in {run_dir.name} — "
              f"run `python bench.py --write-meta {run_dir.relative_to(REPO)}` "
              f"to tag it with this machine's rig info.")
    wavs = list(run_dir.glob("*.wav"))
    for wav in wavs:
        shutil.copy2(wav, dest)

    total_bytes = sum(p.stat().st_size for p in dest.iterdir())
    has_meta = " + meta.json" if meta_src.exists() else ""
    print(f"Copied report.html + results.csv{has_meta} + {len(wavs)} wav(s) "
          f"({total_bytes / 1024 / 1024:.1f} MB)")

    (WORKTREE / ".nojekyll").touch()
    build_top_level()
    print(f"Rebuilt landing (index/listen/speed/archive) — {_count_published()} run(s) total")

    _git("add", "-A", cwd=WORKTREE)
    if not _git("diff", "--cached", "--name-only", cwd=WORKTREE):
        print("No changes — already up to date.")
        return
    _git("commit", "-m", f"Publish bench run {run_dir.name}", cwd=WORKTREE)
    print(f"Committed to {BRANCH}.")

    if no_push:
        print(f"\n--no-push: skipping push. Push manually with:\n  "
              f"git -C {WORKTREE} push -u origin {BRANCH}")
        return

    if _branch_exists_remote(BRANCH):
        _git("push", "origin", BRANCH, cwd=WORKTREE, capture=False)
    else:
        _git("push", "-u", "origin", BRANCH, cwd=WORKTREE, capture=False)
        print("\nFirst push of gh-pages — now enable Pages in repo settings:")
        print("  Settings → Pages → Source: 'Deploy from a branch' → Branch: gh-pages / root")

    url = _pages_url()
    if url:
        print(f"\nPublished. After Pages finishes deploying (~30s):")
        print(f"  Index:  {url}")
        print(f"  Report: {url}{run_dir.name}/report.html")


def rebuild_all(no_push: bool = False) -> None:
    """Regenerate every results/<dated>/ via build_report, then republish each to gh-pages."""
    if not (REPO / "results").exists():
        raise SystemExit("No results/ dir; nothing to rebuild.")
    ensure_worktree()
    dirs = sorted(
        [d for d in (REPO / "results").iterdir()
         if d.is_dir() and (d / "results.csv").exists()],
        reverse=True,
    )
    if not dirs:
        raise SystemExit("No results/<dated>/ subdirs with results.csv.")
    for d in dirs:
        print(f"Rebuilding {d.name}...")
        build_report(d)
        dest = WORKTREE / d.name
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir()
        for fname in ("index.html", "speed.html", "samples.html",
                      "report.html", "results.csv"):
            src = d / fname
            if src.exists():
                if fname == "results.csv":
                    _publish_csv(src, dest)
                else:
                    shutil.copy2(src, dest)
        meta_src = d / "meta.json"
        if meta_src.exists():
            shutil.copy2(meta_src, dest)
        for wav in d.glob("*.wav"):
            shutil.copy2(wav, dest)
    (WORKTREE / ".nojekyll").touch()
    build_top_level()
    print(f"Rebuilt {len(dirs)} run(s); landing updated.")

    _git("add", "-A", cwd=WORKTREE)
    if not _git("diff", "--cached", "--name-only", cwd=WORKTREE):
        print("No changes — already up to date.")
        return
    _git("commit", "-m", "Rebuild gh-pages with three-lens layout", cwd=WORKTREE)
    print("Committed to gh-pages.")
    if no_push:
        print(f"--no-push: skipping push. Push manually with:\n  "
              f"git -C {WORKTREE} push origin {BRANCH}")
        return
    if _branch_exists_remote(BRANCH):
        _git("push", "origin", BRANCH, cwd=WORKTREE, capture=False)
    else:
        _git("push", "-u", "origin", BRANCH, cwd=WORKTREE, capture=False)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Publish a bench run to gh-pages for GitHub Pages hosting.")
    p.add_argument("run_dir", nargs="?", help="results/<date> dir to publish.")
    p.add_argument("--no-push", action="store_true",
                   help="Commit to gh-pages but don't push to origin.")
    p.add_argument("--list", action="store_true",
                   help="List runs already published to gh-pages and exit.")
    p.add_argument("--rebuild-all", action="store_true",
                   help="Regenerate every results/<dated>/ and republish to gh-pages.")
    p.add_argument("--rebuild-pages", action="store_true",
                   help="Regenerate only the top-level landing pages (index/listen/speed/"
                        "archive) from the canonical dirs already in the worktree. "
                        "No new run copied, no commit, no push — for local preview.")
    args = p.parse_args()

    if args.rebuild_pages:
        ensure_worktree()
        build_top_level()
        print(f"Rebuilt landing pages in {WORKTREE.relative_to(REPO)}: "
              f"index.html · listen.html · speed.html · archive.html (not committed).")
        return 0

    if args.rebuild_all:
        rebuild_all(no_push=args.no_push)
        return 0

    if args.list:
        list_published()
        return 0

    if not args.run_dir:
        p.error("Provide a run dir (e.g. results/2026-05-23_2203) or --list.")

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO / args.run_dir
    publish(run_dir, no_push=args.no_push)
    return 0


if __name__ == "__main__":
    sys.exit(main())
