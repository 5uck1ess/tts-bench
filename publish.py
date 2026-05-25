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
    STYLE, CONTROLS, SCRIPT, SHOW_NAQ_PUBLIC,
    _ds, _read_csv, _read_meta, _rig_summary, _sort_prompt_ids, build_report,
)

REPO = Path(__file__).parent
WORKTREE = REPO / "_gh-pages"
BRANCH = "gh-pages"


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
    '<img class="site-logo site-logo--dark" src="logo-flat-dark.svg" '
    'alt="tts-bench" width="320" height="auto">'
    '<img class="site-logo site-logo--light" src="logo-flat-light.svg" '
    'alt="tts-bench" width="320" height="auto">'
    '</header>'
)

LOGO_STYLE = (
    '<style>'
    '.site-header{margin:1rem 0 1.25rem;}'
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


def build_pages_index():
    """Build _gh-pages/index.html listing all published runs."""
    _copy_branding_assets()
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
           '<title>TTS Bench — Published Runs</title>',
           FAVICON_LINK,
           STYLE,
           LOGO_STYLE,
           '</head><body>',
           CONTROLS,
           LOGO_HEADER,
           '<h1>Published Runs</h1>',
           ('<div class="meta">Open-source TTS models benchmarked side-by-side. Three axes: '
            '<strong>speed</strong> (TTFA, RTF), <strong>quality</strong> (NAQ), '
            '<strong>voice cloning</strong>. Each run has inline audio so you can listen '
            'to every model × prompt without downloading. '
            '<a href="https://github.com/5uck1ess/tts-bench">Repo on GitHub →</a></div>'
            if SHOW_NAQ_PUBLIC else
            '<div class="meta">Open-source TTS models benchmarked side-by-side: '
            '<strong>speed</strong> (TTFA, RTF) and <strong>voice cloning</strong>. '
            'Each run has inline audio so you can listen to every model × prompt '
            'without downloading. '
            '<a href="https://github.com/5uck1ess/tts-bench">Repo on GitHub →</a></div>'),
           f'<div class="meta">{len(runs)} published run(s)</div>',
           '<table><thead><tr>']
    for col in ("Date", "Label", "Rig", "Models", "Devices", "Prompts", "Rows", "OK", "Report"):
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
            if SHOW_NAQ_PUBLIC:
                parts.append(f'<a href="{escape(r["name"])}/quality.html">quality</a>')
            parts.append(f'<a href="{escape(r["name"])}/samples.html">samples</a>')
            out.append(f'<td>{" · ".join(parts)}</td>')
        else:
            out.append(f'<td><a href="{escape(r["name"])}/report.html">view (legacy)</a></td>')
        out.append('</tr>')

    out.append('</tbody></table>')
    out.append(SCRIPT)
    out.append('</body></html>')

    (WORKTREE / "index.html").write_text("\n".join(out), encoding="utf-8")


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


def publish(run_dir: Path, no_push: bool = False) -> None:
    if not run_dir.exists():
        raise SystemExit(f"Not found: {run_dir}")
    if not (run_dir / "results.csv").exists():
        raise SystemExit(f"No results.csv in {run_dir}")

    if not (run_dir / "index.html").exists():
        print(f"No index.html in {run_dir.name} — building it from CSV...")
        build_report(run_dir)

    ensure_worktree()

    dest = WORKTREE / run_dir.name
    if dest.exists():
        print(f"Overwriting existing {dest.relative_to(REPO)}")
        shutil.rmtree(dest)
    dest.mkdir()

    html_files = ("index.html", "speed.html", "quality.html", "samples.html",
                  "report.html", "results.csv")
    for fname in html_files:
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, dest)
    meta_src = run_dir / "meta.json"
    if meta_src.exists():
        shutil.copy2(meta_src, dest)
    else:
        print(f"  warning: no meta.json in {run_dir.name} — "
              f"run `python bench.py --write-meta {run_dir.relative_to(REPO)}` "
              f"to tag it with this machine's rig info.")
    wavs = list(run_dir.glob("*.wav"))
    for wav in wavs:
        shutil.copy2(wav, dest)

    # Cloning runs: copy the reference wav into the slug as `_reference.wav` so
    # the samples page can include a "this is the voice we cloned" player at the
    # top. meta.json stores the path the bench was invoked with, which is repo-
    # relative; resolve from REPO if not absolute.
    if meta_src.exists():
        try:
            meta_data = json.loads(meta_src.read_text(encoding="utf-8"))
        except Exception:
            meta_data = {}
        ref_rel = meta_data.get("reference")
        if ref_rel:
            ref_path = Path(ref_rel) if Path(ref_rel).is_absolute() else REPO / ref_rel
            if ref_path.exists():
                shutil.copy2(ref_path, dest / "_reference.wav")

    total_bytes = sum(p.stat().st_size for p in dest.iterdir())
    has_meta = " + meta.json" if meta_src.exists() else ""
    print(f"Copied report.html + results.csv{has_meta} + {len(wavs)} wav(s) "
          f"({total_bytes / 1024 / 1024:.1f} MB)")

    (WORKTREE / ".nojekyll").touch()
    build_pages_index()
    print(f"Rebuilt index.html — {_count_published()} run(s) total")

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
        for fname in ("index.html", "speed.html", "quality.html", "samples.html",
                      "report.html", "results.csv"):
            src = d / fname
            if src.exists():
                shutil.copy2(src, dest)
        meta_src = d / "meta.json"
        if meta_src.exists():
            shutil.copy2(meta_src, dest)
        for wav in d.glob("*.wav"):
            shutil.copy2(wav, dest)
    (WORKTREE / ".nojekyll").touch()
    build_pages_index()
    print(f"Rebuilt {len(dirs)} run(s); index updated.")

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
    args = p.parse_args()

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
