"""Score every published clip in _gh-pages/ → scoring/scores.csv.

ONE machine runs this over the consolidated published clips (audio is
rig-independent). Idempotent: re-runs skip clips already in scores.csv unless
--rescore. Per-clip, per-metric failures leave that cell blank + log to stderr;
the pass never aborts.

OPERATOR PRECONDITION (this script does NOT touch git): refresh the worktree
first so it holds every rig's current clips —
    git -C _gh-pages fetch origin gh-pages
    git -C _gh-pages reset --hard origin/gh-pages

Run (scoring venv):
    venvs/scoring/bin/python -m scoring.score_all            # incremental
    venvs/scoring/bin/python -m scoring.score_all --rescore  # recompute all
"""

import argparse
import os
import sys

from scoring.clips import discover_clips
from scoring.prompts import PROMPT_BY_ID
from scoring.scores_io import read_scores, merge_rows, write_scores

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GH = os.path.join(_REPO, "_gh-pages")
_SCORES_CSV = os.path.join(_REPO, "scoring", "scores.csv")


def _fmt(v):
    return "" if v is None else f"{v:.4f}"


def _try(fn, label, wav):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — never abort the whole pass
        print(f"  WARN {label} failed on {wav}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def ref_for_dir(dir_name):
    """The cloning reference published in each *-cloning/ dir, or None."""
    ref = os.path.join(_GH, dir_name, "_reference.wav")
    return ref if os.path.exists(ref) else None


def score_clips(clips, utmos, wer, sim, health=None, ref_for_dir=ref_for_dir, gh_root=_GH):
    """Score each clip → list of scores.csv row dicts. Pure given injected scorers."""
    rows = []
    for c in clips:
        wav = os.path.join(gh_root, c.dir, c.wav)
        u = _try(lambda: utmos.score(wav), "utmos", c.wav)
        w = None
        if c.prompt_id in PROMPT_BY_ID:
            lang, text = PROMPT_BY_ID[c.prompt_id]
            w = _try(lambda: wer.score(wav, text, lang), "wer", c.wav)
        s = None
        if c.mode == "cloning":
            ref = ref_for_dir(c.dir)
            if ref:
                s = _try(lambda: sim.score(wav, ref), "sim", c.wav)
        # Deterministic health triage — reference-free, no text, no ML. "" = clean.
        h = _try(lambda: health.score(wav), "health", c.wav) if health else None
        rows.append({"dir": c.dir, "wav": c.wav, "model": c.model, "mode": c.mode,
                     "prompt_id": c.prompt_id,
                     "utmos": _fmt(u), "wer": _fmt(w), "sim": _fmt(s),
                     "health": h or ""})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Score published clips → scoring/scores.csv")
    p.add_argument("--rescore", action="store_true",
                   help="Recompute clips already present in scores.csv.")
    p.add_argument("--gh-root", default=_GH, help="Path to the _gh-pages worktree.")
    args = p.parse_args(argv)

    if not os.path.isdir(args.gh_root):
        raise SystemExit(f"No worktree at {args.gh_root}. Refresh it first "
                         f"(git -C _gh-pages reset --hard origin/gh-pages).")

    clips = discover_clips(args.gh_root)
    existing = read_scores(_SCORES_CSV)
    todo = clips if args.rescore else [c for c in clips
                                       if (c.dir, c.wav) not in existing]
    print(f"{len(clips)} clips found; scoring {len(todo)} "
          f"({'rescore' if args.rescore else 'new only'}).", flush=True)
    if not todo:
        print("Nothing to score.")
        return 0

    from scoring.utmos import UtmosScorer
    from scoring.wer import WerScorer
    from scoring.sim import SimScorer
    from scoring.health import HealthScorer
    utmos, wer, sim, health = UtmosScorer(), WerScorer(), SimScorer(), HealthScorer()

    fresh = score_clips(todo, utmos, wer, sim, health, gh_root=args.gh_root)
    merged = merge_rows(existing, fresh, overwrite=args.rescore)
    write_scores(_SCORES_CSV, merged)
    print(f"Wrote {_SCORES_CSV} ({len(merged)} rows).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
