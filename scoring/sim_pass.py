"""Fill the SIM column of scoring/scores.csv — run from the py3.10 SIM venv.

SIM's canonical UniSpeech-SAT path needs `fairseq`, which does not run in the
py3.11 `venvs/scoring` (UTMOS/WER do — they own scores.csv row creation). fairseq
0.12.2 imports cleanly only on Python <=3.10, so SIM lives in its own venv
(`venvs/scoring_sim`, built by install.sh). This pass scores ONLY SIM for cloning
clips that have a published reference and a blank sim cell, leaving the already
computed utmos/wer untouched.

ONE machine runs this over the consolidated published clips (audio is rig
independent). Idempotent: re-runs skip clips already carrying a sim score unless
--rescore. Per-clip failures leave that cell blank + log to stderr; never aborts.

OPERATOR PRECONDITION (this script does NOT touch git): refresh the worktree and
run scoring.score_all (UTMOS/WER) first so scores.csv exists, then —
    git -C _gh-pages fetch origin gh-pages
    git -C _gh-pages reset --hard origin/gh-pages

Run (SIM venv):
    venvs/scoring_sim/bin/python -m scoring.sim_pass            # fill missing only
    venvs/scoring_sim/bin/python -m scoring.sim_pass --rescore  # recompute all SIM
"""

import argparse
import os
import sys

from scoring.clips import discover_clips
from scoring.scores_io import read_scores, write_scores
# ref_for_dir / _fmt / _try live in score_all; importing it is safe here because
# its UTMOS/WER scorer imports are lazy (inside main()), not at module load.
from scoring.score_all import ref_for_dir, _fmt, _try, _GH, _SCORES_CSV


def _sibling_cloning_dir(default_dir):
    """windows-default → windows-cloning. None if not a *-default dir."""
    if not default_dir.endswith("-default"):
        return None
    return default_dir[: -len("-default")] + "-cloning"


def select_todo(clips, existing, ref_for_dir=ref_for_dir, no_preset=frozenset(),
                rescore=False):
    """Return [(clip, ref_path)] for clips needing SIM. Two reference-clone sources:

      - cloning-mode clips → scored against their own dir's published _reference.wav.
      - default-mode clips of a NO_PRESET_VOICE model that has NO cloning clip: their
        no-`--reference` run clones the bundled Chris reference, so they ARE Chris
        clones — score them against the sibling cloning dir's _reference.wav. This
        mirrors publish.py's board fallback (those clips show under Cloning), so the
        SIM column matches the clip the board actually displays. Models WITH a real
        preset voice (e.g. vibevoice 0.5B → en-Emma) are excluded — scoring their
        default clip against Chris would be a different-speaker (meaningless) number.
    """
    has_cloning = {c.model for c in clips if c.mode == "cloning"}
    todo = []
    for c in clips:
        row = existing.get((c.dir, c.wav))
        if row is None:
            continue  # score_all owns row creation (utmos/wer); skip un-scored clips
        if row.get("sim", "").strip() and not rescore:
            continue
        if c.mode == "cloning":
            ref = ref_for_dir(c.dir)
        elif c.model in no_preset and c.model not in has_cloning:
            sib = _sibling_cloning_dir(c.dir)
            ref = ref_for_dir(sib) if sib else None
        else:
            continue  # default clip of a real-preset model → no SIM (cloning-only metric)
        if ref is None:
            continue  # no reference available → SIM not applicable
        todo.append((c, ref))
    return todo


def main(argv=None):
    p = argparse.ArgumentParser(description="Fill SIM in scoring/scores.csv (py3.10 SIM venv).")
    p.add_argument("--rescore", action="store_true",
                   help="Recompute clips already carrying a sim score.")
    p.add_argument("--gh-root", default=_GH, help="Path to the _gh-pages worktree.")
    args = p.parse_args(argv)

    if not os.path.isdir(args.gh_root):
        raise SystemExit(f"No worktree at {args.gh_root}. Refresh it first "
                         f"(git -C _gh-pages reset --hard origin/gh-pages).")

    existing = read_scores(_SCORES_CSV)
    if not existing:
        raise SystemExit(f"{_SCORES_CSV} is empty — run scoring.score_all "
                         f"(UTMOS/WER) in venvs/scoring first.")

    # NO_PRESET_VOICE (models whose no-reference "default" run is a bundled Chris
    # clone) is publish.py's — the single source of truth. Pulled here, not at module
    # import, so scoring.sim_pass stays decoupled and unit-testable without publish.
    from publish import NO_PRESET_VOICE

    clips = discover_clips(args.gh_root)
    todo = select_todo(clips, existing, no_preset=NO_PRESET_VOICE, rescore=args.rescore)
    cloning = sum(1 for c in clips if c.mode == "cloning")
    print(f"{cloning} cloning clips; scoring SIM for {len(todo)} "
          f"({'rescore' if args.rescore else 'missing only'}).", flush=True)
    if not todo:
        print("Nothing to score.")
        return 0

    from scoring.sim import SimScorer
    sim = SimScorer()

    for c, ref in todo:
        wav = os.path.join(args.gh_root, c.dir, c.wav)
        s = _try(lambda: sim.score(wav, ref), "sim", c.wav)
        existing[(c.dir, c.wav)]["sim"] = _fmt(s)

    write_scores(_SCORES_CSV, existing)
    print(f"Wrote {_SCORES_CSV} ({len(existing)} rows).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
