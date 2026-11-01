"""Score the hosted-API clips (docs/hosted.md) → scoring/hosted_scores.csv.

Same scorers as score_all.py (UTMOS / WER / Health; SIM for the cloning lens), over the
collector's layout instead of _gh-pages: hosted/<slug>/<lens>/p<pid>_t<take>.wav. Every take
is scored; the board averages them. `dir` is "<slug>/<lens>", so rows never collide with the
local board's scores.csv. Runs on the Linux box like score_all (copy hosted/ over first,
without hosted/_private/):

    venvs/scoring/bin/python -m scoring.score_hosted            # incremental
    venvs/scoring/bin/python -m scoring.score_hosted --rescore
"""

import argparse
import os
import re
import sys

from scoring.clips import Clip
from scoring.score_all import score_clips
from scoring.scores_io import merge_rows, read_scores, write_scores

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HOSTED = os.path.join(_REPO, "hosted")
_CSV = os.path.join(_REPO, "scoring", "hosted_scores.csv")
_REF = os.path.join(_REPO, "reference", "hosted_clone.wav")
_WAV_RE = re.compile(r"^p(?P<pid>\d+)_t\d+\.wav$")


def discover(root):
    clips = []
    for slug in sorted(os.listdir(root)):
        if slug.startswith("_"):
            continue
        for lens in ("default", "cloning"):
            d = os.path.join(root, slug, lens)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                m = _WAV_RE.match(fn)
                if m:
                    clips.append(Clip(dir=f"{slug}/{lens}", wav=fn, model=slug, dev="api",
                                      prompt_id=m.group("pid"), mode=lens))
    return clips


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rescore", action="store_true")
    p.add_argument("--root", default=_HOSTED)
    p.add_argument("--model", action="append", help="only these slugs; repeatable")
    args = p.parse_args(argv)

    clips = discover(args.root)
    if args.model:
        clips = [c for c in clips if c.model in args.model]
    existing = read_scores(_CSV)
    todo = clips if args.rescore else [c for c in clips if (c.dir, c.wav) not in existing]
    print(f"{len(clips)} clips found; scoring {len(todo)}.", flush=True)
    if not todo:
        return 0

    from scoring.health import HealthScorer
    from scoring.utmos import UtmosScorer
    from scoring.wer import WerScorer
    sim = None
    if any(c.mode == "cloning" for c in todo):
        from scoring.sim import SimScorer  # fairseq; absent in the py3.11 venv → sim_pass owns it
        sim = SimScorer()
    ref = (lambda _d: _REF if os.path.exists(_REF) else None)
    fresh = score_clips(todo, UtmosScorer(), WerScorer(), sim, HealthScorer(),
                        ref_for_dir=ref, gh_root=args.root)
    merged = merge_rows(existing, fresh, overwrite=args.rescore)
    write_scores(_CSV, merged)
    print(f"Wrote {_CSV} ({len(merged)} rows).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
