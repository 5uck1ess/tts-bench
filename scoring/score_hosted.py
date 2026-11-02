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
_REF = os.path.join(_REPO, "reference", "tym_hosted_30.wav")  # SIM target (23 s; the 113 s cut OOMs WavLM on the 3090); published as "Reference voice"
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
    p.add_argument("--sim-only", action="store_true",
                   help="fill SIM on cloning rows; run from venvs/scoring_sim (fairseq, py3.10)")
    args = p.parse_args(argv)

    clips = discover(args.root)
    if args.model:
        clips = [c for c in clips if c.model in args.model]
    existing = read_scores(_CSV)
    ref = (lambda _d: _REF if os.path.exists(_REF) else None)
    if args.sim_only:
        from scoring.score_all import _fmt, _try
        from scoring.sim import SimScorer
        sim = SimScorer()
        n = 0
        for c in clips:
            row = existing.get((c.dir, c.wav))
            if c.mode != "cloning" or row is None or (row["sim"] and not args.rescore):
                continue
            wav = os.path.join(args.root, c.dir, c.wav)
            row["sim"] = _fmt(_try(lambda: sim.score(wav, _REF), "sim", c.wav))
            n += 1
        write_scores(_CSV, existing)
        print(f"SIM filled on {n} clips -> {_CSV}")
        return 0
    todo = clips if args.rescore else [c for c in clips if (c.dir, c.wav) not in existing]
    print(f"{len(clips)} clips found; scoring {len(todo)}.", flush=True)
    if not todo:
        return 0

    from scoring.health import HealthScorer
    from scoring.utmos import UtmosScorer
    from scoring.wer import WerScorer

    class _NoSim:  # fairseq lives in venvs/scoring_sim; --sim-only fills SIM afterwards
        def score(self, *_a):
            return None
    sim = _NoSim()
    fresh = score_clips(todo, UtmosScorer(), WerScorer(), sim, HealthScorer(),
                        ref_for_dir=ref, gh_root=args.root)
    merged = merge_rows(existing, fresh, overwrite=args.rescore)
    write_scores(_CSV, merged)
    print(f"Wrote {_CSV} ({len(merged)} rows).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
