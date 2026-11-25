# Does prosody predict arena votes? (2026-11-10)

**Result: no.** None of the pitch/prosody features adds measurable
predictive power over UTMOS for human arena votes, so the Scores board gets no
Prosody column.

## Test

- Votes: 6,572 clean arena votes (export of 2026-11-10). Decisive votes joined
  to clips: 2,776 default, 1,683 cloning. Ties and "both bad" votes were dropped.
- Clips: the exact gh-pages clips the arena serves (`arena/clips_manifest.json`).
  Features come from the fixed `scoring/prosody.py` (PR #51) and UTMOS from
  `scoring/scores.csv`.
- Features: |terminal slope|, declination, F0 SD, F0 range, distance of F0 SD
  and speaking rate from the lens median (two-sided), and the octave-rejected
  fraction. The cloning lens adds the F0-SD and F0-median gap to the reference
  voice. Contour shape is not compared, because the reference speaks
  different words.
- Bar, fixed before looking at votes: a no-intercept logistic model on
  ΔUTMOS + Δfeature must beat ΔUTMOS alone by ≥ 2 points of pairwise accuracy.
  Accuracy is measured on pairs whose two models were both held out, over 200
  random model splits, and the 95% interval must exclude 0.

## Numbers

| lens | held-out UTMOS | best feature gain | passes |
|---|---:|---|---|
| default | 57.2% | f0_range +2.5 pts [−6.6, +11.5] | none |
| cloning | 59.2% | f0_range +0.6 pts [−8.6, +8.8] | none |

Every interval spans roughly ±9 points: on this vote volume a feature would
need a large effect to show up, and none does. Votes cast after the 2026-10-21
house-reference re-bench are too few to analyse on their own (234 default, 96
cloning).

Health-style flags were also checked. Three clips have wild pitch
(F0 SD > 6.5 st): `dia` p1, `pocket` p1 and `voxcpm` p1 (cloning). They lose 78%
of their votes, but UTMOS already predicts 71% of those outcomes. Seven "flat"
clips (F0 SD < 1.2 st) lose 60%. Neither flag adds enough to put on the board.

Full output: `scoring/prosody_vote_test.txt`. Per-clip features:
`scoring/prosody_vote_test_clips.csv`.

## Rerun

```
python -m arena.export_votes --clean --raw --out votes.csv   # needs arena/.env (Turso)
python scripts/prosody_vote_test.py --votes votes.csv
```

The vote export holds voter tokens and is never committed.

## Caveat

The bench prompts are four short technical English sentences, which may not
exercise prosody much. A negative result here means "does not predict votes on
these prompts", not "prosody doesn't matter".
