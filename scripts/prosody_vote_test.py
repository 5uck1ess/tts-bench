"""Does any prosody feature predict arena votes beyond UTMOS?

Read-only analysis. Features come from the fixed scoring/prosody.py run on the
exact gh-pages clips the arena serves (via arena/clips_manifest.json), and
UTMOS comes from scoring/scores.csv. Feature list, signs and the ship bar are
fixed here, before the votes are looked at:

  ship bar (per lens): held-out-by-model pairwise accuracy of
  logit(DeltaUTMOS + DeltaFeature) beats logit(DeltaUTMOS) by >= 2 points,
  with the 95% interval over random model splits excluding 0.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth

import argparse

REPO = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--votes", type=Path, required=True,
                 help="CSV from `python -m arena.export_votes --clean --raw`")
_ap.add_argument("--gh-pages", type=Path, default=REPO / "_gh-pages")
_ap.add_argument("--out", type=Path, default=REPO / "scoring")
_args = _ap.parse_args()
VOTES, OUT, GH = _args.votes, _args.out, _args.gh_pages
sys.path.insert(0, str(REPO))
from scoring.prosody import analyze_sound, infer_pitch_bounds  # noqa: E402
from scoring.prompts import PROMPT_BY_ID  # noqa: E402

BASE = "https://5uck1ess.github.io/tts-bench/"
rng = np.random.default_rng(0)


def clip_features(path):
    snd = parselmouth.Sound(str(path))
    m = analyze_sound(snd)
    lo, hi = infer_pitch_bounds(snd)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=lo, pitch_ceiling=hi)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    voiced_s = m.n_voiced * 0.01
    return {
        "terminal_abs": abs(m.terminal_slope),
        "declination": m.declination_slope,
        "f0_sd": m.f0_sd_st,
        "f0_range": m.f0_range_st,
        "octave_frac": m.n_octave_rejected / max(1, m.n_voiced + m.n_octave_rejected),
        "duration_s": m.duration_s,
        "voiced_s": voiced_s,
        "f0_median_st": 12 * np.log2(np.median(f0) / 100.0) if f0.size else np.nan,
    }


manifest = json.loads((REPO / "arena/clips_manifest.json").read_text(encoding="utf-8"))
scores = pd.read_csv(REPO / "scoring/scores.csv")
utmos = {(r.dir, r.wav): r.utmos for r in scores.itertuples()}

rows = []
refs = {}
for mode in ("default", "cloning"):
    block = manifest["modes"][mode]
    if block.get("reference_url"):
        rel = block["reference_url"].replace(BASE, "")
        refs[mode] = clip_features(GH / rel)
    for c in block["clips"]:
        rel = c["url"].replace(BASE, "")
        d, wav = rel.split("/", 1)
        p = GH / rel
        if not p.exists():
            continue
        f = clip_features(p)
        words = len(PROMPT_BY_ID[str(c["prompt"])][1].split())
        f["rate_wps"] = words / f["voiced_s"] if f["voiced_s"] > 0 else np.nan
        rows.append({"mode": mode, "model": c["model"], "prompt": int(c["prompt"]),
                     "dir": d, "wav": wav, "utmos": utmos.get((d, wav), np.nan), **f})
clips = pd.DataFrame(rows)
clips.to_csv(OUT / "prosody_vote_test_clips.csv", index=False)

# Reference-match features for cloning (reference speaks different words, so
# only global level/spread, never contour shape).
ref = refs.get("cloning")
cl = clips["mode"] == "cloning"
clips.loc[cl, "ref_f0_sd_gap"] = (clips.loc[cl, "f0_sd"] - ref["f0_sd"]).abs()
clips.loc[cl, "ref_f0_med_gap"] = (clips.loc[cl, "f0_median_st"] - ref["f0_median_st"]).abs()
# two-sided spread: distance from the lens median (too flat and too wild both bad)
for mode in ("default", "cloning"):
    sel = clips["mode"] == mode
    clips.loc[sel, "f0_sd_dev"] = (clips.loc[sel, "f0_sd"] - clips.loc[sel, "f0_sd"].median()).abs()
    clips.loc[sel, "rate_dev"] = (np.log(clips.loc[sel, "rate_wps"]) - np.log(clips.loc[sel, "rate_wps"].median())).abs()

key = clips.set_index(["mode", "model", "prompt"])

votes = pd.read_csv(VOTES)
votes = votes[votes["choice"].isin(["left", "right"]) & votes["mode"].isin(["default", "cloning"])]
votes["ts"] = pd.to_numeric(votes["ts"], errors="coerce")

FEATURES = ["terminal_abs", "declination", "f0_sd", "f0_range", "f0_sd_dev",
            "rate_dev", "octave_frac", "ref_f0_sd_gap", "ref_f0_med_gap"]


def build_pairs(v):
    out = []
    for r in v.itertuples():
        try:
            a = key.loc[(r.mode, r.left_model, int(r.prompt_id))]
            b = key.loc[(r.mode, r.right_model, int(r.prompt_id))]
        except KeyError:
            continue
        if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
            continue
        rec = {"mode": r.mode, "left": r.left_model, "right": r.right_model,
               "y": 1 if r.choice == "left" else 0, "ts": r.ts,
               "d_utmos": a.utmos - b.utmos}
        for f in FEATURES:
            rec["d_" + f] = a.get(f, np.nan) - b.get(f, np.nan)
        out.append(rec)
    return pd.DataFrame(out)


def fit_logit(X, y, l2=1e-3, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w


def heldout(pairs, cols, n_splits=200):
    """Accuracy of a no-intercept logit on pairs whose BOTH models are held out."""
    models = sorted(set(pairs.left) | set(pairs.right))
    accs, ns = [], []
    for _ in range(n_splits):
        test = set(rng.choice(models, size=len(models) // 2, replace=False))
        tr = pairs[~pairs.left.isin(test) & ~pairs.right.isin(test)]
        te = pairs[pairs.left.isin(test) & pairs.right.isin(test)]
        if len(tr) < 30 or len(te) < 30:
            continue
        Xtr = tr[cols].to_numpy(float)
        sd = Xtr.std(0) + 1e-9
        w = fit_logit(Xtr / sd, tr.y.to_numpy(float))
        pred = (te[cols].to_numpy(float) / sd) @ w > 0
        accs.append((pred == (te.y.to_numpy() == 1)).mean())
        ns.append(len(te))
    return np.array(accs), int(np.median(ns)) if ns else 0


def single_feature_agreement(pairs, col):
    d = pairs[col]
    m = d != 0
    agree = ((d[m] > 0) == (pairs.y[m] == 1)).mean()
    return agree, int(m.sum())


lines = []
for label, vsub in (("all clean votes", votes),
                    ("votes since 2026-10-21 (after house-ref re-bench)",
                     votes[votes["ts"] >= pd.Timestamp("2026-10-21").timestamp()])):
    lines.append(f"\n=== {label} ===")
    allpairs = build_pairs(vsub)
    if allpairs.empty:
        lines.append("   no votes joined")
        continue
    for mode in ("default", "cloning"):
        pairs = allpairs[allpairs["mode"] == mode].dropna(subset=["d_utmos"])
        lines.append(f"\n-- {mode}: {len(pairs)} decisive votes joined, "
                     f"{len(set(pairs.left) | set(pairs.right))} models")
        if len(pairs) < 100:
            lines.append("   too few votes")
            continue
        a, n = single_feature_agreement(pairs, "d_utmos")
        lines.append(f"   UTMOS alone (raw sign): {a:.3f} on {n}")
        base, nte = heldout(pairs, ["d_utmos"])
        lines.append(f"   held-out UTMOS logit: {base.mean():.3f} (median test n={nte})")
        feats = [f for f in FEATURES if not (mode == "default" and f.startswith("ref_"))]
        for f in feats:
            col = "d_" + f
            sub = pairs.dropna(subset=[col])
            if len(sub) < 100:
                continue
            a, n = single_feature_agreement(sub, col)
            b, _ = heldout(sub, ["d_utmos"])
            c, _ = heldout(sub, ["d_utmos", col])
            k = min(len(b), len(c))
            diff = c[:k] - b[:k]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            verdict = "PASS" if diff.mean() >= 0.02 and lo > 0 else "fail"
            lines.append(
                f"   {f:15s} alone(+sign)={a:.3f}  UTMOS+f - UTMOS = {diff.mean():+.3f} "
                f"[{lo:+.3f}, {hi:+.3f}]  {verdict}")

# Health-style flags: do flagged clips lose more often?
lines.append("\n=== flags (all clean votes) ===")
def clip_flag_loss(flag_fn, name):
    wins = losses = utmos_right = 0
    names = set()
    for r in votes.itertuples():
        for side, other, won in ((r.left_model, r.right_model, r.choice == "left"),
                                 (r.right_model, r.left_model, r.choice == "right")):
            try:
                c = key.loc[(r.mode, side, int(r.prompt_id))]
            except (KeyError, ValueError):
                continue
            if isinstance(c, pd.DataFrame) or not flag_fn(c):
                continue
            wins += won
            losses += not won
            try:
                o = key.loc[(r.mode, other, int(r.prompt_id))]
                utmos_right += (c.utmos > o.utmos) == won
            except (KeyError, ValueError):
                pass
            names.add(f"{r.mode}:{side}:p{r.prompt_id}")
    n = wins + losses
    lines.append(f"   {name}: flagged clip appearances={n}, loss rate="
                 f"{(losses / n if n else float('nan')):.3f}, UTMOS already "
                 f"calls {(utmos_right / n if n else float('nan')):.3f} of these votes")
    lines.append(f"      clips: {sorted(names)}")
clip_flag_loss(lambda c: c.f0_sd < 1.2, "flat (f0_sd < 1.2 st)")
clip_flag_loss(lambda c: c.f0_sd > 6.5, "wild (f0_sd > 6.5 st)")
clip_flag_loss(lambda c: c.octave_frac > 0.05, "unstable (>5% octave-rejected)")
flat_n = (clips.f0_sd < 1.2).sum(); wild_n = (clips.f0_sd > 6.5).sum()
lines.append(f"   clips flagged: flat={flat_n}, wild={wild_n}, unstable={(clips.octave_frac > 0.05).sum()} of {len(clips)}")

report = "\n".join(lines)
(OUT / "prosody_vote_test.txt").write_text(report, encoding="utf-8")
print(report)
