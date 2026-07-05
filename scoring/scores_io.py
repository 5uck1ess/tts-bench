"""Read / merge / write scoring/scores.csv (pure stdlib).

Keyed by (dir, wav). One row per scored clip. Missing metric = empty string.
Deterministic row order (sorted by dir, wav) so diffs stay small across re-runs.
"""

import csv

FIELDNAMES = ["dir", "wav", "model", "mode", "prompt_id", "utmos", "wer", "sim", "health"]


def read_scores(path):
    """Return {(dir, wav): row_dict}; {} if the file is absent."""
    out = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[(row["dir"], row["wav"])] = {k: row.get(k, "") for k in FIELDNAMES}
    except FileNotFoundError:
        return {}
    return out


# Metric fields owned by more than one scoring pass/venv: a blank fresh value
# means "this pass couldn't compute it" (e.g. SIM from the py3.11 scoring venv,
# where fairseq is absent — sim_pass.py owns SIM from its own py3.10 venv), NOT
# "the metric is now blank". An overwrite merge must never clobber a real
# existing value with "". `health` is excluded on purpose: "" is its legitimate
# "clean" verdict, so a fresh blank there is a real result and wins.
_KEEP_EXISTING_IF_BLANK = ("utmos", "wer", "sim")


def merge_rows(existing, fresh_rows, overwrite=False):
    """Merge freshly-scored rows into existing. overwrite=False keeps existing
    rows for keys already present (incremental); True replaces them (--rescore) —
    except a blank metric cell never overwrites a non-blank one (see above)."""
    merged = dict(existing)
    for row in fresh_rows:
        key = (row["dir"], row["wav"])
        if key in merged and not overwrite:
            continue
        new = {k: row.get(k, "") for k in FIELDNAMES}
        old = merged.get(key)
        if old:
            for k in _KEEP_EXISTING_IF_BLANK:
                if not new[k] and old.get(k, ""):
                    new[k] = old[k]
        merged[key] = new
    return merged


def write_scores(path, rows):
    """Write rows (a list of dicts OR a {key: row} mapping) sorted by (dir, wav)."""
    if isinstance(rows, dict):
        rows = list(rows.values())
    rows = sorted(rows, key=lambda r: (r["dir"], r["wav"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
