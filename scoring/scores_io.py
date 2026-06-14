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


def merge_rows(existing, fresh_rows, overwrite=False):
    """Merge freshly-scored rows into existing. overwrite=False keeps existing
    rows for keys already present (incremental); True replaces them (--rescore)."""
    merged = dict(existing)
    for row in fresh_rows:
        key = (row["dir"], row["wav"])
        if key in merged and not overwrite:
            continue
        merged[key] = {k: row.get(k, "") for k in FIELDNAMES}
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
