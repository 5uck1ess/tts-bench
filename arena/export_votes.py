"""Pull the arena vote log out of the datastore into a local CSV.

Reads the append-only ``votes`` table (Turso in prod, sqlite locally) and writes
a CSV. The default layout mirrors ``naq_lab/naq_votes_*.csv`` so an export drops
straight into the NAQ analysis pipeline; ``--raw`` instead dumps every column of
the votes schema for deep QC.

Connection comes from the same env the Space uses (TURSO_URL/TURSO_TOKEN, else
the local sqlite at ARENA_DB) via config.load_settings(), so:

    # against prod Turso (export the secrets first, or run where they're set)
    python -m arena.export_votes --out votes_export.csv

    # gate-passing rows only (matches the live Elo board's inclusion)
    python -m arena.export_votes --clean --out votes_clean.csv

    # full schema, single lens
    python -m arena.export_votes --raw --mode cloning --out cloning_raw.csv
"""

import argparse
import csv
import sys
from datetime import datetime, timezone

from . import db as dbmod
from .config import load_settings

# NAQ-compatible header (matches naq_lab/naq_votes_*.csv).
NAQ_HEADER = ["ts", "rater", "prompt_id", "mode", "left_model", "right_model",
              "left_clip", "right_clip", "choice", "winner"]


def iso_ts(epoch: float) -> str:
    """Float epoch seconds -> ISO-8601 UTC (the format naq_lab CSVs use)."""
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()


def winner_of(choice: str, left_model: str, right_model: str) -> str:
    """The NAQ 'winner' column: the chosen model, or the verdict for tie/bad."""
    if choice == "left":
        return left_model
    if choice == "right":
        return right_model
    return choice  # 'tie' / 'bad' carried through verbatim


def fetch_rows(conn, mode: str | None = None, clean: bool = False) -> list:
    """Return votes as dicts ordered by insertion (id). ``mode`` None = both
    lenses. ``clean`` restricts to gate-passing rows from non-flagged tokens —
    the same inclusion the live board uses (still keeps tie/bad as good-faith
    votes; the board itself drops 'bad' downstream)."""
    where, params = [], []
    if mode:
        where.append("mode = ?")
        params.append(mode)
    if clean:
        where.append("elo_clean = 1")
        where.append("token NOT IN (SELECT token FROM tokens WHERE flagged = 1)")
    sql = "SELECT * FROM votes"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id"
    cur = conn.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]


def to_naq(rows: list) -> list:
    """Map raw vote dicts onto the NAQ_HEADER column layout."""
    out = []
    for r in rows:
        out.append([
            iso_ts(r["ts"]), r["token"], r["prompt_id"], r["mode"],
            r["left_model"], r["right_model"], r["left_clip"], r["right_clip"],
            r["choice"], winner_of(r["choice"], r["left_model"], r["right_model"]),
        ])
    return out


def write_csv(path: str, header: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def open_conn(settings):
    """Open the datastore the Space uses: Turso if configured, else sqlite."""
    if settings.use_turso:
        from . import turso
        return turso.connect(settings.turso_url, settings.turso_token)
    return dbmod.connect(settings.db_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export arena votes to CSV.")
    ap.add_argument("--out", default="votes_export.csv", help="output CSV path")
    ap.add_argument("--mode", choices=["default", "cloning", "all"], default="all",
                    help="which lens to export (default: all)")
    ap.add_argument("--clean", action="store_true",
                    help="only gate-passing rows from non-flagged tokens")
    ap.add_argument("--raw", action="store_true",
                    help="dump the full votes schema instead of the NAQ layout")
    args = ap.parse_args(argv)

    settings = load_settings()
    conn = open_conn(settings)
    mode = None if args.mode == "all" else args.mode
    rows = fetch_rows(conn, mode=mode, clean=args.clean)

    if args.raw:
        header = dbmod._VOTE_COLS
        data = [[r.get(c) for c in header] for r in rows]
    else:
        header, data = NAQ_HEADER, to_naq(rows)

    write_csv(args.out, header, data)
    print(f"wrote {len(data)} vote(s) -> {args.out}"
          f"  [{args.mode}{', clean' if args.clean else ''}"
          f"{', raw' if args.raw else ''}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
