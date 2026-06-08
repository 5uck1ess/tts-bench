"""SQLite datastore for the arena (votes log + tokens + rankings cache).

Standard SQLite SQL only, so the identical statements run on Turso/libSQL in
production (see arena/turso.py). stdlib sqlite3 backs local dev and tests.
"""

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS votes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    token       TEXT    NOT NULL,
    session_id  TEXT,
    mode        TEXT    NOT NULL,
    prompt_id   INTEGER NOT NULL,
    left_model  TEXT    NOT NULL,
    right_model TEXT    NOT NULL,
    left_clip   TEXT,
    right_clip  TEXT,
    choice      TEXT    NOT NULL,
    dwell_ms    INTEGER,
    both_played INTEGER,
    turnstile_ok INTEGER,
    pair_nonce  TEXT    UNIQUE,
    gold_pair_id TEXT,
    ua          TEXT,
    ip_hash     TEXT,
    elo_clean   INTEGER NOT NULL DEFAULT 0,
    rate_ok     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_votes_mode  ON votes(mode);
CREATE INDEX IF NOT EXISTS ix_votes_token ON votes(token);
CREATE INDEX IF NOT EXISTS ix_votes_ip    ON votes(ip_hash);

CREATE TABLE IF NOT EXISTS tokens (
    token        TEXT PRIMARY KEY,
    first_seen   REAL,
    last_vote_ts REAL,
    vote_count   INTEGER NOT NULL DEFAULT 0,
    flagged      INTEGER NOT NULL DEFAULT 0,
    gold_pass_rate REAL
);

CREATE TABLE IF NOT EXISTS rankings (
    mode       TEXT NOT NULL,
    model      TEXT NOT NULL,
    elo        REAL NOT NULL,
    games      INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (mode, model)
);
"""

_VOTE_COLS = ["ts", "token", "session_id", "mode", "prompt_id", "left_model",
              "right_model", "left_clip", "right_clip", "choice", "dwell_ms",
              "both_played", "turnstile_ok", "pair_nonce", "gold_pair_id", "ua",
              "ip_hash", "elo_clean", "rate_ok"]


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def insert_vote(conn, row: dict) -> None:
    cols = ", ".join(_VOTE_COLS)
    ph = ", ".join("?" for _ in _VOTE_COLS)
    vals = [row.get(c) for c in _VOTE_COLS]
    conn.execute(f"INSERT INTO votes ({cols}) VALUES ({ph})", vals)
    conn.commit()


def nonce_seen(conn, pair_nonce: str) -> bool:
    cur = conn.execute("SELECT 1 FROM votes WHERE pair_nonce = ? LIMIT 1", (pair_nonce,))
    return cur.fetchone() is not None


def clean_votes(conn, mode: str) -> list:
    """Ordered (left, right, choice) for the live Elo of ``mode``."""
    cur = conn.execute(
        """SELECT left_model, right_model, choice FROM votes
           WHERE mode = ? AND elo_clean = 1 AND choice != 'bad'
             AND token NOT IN (SELECT token FROM tokens WHERE flagged = 1)
           ORDER BY id""", (mode,))
    return [(r["left_model"], r["right_model"], r["choice"]) for r in cur.fetchall()]


def clean_vote_count(conn, mode: str) -> int:
    """Count gate-passing (clean) votes for ``mode``, INCLUDING tie/bad — i.e. all
    good-faith votes collected. This is the public 'votes collected' tally shown on
    the page; it differs from the Elo board, which still excludes 'bad' as a non-
    preference. Flagged tokens are excluded (mirrors the board)."""
    cur = conn.execute(
        """SELECT COUNT(*) AS n FROM votes
           WHERE mode = ? AND elo_clean = 1
             AND token NOT IN (SELECT token FROM tokens WHERE flagged = 1)""",
        (mode,))
    return cur.fetchone()["n"]


def bump_token(conn, token: str, ts: float) -> None:
    conn.execute(
        """INSERT INTO tokens (token, first_seen, last_vote_ts, vote_count)
           VALUES (?, ?, ?, 1)
           ON CONFLICT(token) DO UPDATE SET
             last_vote_ts = excluded.last_vote_ts,
             vote_count   = vote_count + 1""", (token, ts, ts))
    conn.commit()


def flag_token(conn, token: str) -> None:
    conn.execute(
        "INSERT INTO tokens (token, flagged) VALUES (?, 1) "
        "ON CONFLICT(token) DO UPDATE SET flagged = 1", (token,))
    conn.commit()


def token_state(conn, token: str):
    cur = conn.execute("SELECT * FROM tokens WHERE token = ?", (token,))
    r = cur.fetchone()
    return dict(r) if r else None


def is_token_flagged(conn, token: str) -> bool:
    st = token_state(conn, token)
    return bool(st and st["flagged"])


def daily_count(conn, token: str, since_ts: float) -> int:
    cur = conn.execute(
        "SELECT COUNT(*) AS n FROM votes WHERE token = ? AND ts >= ?", (token, since_ts))
    return cur.fetchone()["n"]


def ip_distinct_tokens(conn, ip_hash: str, since_ts: float) -> int:
    cur = conn.execute(
        "SELECT COUNT(DISTINCT token) AS n FROM votes WHERE ip_hash = ? AND ts >= ?",
        (ip_hash, since_ts))
    return cur.fetchone()["n"]


def upsert_ranking(conn, mode: str, model: str, elo: float, games: int,
                   updated_at: float) -> None:
    conn.execute(
        """INSERT INTO rankings (mode, model, elo, games, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(mode, model) DO UPDATE SET
             elo = excluded.elo, games = excluded.games,
             updated_at = excluded.updated_at""",
        (mode, model, elo, games, updated_at))
    conn.commit()


def get_ranking(conn, mode: str) -> list:
    cur = conn.execute(
        "SELECT model, elo, games FROM rankings WHERE mode = ? ORDER BY elo DESC",
        (mode,))
    return [{"model": r["model"], "elo": round(r["elo"]), "games": r["games"]}
            for r in cur.fetchall()]
