import csv

from arena import db, export_votes


def _conn(tmp_path):
    c = db.connect(str(tmp_path / "arena.db"))
    db.init_schema(c)
    return c


def _row(**kw):
    base = dict(ts=1000.0, token="t1", session_id="s1", mode="default", prompt_id=1,
                left_model="a", right_model="b", left_clip="a_p1", right_clip="b_p1",
                choice="left", dwell_ms=2000, both_played=1, turnstile_ok=1,
                pair_nonce="n1", ua="ua", ip_hash="ip1", elo_clean=1, rate_ok=1)
    base.update(kw)
    return base


def test_iso_ts_is_utc_iso8601():
    assert export_votes.iso_ts(0) == "1970-01-01T00:00:00+00:00"


def test_winner_of():
    assert export_votes.winner_of("left", "a", "b") == "a"
    assert export_votes.winner_of("right", "a", "b") == "b"
    assert export_votes.winner_of("tie", "a", "b") == "tie"
    assert export_votes.winner_of("bad", "a", "b") == "bad"


def test_fetch_rows_mode_filter(tmp_path):
    c = _conn(tmp_path)
    db.insert_vote(c, _row(pair_nonce="n1", mode="default"))
    db.insert_vote(c, _row(pair_nonce="n2", mode="cloning"))
    assert len(export_votes.fetch_rows(c)) == 2
    assert len(export_votes.fetch_rows(c, mode="cloning")) == 1


def test_fetch_rows_clean_excludes_gatefailed_and_flagged(tmp_path):
    c = _conn(tmp_path)
    db.insert_vote(c, _row(pair_nonce="n1", token="t1", elo_clean=1))
    db.insert_vote(c, _row(pair_nonce="n2", token="t1", elo_clean=0))   # gate-failed
    db.insert_vote(c, _row(pair_nonce="n3", token="bad", elo_clean=1))  # flagged token
    db.flag_token(c, "bad")
    assert len(export_votes.fetch_rows(c)) == 3
    assert len(export_votes.fetch_rows(c, clean=True)) == 1


def test_to_naq_layout(tmp_path):
    c = _conn(tmp_path)
    db.insert_vote(c, _row(pair_nonce="n1", choice="right", ts=0))
    rows = export_votes.to_naq(export_votes.fetch_rows(c))
    assert rows == [["1970-01-01T00:00:00+00:00", "t1", 1, "default",
                     "a", "b", "a_p1", "b_p1", "right", "b"]]


def test_main_writes_naq_csv(tmp_path, monkeypatch):
    c = _conn(tmp_path)
    db.insert_vote(c, _row(pair_nonce="n1", choice="left"))
    db.insert_vote(c, _row(pair_nonce="n2", choice="tie"))
    monkeypatch.setattr(export_votes, "load_settings", lambda: object())
    monkeypatch.setattr(export_votes, "open_conn", lambda _s: c)
    out = str(tmp_path / "out.csv")
    assert export_votes.main(["--out", out]) == 0
    with open(out, newline="", encoding="utf-8") as f:
        got = list(csv.reader(f))
    assert got[0] == export_votes.NAQ_HEADER
    assert len(got) == 3  # header + 2 votes
    assert got[1][-1] == "a" and got[2][-1] == "tie"  # winner column


def test_main_raw_dumps_full_schema(tmp_path, monkeypatch):
    c = _conn(tmp_path)
    db.insert_vote(c, _row(pair_nonce="n1"))
    monkeypatch.setattr(export_votes, "load_settings", lambda: object())
    monkeypatch.setattr(export_votes, "open_conn", lambda _s: c)
    out = str(tmp_path / "raw.csv")
    assert export_votes.main(["--raw", "--out", out]) == 0
    with open(out, newline="", encoding="utf-8") as f:
        got = list(csv.reader(f))
    assert got[0] == db._VOTE_COLS
    assert len(got) == 2
