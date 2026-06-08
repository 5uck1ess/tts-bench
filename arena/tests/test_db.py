from arena import db


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


def test_insert_and_nonce_seen(tmp_path):
    c = _conn(tmp_path)
    assert db.nonce_seen(c, "n1") is False
    db.insert_vote(c, _row(pair_nonce="n1"))
    assert db.nonce_seen(c, "n1") is True


def test_clean_votes_filters_bad_and_unclean(tmp_path):
    c = _conn(tmp_path)
    db.insert_vote(c, _row(pair_nonce="n1", choice="left", elo_clean=1))
    db.insert_vote(c, _row(pair_nonce="n2", choice="bad", elo_clean=1))      # bad excluded
    db.insert_vote(c, _row(pair_nonce="n3", choice="right", elo_clean=0))    # unclean excluded
    rows = db.clean_votes(c, "default")
    assert rows == [("a", "b", "left")]


def test_clean_vote_count_includes_tie_and_bad_but_not_gatefailed(tmp_path):
    c = _conn(tmp_path)
    db.insert_vote(c, _row(pair_nonce="n1", choice="left", elo_clean=1))
    db.insert_vote(c, _row(pair_nonce="n2", choice="bad", elo_clean=1))     # counts (good-faith)
    db.insert_vote(c, _row(pair_nonce="n3", choice="tie", elo_clean=1))     # counts
    db.insert_vote(c, _row(pair_nonce="n4", choice="left", elo_clean=0))    # gate-failed -> excluded
    db.insert_vote(c, _row(pair_nonce="n5", choice="left", elo_clean=1, mode="cloning"))  # other mode
    assert db.clean_vote_count(c, "default") == 3   # left + bad + tie, not the gate-failed one
    assert db.clean_vote_count(c, "cloning") == 1
    # the Elo board still excludes 'bad' from default (only left counts there)
    assert db.clean_votes(c, "default") == [("a", "b", "left"), ("a", "b", "tie")]


def test_flagged_token_excluded_from_clean(tmp_path):
    c = _conn(tmp_path)
    db.insert_vote(c, _row(pair_nonce="n1", token="bad", choice="left"))
    db.bump_token(c, "bad", 1000.0)
    assert db.clean_votes(c, "default") == [("a", "b", "left")]
    db.flag_token(c, "bad")
    assert db.clean_votes(c, "default") == []


def test_token_counts_for_ratelimit(tmp_path):
    c = _conn(tmp_path)
    db.bump_token(c, "t1", 100.0)
    db.bump_token(c, "t1", 105.0)
    st = db.token_state(c, "t1")
    assert st["vote_count"] == 2
    assert st["last_vote_ts"] == 105.0
    assert st["flagged"] == 0
    assert db.daily_count(c, "t1", since_ts=0.0) == 0  # counts votes table, not tokens
    db.insert_vote(c, _row(token="t1", ts=200.0, pair_nonce="n9"))
    assert db.daily_count(c, "t1", since_ts=0.0) == 1


def test_ip_distinct_tokens(tmp_path):
    c = _conn(tmp_path)
    db.insert_vote(c, _row(token="t1", ip_hash="ipX", pair_nonce="n1", ts=10.0))
    db.insert_vote(c, _row(token="t2", ip_hash="ipX", pair_nonce="n2", ts=20.0))
    db.insert_vote(c, _row(token="t3", ip_hash="ipY", pair_nonce="n3", ts=30.0))
    assert db.ip_distinct_tokens(c, "ipX", since_ts=0.0) == 2
    assert db.ip_distinct_tokens(c, "ipY", since_ts=0.0) == 1


def test_ranking_cache_roundtrip(tmp_path):
    c = _conn(tmp_path)
    db.upsert_ranking(c, "default", "a", 1012.0, 3, 1000.0)
    db.upsert_ranking(c, "default", "b", 988.0, 3, 1000.0)
    db.upsert_ranking(c, "default", "a", 1020.0, 4, 1001.0)  # update
    rank = db.get_ranking(c, "default")
    assert rank[0] == {"model": "a", "elo": 1020, "games": 4}
    assert rank[1] == {"model": "b", "elo": 988, "games": 3}
