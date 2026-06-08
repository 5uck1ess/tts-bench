import pytest
from arena.elo import BASE_ELO, K_FACTOR, expected, apply_vote, derive


def test_expected_is_half_for_equal_ratings():
    assert expected(1000.0, 1000.0) == pytest.approx(0.5)


def test_apply_left_win_moves_ratings_by_half_k_at_parity():
    elo = {"a": 1000.0, "b": 1000.0}
    games = {"a": 0, "b": 0}
    apply_vote(elo, games, "a", "b", "left")
    assert elo["a"] == pytest.approx(1000.0 + K_FACTOR * 0.5)
    assert elo["b"] == pytest.approx(1000.0 - K_FACTOR * 0.5)
    assert games == {"a": 1, "b": 1}


def test_tie_at_parity_is_noop_on_ratings_but_counts_games():
    elo = {"a": 1000.0, "b": 1000.0}
    games = {"a": 0, "b": 0}
    apply_vote(elo, games, "a", "b", "tie")
    assert elo["a"] == pytest.approx(1000.0)
    assert elo["b"] == pytest.approx(1000.0)
    assert games == {"a": 1, "b": 1}


def test_derive_replays_votes_and_skips_bad_and_unknown():
    models = ["a", "b", "c"]
    votes = [
        ("a", "b", "left"),
        ("a", "b", "left"),
        ("a", "b", "bad"),      # excluded from Elo + games
        ("a", "zzz", "left"),   # unknown model -> skipped
    ]
    elo, games = derive(models, votes)
    assert elo["a"] > elo["b"]
    assert elo["c"] == pytest.approx(BASE_ELO)  # never played
    assert games == {"a": 2, "b": 2, "c": 0}


def test_derive_is_order_independent_in_total_games_but_deterministic():
    models = ["a", "b"]
    v = [("a", "b", "left"), ("b", "a", "right")]
    e1, _ = derive(models, v)
    e2, _ = derive(models, v)
    assert e1 == e2
