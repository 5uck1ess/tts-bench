from arena.gates import DWELL_FLOOR_MS, passes_clean_gate

OK = dict(both_played=True, dwell_ms=1500, turnstile_ok=True,
          nonce_ok=True, token_flagged=False)


def test_all_pass_is_clean():
    assert passes_clean_gate(**OK) is True
    assert DWELL_FLOOR_MS == 1500


def test_each_failure_blocks():
    assert passes_clean_gate(**{**OK, "both_played": False}) is False
    assert passes_clean_gate(**{**OK, "dwell_ms": 1499}) is False
    assert passes_clean_gate(**{**OK, "turnstile_ok": False}) is False
    assert passes_clean_gate(**{**OK, "nonce_ok": False}) is False
    assert passes_clean_gate(**{**OK, "token_flagged": True}) is False


def test_dwell_exactly_at_floor_passes():
    assert passes_clean_gate(**{**OK, "dwell_ms": DWELL_FLOOR_MS}) is True
