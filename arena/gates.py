"""Clean-vote gate: which logged votes feed the live Elo. Pure."""

DWELL_FLOOR_MS = 1500


def passes_clean_gate(*, both_played: bool, dwell_ms: int, turnstile_ok: bool,
                      nonce_ok: bool, token_flagged: bool) -> bool:
    """True iff a vote row qualifies for the live Elo. ALL conditions required."""
    return (
        both_played
        and dwell_ms >= DWELL_FLOOR_MS
        and turnstile_ok
        and nonce_ok
        and not token_flagged
    )
