"""Rate-limit eligibility for the live Elo (never API rejection). Pure."""

BURST_MIN_INTERVAL_S = 3.0
DAILY_SOFT_CAP = 500
# Per-IP distinct-token cap feeding the live Elo. Deliberately generous: behind
# NAT / CGNAT / a shared Discord link, many honest voters egress one IP, and a
# too-low cap silently drops their votes from the board. IP is only a SECONDARY,
# spoofable signal (the persistent `voter` token is the primary Sybil control),
# so this catches blatant single-IP localStorage-clear farming, not households.
IP_MAX_TOKENS_PER_HOUR = 8


def within_burst(last_vote_ts: float | None, now: float) -> bool:
    """True if this vote is too soon after the token's previous one (<3 s)."""
    if last_vote_ts is None:
        return False
    return (now - last_vote_ts) < BURST_MIN_INTERVAL_S


def under_daily_cap(today_count: int) -> bool:
    """True if the token is still under its daily soft cap."""
    return today_count < DAILY_SOFT_CAP


def ip_under_token_cap(distinct_tokens_last_hour: int) -> bool:
    """True if the IP has not exceeded its active-token-per-hour cap."""
    return distinct_tokens_last_hour <= IP_MAX_TOKENS_PER_HOUR


def rate_eligible(*, last_vote_ts: float | None, now: float, today_count: int,
                  ip_tokens_last_hour: int) -> bool:
    """Combined: eligible for the live Elo on rate grounds."""
    return (
        not within_burst(last_vote_ts, now)
        and under_daily_cap(today_count)
        and ip_under_token_cap(ip_tokens_last_hour)
    )
