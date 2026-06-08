from arena.ratelimit import (BURST_MIN_INTERVAL_S, DAILY_SOFT_CAP,
                             IP_MAX_TOKENS_PER_HOUR, within_burst,
                             under_daily_cap, ip_under_token_cap, rate_eligible)


def test_burst():
    assert within_burst(last_vote_ts=None, now=100.0) is False         # first vote, no prior
    assert within_burst(last_vote_ts=100.0, now=100.5) is True          # 0.5s -> too fast
    assert within_burst(last_vote_ts=100.0, now=100.0 + BURST_MIN_INTERVAL_S) is False


def test_daily_cap():
    assert under_daily_cap(0) is True
    assert under_daily_cap(DAILY_SOFT_CAP - 1) is True
    assert under_daily_cap(DAILY_SOFT_CAP) is False


def test_ip_token_cap():
    assert ip_under_token_cap(1) is True
    assert ip_under_token_cap(IP_MAX_TOKENS_PER_HOUR) is True
    assert ip_under_token_cap(IP_MAX_TOKENS_PER_HOUR + 1) is False


def test_rate_eligible_requires_all():
    assert rate_eligible(last_vote_ts=50.0, now=60.0, today_count=10,
                         ip_tokens_last_hour=1) is True
    assert rate_eligible(last_vote_ts=59.9, now=60.0, today_count=10,
                         ip_tokens_last_hour=1) is False   # burst
    assert rate_eligible(last_vote_ts=50.0, now=60.0, today_count=DAILY_SOFT_CAP,
                         ip_tokens_last_hour=1) is False    # daily
    assert rate_eligible(last_vote_ts=50.0, now=60.0, today_count=10,
                         ip_tokens_last_hour=99) is False   # ip
