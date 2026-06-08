"""Pair nonce: HMAC-signed proof that the server served this exact pair.

Blocks forged votes (pairs never served) and replays beyond a max age. The
nonce travels to the client in /api/next and is echoed back on the vote.
"""

import hashlib
import hmac


def pair_fields(mode: str, left_id: str, right_id: str, prompt_id: int) -> str:
    """Canonical, side-order-sensitive identity string for a served pair."""
    return f"{mode}|{left_id}|{right_id}|{prompt_id}"


def make_nonce(secret: bytes, fields: str, ts: int) -> str:
    """Return ``"{ts}.{hexsig}"`` binding ``fields`` to issue time ``ts``."""
    msg = f"{ts}.{fields}".encode("utf-8")
    sig = hmac.new(secret, msg, hashlib.sha256).hexdigest()
    return f"{ts}.{sig}"


def verify_nonce(secret: bytes, nonce: str, fields: str, now: int,
                 max_age_s: int) -> bool:
    """True iff ``nonce`` is a valid, unexpired signature over ``fields``."""
    if not nonce or "." not in nonce:
        return False
    ts_str, _, sig = nonce.partition(".")
    try:
        ts = int(ts_str)
    except ValueError:
        return False
    if now < ts or now - ts > max_age_s:  # expired or future-skewed
        return False
    expected = make_nonce(secret, fields, ts)
    return hmac.compare_digest(nonce, expected)
