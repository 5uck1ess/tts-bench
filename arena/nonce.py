"""Pair nonce: HMAC-signed proof that the server served this exact pair.

Blocks forged votes (pairs never served) and replays beyond a max age. The
nonce travels to the client in /api/next and is echoed back on the vote.
"""

import hashlib
import hmac
import secrets


def pair_fields(mode: str, left_id: str, right_id: str, prompt_id: int) -> str:
    """Canonical, side-order-sensitive identity string for a served pair."""
    return f"{mode}|{left_id}|{right_id}|{prompt_id}"


def make_nonce(secret: bytes, fields: str, ts: int, jti: str | None = None) -> str:
    """Return ``"{ts}.{jti}.{hexsig}"`` binding ``fields`` to issue time ``ts``.

    ``jti`` is a per-issuance random salt (generated if not given) so two
    issuances of the SAME pair within the same one-second ``ts`` still produce
    distinct nonces — otherwise two honest voters served a hot pair in the same
    second would collide and the second would be falsely rejected as a replay.
    The salt rides inside the nonce, so it round-trips with ``pair_nonce`` and
    ``verify_nonce`` recovers it; no extra field is needed.
    """
    if jti is None:
        jti = secrets.token_hex(8)
    msg = f"{ts}.{jti}.{fields}".encode("utf-8")
    sig = hmac.new(secret, msg, hashlib.sha256).hexdigest()
    return f"{ts}.{jti}.{sig}"


def verify_nonce(secret: bytes, nonce: str, fields: str, now: int,
                 max_age_s: int) -> bool:
    """True iff ``nonce`` is a valid, unexpired signature over ``fields``."""
    if not nonce:
        return False
    parts = nonce.split(".")
    if len(parts) != 3:
        return False
    ts_str, jti, _sig = parts
    try:
        ts = int(ts_str)
    except ValueError:
        return False
    if now < ts or now - ts > max_age_s:  # expired or future-skewed
        return False
    expected = make_nonce(secret, fields, ts, jti)
    return hmac.compare_digest(nonce, expected)
