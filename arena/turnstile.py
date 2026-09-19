"""Cloudflare Turnstile siteverify (bot gate). Empty secret => disabled (dev)."""

import logging

import httpx

logger = logging.getLogger(__name__)

SITEVERIFY = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


async def verify(secret: str, token: str, remoteip: str | None,
                 client: httpx.AsyncClient | None) -> bool:
    """True iff the Turnstile token is valid. Empty secret disables the gate."""
    if not secret:
        return True
    if not token or client is None:
        # Distinct from a Cloudflare rejection: the gate is ON and the client sent
        # nothing (widget blocked, script blocked, or a non-browser caller). Logged
        # separately so the Space logs tell the two failure classes apart.
        logger.warning("Turnstile gate enabled but no token submitted "
                       "(token=%s, client=%s)", bool(token), client is not None)
        return False
    try:
        data = {"secret": secret, "response": token}
        if remoteip:
            data["remoteip"] = remoteip
        resp = await client.post(SITEVERIFY, data=data, timeout=5.0)
        result = resp.json()
        success = bool(result.get("success"))
        if not success:
            logger.warning("Turnstile siteverify failed: error-codes=%s",
                           result.get("error-codes", []))
        return success
    except (httpx.HTTPError, ValueError):
        logger.warning("Turnstile siteverify request/JSON failure", exc_info=True)
        return False
