"""Cloudflare Turnstile siteverify (bot gate). Empty secret => disabled (dev)."""

import logging

import httpx

logger = logging.getLogger(__name__)

SITEVERIFY = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


async def verify_state(secret: str, token: str, remoteip: str | None,
                       client: httpx.AsyncClient | None) -> str:
    """Return ok, failed, or unreachable; retry transport failures once."""
    if not secret:
        return "ok"
    if not token or client is None:
        # Distinct from a Cloudflare rejection: the gate is ON and the client sent
        # nothing (widget blocked, script blocked, or a non-browser caller). Logged
        # separately so the Space logs tell the two failure classes apart.
        logger.warning("Turnstile gate enabled but no token submitted "
                       "(token=%s, client=%s)", bool(token), client is not None)
        return "failed"
    data = {"secret": secret, "response": token}
    if remoteip:
        data["remoteip"] = remoteip
    for attempt in range(2):
        try:
            resp = await client.post(SITEVERIFY, data=data, timeout=10.0)
            result = resp.json()
            if not isinstance(result, dict):
                raise ValueError("Turnstile siteverify response is not an object")
            if result.get("success"):
                return "ok"
            logger.warning("Turnstile siteverify failed: error-codes=%s",
                           result.get("error-codes", []))
            return "failed"
        except httpx.TransportError:
            logger.warning("Turnstile siteverify transport failure (attempt %s/2)",
                           attempt + 1, exc_info=True)
            if attempt == 1:
                return "unreachable"
        except (httpx.HTTPError, ValueError):
            logger.warning("Turnstile siteverify request/JSON failure", exc_info=True)
            return "unreachable"
    return "unreachable"  # unreachable in practice; never fall out returning None


async def verify(secret: str, token: str, remoteip: str | None,
                 client: httpx.AsyncClient | None) -> bool:
    """True iff the Turnstile token is valid. Empty secret disables the gate."""
    return await verify_state(secret, token, remoteip, client) == "ok"
