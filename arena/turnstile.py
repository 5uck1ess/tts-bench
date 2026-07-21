"""Cloudflare Turnstile siteverify (bot gate). Empty secret => disabled (dev)."""

import httpx

SITEVERIFY = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


async def verify(secret: str, token: str, remoteip: str | None,
                 client: httpx.AsyncClient | None) -> bool:
    """True iff the Turnstile token is valid. Empty secret disables the gate."""
    if not secret:
        return True
    if not token or client is None:
        return False
    try:
        data = {"secret": secret, "response": token}
        if remoteip:
            data["remoteip"] = remoteip
        resp = await client.post(SITEVERIFY, data=data, timeout=5.0)
        return bool(resp.json().get("success"))
    except (httpx.HTTPError, ValueError):
        return False
