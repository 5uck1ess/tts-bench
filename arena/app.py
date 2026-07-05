"""FastAPI arena Space: blind 2AFC vote collector + live Elo.

Startup: load both inventories from the manifest, open the datastore, replay
clean votes into in-memory Elo per mode, start a 6 h re-derive loop. Audio is
served by gh-pages via opaque /clip/<id> 302 redirects (blindness preserved).
"""

import asyncio
import hashlib
import hmac
import threading
import time

import httpx
from fastapi import FastAPI, Request, Response, Header
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from . import db as dbmod
from . import elo as elomod
from . import gates
from . import ratelimit
from . import turnstile
from .config import load_settings
from .inventory import load_inventory
from .nonce import pair_fields, make_nonce, verify_nonce
from .pairing import choose_pair
import json
import os
import random

SETTINGS = load_settings()

# A Turso-backed deploy is production: refuse to start if the anti-abuse secrets
# are missing/defaulted. Failing open here means forgeable pair nonces (public
# dev HMAC) and a silently-disabled bot gate — worse than downtime.
if SETTINGS.use_turso and SETTINGS.missing_prod_secrets:
    raise RuntimeError(
        f"arena refusing to start: {', '.join(SETTINGS.missing_prod_secrets)} not set "
        "while TURSO_URL is configured. Set them as Space secrets (or unset TURSO_URL "
        "for a local dev run).")

RE_DERIVE_INTERVAL_S = 6 * 3600

with open(SETTINGS.manifest_path, encoding="utf-8") as f:
    _MANIFEST = json.load(f)

_PAGE_PATH = os.path.join(os.path.dirname(__file__), "page.html")
with open(_PAGE_PATH, encoding="utf-8") as f:
    _PAGE = f.read()

# slug -> {"name", "url"} for the post-vote reveal. Older manifests lack the key;
# _reveal_meta falls back to the raw slug so the endpoint never depends on it.
_MODEL_META = _MANIFEST.get("models", {})


def _reveal_meta(model: str) -> dict:
    meta = _MODEL_META.get(model, {})
    return {"model": model, "name": meta.get("name") or model, "url": meta.get("url")}


def _open_db():
    if SETTINGS.use_turso:
        from . import turso
        conn = turso.connect(SETTINGS.turso_url, SETTINGS.turso_token)
    else:
        conn = dbmod.connect(SETTINGS.db_path)
    dbmod.init_schema(conn)
    return conn


class ModeState:
    """In-memory Elo + pairing state for one lens, rebuilt from the DB."""

    def __init__(self, mode: str, inventory, conn):
        self.mode = mode
        self.inv = inventory
        self.lock = threading.Lock()
        self.rng = random.Random()
        self.elo = {}
        self.games = {}
        self.pair_count = {}
        self.rebuild(conn)

    def rebuild(self, conn):
        with _db_lock:
            votes = dbmod.clean_votes(conn, self.mode)
        elo, games = elomod.derive(self.inv.models, votes)
        pair_count = {}
        for left, right, choice in votes:
            if choice != "bad" and left in elo and right in elo:
                key = frozenset((left, right))
                pair_count[key] = pair_count.get(key, 0) + 1
        with self.lock:
            self.elo, self.games, self.pair_count = elo, games, pair_count


app = FastAPI()
_conn = _open_db()
_db_lock = threading.Lock()  # serializes all access to the shared sqlite _conn
_STATES = {}
for _m in ("default", "cloning"):
    _inv = load_inventory(_MANIFEST, _m, SETTINGS.langs)
    if _inv.by_prompt:
        _STATES[_m] = ModeState(_m, _inv, _conn)
_INITIAL = "default" if "default" in _STATES else next(iter(_STATES))


def _ip_hash(request: Request) -> str:
    # NOTE: X-Forwarded-For's leftmost entry is client-supplied and spoofable, so
    # the per-IP token cap is a BEST-EFFORT SECONDARY signal only (the persistent
    # `voter` token is the primary Sybil control). Behind a trusted proxy that
    # rewrites XFF, harden this to the platform's trusted client-IP header.
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?")
    ip = ip.split(",")[0].strip()
    return hashlib.sha256((ip + "|" + SETTINGS.hmac_secret).encode()).hexdigest()[:16]


@app.get("/", response_class=HTMLResponse)
def index():
    modes = list(_STATES)
    hasref = {m: bool(s.inv.reference_url) for m, s in _STATES.items()}
    counts = {m: len(s.inv.models) for m, s in _STATES.items()}
    html = (_PAGE.replace("__MODES__", json.dumps(modes))
                 .replace("__HASREF__", json.dumps(hasref))
                 .replace("__INITIAL_MODE__", json.dumps(_INITIAL))
                 .replace("__COUNTS__", json.dumps(counts))
                 .replace("__SITEKEY__", json.dumps(SETTINGS.turnstile_sitekey)))
    return HTMLResponse(html)


@app.get("/api/next")
def api_next(mode: str = _INITIAL):
    st = _STATES.get(mode) or _STATES[_INITIAL]
    with st.lock:
        prompt, left, right = choose_pair(st.inv.by_prompt, st.elo, st.games,
                                          st.pair_count, st.rng)
    left_id = st.inv.id_of[(left, prompt)]
    right_id = st.inv.id_of[(right, prompt)]
    ts = int(time.time())
    fields = pair_fields(st.mode, left_id, right_id, prompt)
    nonce = make_nonce(SETTINGS.hmac_secret_bytes, fields, ts)
    lang, text = st.inv.prompts.get(prompt, ("en", ""))
    with _db_lock:
        total = dbmod.clean_vote_count(_conn, st.mode)  # all good-faith votes (incl tie/bad)
    return {
        "token": nonce,            # legacy field name kept for page parity
        "pair_nonce": nonce,
        "prompt_id": prompt,
        "lang": lang,
        "text": text,
        "left_id": left_id,
        "right_id": right_id,
        "left_url": f"/clip/{left_id}",
        "right_url": f"/clip/{right_id}",
        "votes": total,
        "models": len(st.inv.models),
    }


def _url_for_clip(cid: str):
    for st in _STATES.values():
        if cid in st.inv.url_of:
            return st.inv.url_of[cid]
    return None


@app.get("/clip/{cid}")
def clip(cid: str):
    url = _url_for_clip(cid)
    if not url:
        return Response("no clip", status_code=404)
    return RedirectResponse(url, status_code=302)


@app.get("/reference")
def reference(mode: str = _INITIAL):
    st = _STATES.get(mode)
    if not st or not st.inv.reference_url:
        return Response("no reference", status_code=404)
    return RedirectResponse(st.inv.reference_url, status_code=302)


@app.get("/api/ranking")
def api_ranking(mode: str = _INITIAL):
    st = _STATES.get(mode) or _STATES[_INITIAL]
    with st.lock:
        rows = [{"model": m, "elo": round(st.elo[m]), "games": st.games[m]}
                for m in st.inv.models]
    rows.sort(key=lambda r: -r["elo"])
    with _db_lock:
        votes = dbmod.clean_vote_count(_conn, st.mode)  # all good-faith votes (incl tie/bad)
    return {"ranking": rows, "votes": votes}


@app.post("/api/vote")
async def api_vote(request: Request):
    body = await request.json()
    mode = body.get("mode", _INITIAL)
    st = _STATES.get(mode)
    if st is None:
        return JSONResponse({"ok": False, "error": "bad mode"}, status_code=400)
    choice = body.get("choice", "")
    if choice not in ("left", "right", "tie", "bad"):
        return JSONResponse({"ok": False, "error": "bad choice"}, status_code=400)

    left_id, right_id = body.get("left_id", ""), body.get("right_id", "")
    try:
        prompt_id = int(body.get("prompt_id", 0))
        dwell_ms = int(body.get("dwell_ms", 0))
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "bad field"}, status_code=400)
    pair_nonce = body.get("pair_nonce", "")
    # Persistent anonymous rater token. The page sends it as `voter`; `token` is
    # also accepted (test/back-compat). Unknown -> "anon" (a malformed client).
    token = body.get("voter") or body.get("token") or "anon"

    now = time.time()
    fields = pair_fields(mode, left_id, right_id, prompt_id)
    nonce_ok = verify_nonce(SETTINGS.hmac_secret_bytes, pair_nonce, fields,
                            int(now), SETTINGS.nonce_max_age_s)
    if not nonce_ok:
        return JSONResponse({"ok": False, "error": "bad nonce"}, status_code=400)

    # Resolve models from the opaque ids.
    left = right = None
    for (m, p), cid in st.inv.id_of.items():
        if cid == left_id and p == prompt_id:
            left = m
        if cid == right_id and p == prompt_id:
            right = m
    if left is None or right is None:
        return JSONResponse({"ok": False, "error": "unknown clip"}, status_code=400)

    # Turnstile (network) BEFORE taking the DB lock — never await under the lock.
    async with httpx.AsyncClient() as hc:
        ts_ok = await turnstile.verify(SETTINGS.turnstile_secret,
                                       body.get("turnstile_token", ""),
                                       _ip_hash(request), hc)

    both_played = bool(body.get("both_played"))
    ip_hash = _ip_hash(request)
    day_ago, hour_ago = now - 86400, now - 3600

    # All DB reads+writes for this vote are serialized so the replay check and the
    # insert are atomic (one shared sqlite connection across the threadpool).
    with _db_lock:
        if dbmod.nonce_seen(_conn, pair_nonce):
            return JSONResponse({"ok": False, "error": "replay"}, status_code=409)
        tok_state = dbmod.token_state(_conn, token)
        last_ts = tok_state["last_vote_ts"] if tok_state else None
        rate_ok = ratelimit.rate_eligible(
            last_vote_ts=last_ts, now=now,
            today_count=dbmod.daily_count(_conn, token, day_ago),
            ip_tokens_last_hour=dbmod.ip_distinct_tokens(_conn, ip_hash, hour_ago))
        token_flagged = dbmod.is_token_flagged(_conn, token)
        clean = gates.passes_clean_gate(
            both_played=both_played, dwell_ms=dwell_ms, turnstile_ok=ts_ok,
            nonce_ok=nonce_ok, token_flagged=token_flagged) and rate_ok
        row = {
            "ts": now, "token": token, "session_id": body.get("session_id", ""),
            "mode": mode, "prompt_id": prompt_id, "left_model": left, "right_model": right,
            "left_clip": left_id, "right_clip": right_id, "choice": choice,
            "dwell_ms": dwell_ms, "both_played": int(both_played),
            "turnstile_ok": int(ts_ok), "pair_nonce": pair_nonce, "gold_pair_id": None,
            "ua": request.headers.get("user-agent", "")[:300], "ip_hash": ip_hash,
            "elo_clean": int(clean), "rate_ok": int(rate_ok),
        }
        dbmod.insert_vote(_conn, row)
        dbmod.bump_token(_conn, token, now)

    if clean and choice != "bad":
        with st.lock:
            elomod.apply_vote(st.elo, st.games, left, right, choice)
            key = frozenset((left, right))
            st.pair_count[key] = st.pair_count.get(key, 0) + 1

    # Reveal AFTER the vote is durably recorded — the voter can no longer change
    # it, so showing names here doesn't break the blind for the decided pair.
    return {"ok": True, "clean": bool(clean),
            "reveal": {"left": _reveal_meta(left), "right": _reveal_meta(right)}}


@app.post("/admin/flag_token")
def admin_flag(body: dict, x_admin_token: str = Header(default="")):
    if not SETTINGS.admin_token or not hmac.compare_digest(x_admin_token, SETTINGS.admin_token):
        return JSONResponse({"ok": False}, status_code=401)
    with _db_lock:
        dbmod.flag_token(_conn, body.get("token", ""))
    for st in _STATES.values():       # token sweep -> self-healing re-derive
        st.rebuild(_conn)
    return {"ok": True}


@app.on_event("startup")
async def _start_rederive_loop():
    async def loop():
        while True:
            await asyncio.sleep(RE_DERIVE_INTERVAL_S)
            for st in _STATES.values():
                st.rebuild(_conn)
    asyncio.create_task(loop())
