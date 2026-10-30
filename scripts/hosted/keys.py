"""API keys for the hosted collector.

Source of truth is 1Password (item "TTS API Keys"). Each new `op` process asks Tym to approve,
so the first run fetches every field once and caches it DPAPI-encrypted to his Windows account
(%LOCALAPPDATA%/tts-bench/hosted_keys.dpapi, 12 h TTL). Later runs read the cache and never touch
1Password. Keys are never printed, logged or written in plaintext; sidecars carry an 8-char
sha256 fingerprint at most. `--refresh-keys` (or deleting the file) forces a new fetch.

Tym has two 1Password accounts on the Windows box, so every `op` call needs --account.
"""

import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

OP_ACCOUNT = "my.1password.com"
OP_ITEM = "TTS API Keys"
OP_VAULT = "Home"
CACHE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "tts-bench" / "hosted_keys.dpapi"
TTL_S = 12 * 3600

# provider slug -> field label in the item, where Tym's label differs from the slug.
FIELD = {"soniox": "sonioux", "openrouter": "minimaxOPENROUTER"}

_cache = None


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _dpapi(data, protect):
    src = _Blob(len(data), ctypes.cast(ctypes.create_string_buffer(data, len(data)),
                                       ctypes.POINTER(ctypes.c_char)))
    dst = _Blob()
    fn = ctypes.windll.crypt32.CryptProtectData if protect else ctypes.windll.crypt32.CryptUnprotectData
    if not fn(ctypes.byref(src), None, None, None, None, 0, ctypes.byref(dst)):
        raise OSError("DPAPI call failed")
    try:
        return ctypes.string_at(dst.pbData, dst.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(dst.pbData)


def _fetch_op():
    out = subprocess.run(
        ["op", "item", "get", OP_ITEM, "--vault", OP_VAULT, "--account", OP_ACCOUNT,
         "--format", "json"],
        capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"op item get failed: {out.stderr.strip()[:300]}")
    fields = json.loads(out.stdout)["fields"]
    return {f["label"]: f["value"] for f in fields if f.get("label") and f.get("value")}


def _load(refresh=False):
    global _cache
    if _cache is not None and not refresh:
        return _cache
    if not refresh and CACHE.exists() and time.time() - CACHE.stat().st_mtime < TTL_S:
        try:
            _cache = json.loads(_dpapi(CACHE.read_bytes(), protect=False))
            return _cache
        except (OSError, ValueError):
            pass
    _cache = _fetch_op()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(_dpapi(json.dumps(_cache).encode(), protect=True))
    return _cache


def refresh():
    _load(refresh=True)


def get(provider):
    label = FIELD.get(provider, provider)
    key = _load().get(label)
    if not key:
        raise SystemExit(f"no '{label}' field in 1Password item '{OP_ITEM}'")
    return key.strip()


def fingerprint(key):
    return hashlib.sha256(key.encode()).hexdigest()[:8]
