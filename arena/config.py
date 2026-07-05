"""Environment-driven settings for the arena Space."""

import os
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEV_HMAC = "dev-insecure-hmac-secret-change-me"


@dataclass
class Settings:
    turnstile_secret: str
    turnstile_sitekey: str
    hmac_secret: str
    turso_url: str
    turso_token: str
    admin_token: str
    gh_pages_base: str
    manifest_path: str
    db_path: str
    langs: set | None
    nonce_max_age_s: int

    @property
    def hmac_secret_bytes(self) -> bytes:
        return self.hmac_secret.encode("utf-8")

    @property
    def use_turso(self) -> bool:
        return bool(self.turso_url)

    @property
    def missing_prod_secrets(self) -> list:
        """Anti-abuse secrets still at their insecure dev defaults. Non-empty in a
        production (Turso-backed) deploy means: nonces are signed with the PUBLIC
        _DEV_HMAC committed to the repo (pair tokens forgeable) and/or Turnstile
        verification is silently disabled (bot gate off)."""
        out = []
        if self.hmac_secret == _DEV_HMAC:
            out.append("HMAC_SECRET")
        if not self.turnstile_secret:
            out.append("TURNSTILE_SECRET")
        return out


def load_settings(env: dict | None = None) -> Settings:
    e = os.environ if env is None else env
    langs_raw = (e.get("ARENA_LANGS", "en") or "en").strip().lower()
    langs = None if langs_raw == "all" else set(p for p in langs_raw.split(",") if p)
    return Settings(
        turnstile_secret=e.get("TURNSTILE_SECRET", ""),
        turnstile_sitekey=e.get("TURNSTILE_SITEKEY", ""),
        hmac_secret=e.get("HMAC_SECRET", "") or _DEV_HMAC,
        turso_url=e.get("TURSO_URL", ""),
        turso_token=e.get("TURSO_TOKEN", ""),
        admin_token=e.get("ADMIN_TOKEN", ""),
        gh_pages_base=e.get("GH_PAGES_BASE", "https://5uck1ess.github.io/tts-bench/"),
        manifest_path=e.get("ARENA_MANIFEST", os.path.join(_HERE, "clips_manifest.json")),
        db_path=e.get("ARENA_DB", os.path.join(_HERE, "arena.db")),
        langs=langs,
        nonce_max_age_s=int(e.get("NONCE_MAX_AGE_S", "1800")),
    )
