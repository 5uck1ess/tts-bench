from arena.config import Settings, load_settings


def test_defaults_when_env_empty():
    s = load_settings(env={})
    assert isinstance(s, Settings)
    assert s.turnstile_secret == ""        # disabled in dev
    assert s.langs == {"en"}
    assert s.db_path.endswith("arena.db")
    assert s.turso_url == ""
    assert s.hmac_secret  # non-empty dev fallback


def test_env_overrides():
    s = load_settings(env={
        "TURNSTILE_SECRET": "sek", "TURNSTILE_SITEKEY": "site",
        "HMAC_SECRET": "hs", "TURSO_URL": "libsql://x", "TURSO_TOKEN": "tk",
        "ARENA_LANGS": "en,fr", "ADMIN_TOKEN": "admin",
        "GH_PAGES_BASE": "https://h.test/x/",
    })
    assert s.turnstile_secret == "sek"
    assert s.hmac_secret_bytes == b"hs"
    assert s.langs == {"en", "fr"}
    assert s.turso_url == "libsql://x"
    assert s.use_turso is True
    assert s.admin_token == "admin"


def test_langs_all_means_none():
    s = load_settings(env={"ARENA_LANGS": "all"})
    assert s.langs is None


def test_turso_requires_all_production_secrets():
    s = load_settings(env={"TURSO_URL": "libsql://db.example"})
    assert s.missing_prod_secrets == [
        "HMAC_SECRET", "TURNSTILE_SECRET", "TURNSTILE_SITEKEY", "TURSO_TOKEN"]


def test_turso_accepts_complete_production_secrets():
    s = load_settings(env={
        "TURSO_URL": "libsql://db.example",
        "TURSO_TOKEN": "db-token",
        "HMAC_SECRET": "hmac-secret",
        "TURNSTILE_SECRET": "turnstile-secret",
        "TURNSTILE_SITEKEY": "turnstile-sitekey",
    })
    assert s.missing_prod_secrets == []
