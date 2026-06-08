import json
import os
import tempfile
import pytest
from fastapi.testclient import TestClient

MANIFEST = {
    "base_url": "https://gh.test/tts-bench/",
    "prompts": {"1": ["en", "hello world"]},
    "modes": {
        "default": {"reference_url": None, "clips": [
            {"model": "alpha", "prompt": 1, "url": "https://gh.test/tts-bench/windows-default/alpha_cpu_p1.wav"},
            {"model": "beta", "prompt": 1, "url": "https://gh.test/tts-bench/windows-default/beta_cpu_p1.wav"},
        ]},
        "cloning": {"reference_url": "https://gh.test/tts-bench/windows-cloning/_reference.wav", "clips": [
            {"model": "echo", "prompt": 1, "url": "https://gh.test/tts-bench/windows-cloning/echo_cuda_p1.wav"},
            {"model": "indextts", "prompt": 1, "url": "https://gh.test/tts-bench/windows-cloning/indextts_cuda_p1.wav"},
        ]},
    },
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(MANIFEST), encoding="utf-8")
    monkeypatch.setenv("ARENA_MANIFEST", str(mpath))
    monkeypatch.setenv("ARENA_DB", str(tmp_path / "arena.db"))
    monkeypatch.setenv("TURNSTILE_SECRET", "")      # disabled in tests
    monkeypatch.setenv("HMAC_SECRET", "test-hmac")
    monkeypatch.setenv("ADMIN_TOKEN", "letmein")
    import arena.app as appmod
    import importlib
    importlib.reload(appmod)
    with TestClient(appmod.app) as c:
        yield c


def test_index_serves_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Blind vote" in r.text or "blind" in r.text.lower()


def test_next_returns_signed_pair(client):
    r = client.get("/api/next?mode=default")
    assert r.status_code == 200
    d = r.json()
    assert {"token", "prompt_id", "left_url", "right_url", "pair_nonce"} <= set(d)
    assert d["left_url"].startswith("/clip/")   # opaque, no model name
    assert "alpha" not in d["left_url"] and "beta" not in d["left_url"]


def test_clip_redirects_to_ghpages(client):
    d = client.get("/api/next?mode=default").json()
    cid = d["left_url"].split("/clip/")[1]
    r = client.get(f"/clip/{cid}", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"].startswith("https://gh.test/tts-bench/")


def test_vote_clean_path_records_and_updates_elo(client):
    d = client.get("/api/next?mode=default").json()
    body = {"token": "voter-1", "session_id": "s", "mode": "default",
            "prompt_id": d["prompt_id"], "left_id": d["left_id"], "right_id": d["right_id"],
            "choice": "left", "dwell_ms": 3000, "both_played": True,
            "turnstile_token": "x", "pair_nonce": d["pair_nonce"]}
    r = client.post("/api/vote", json=body)
    assert r.status_code == 200 and r.json()["ok"] is True
    rank = client.get("/api/ranking?mode=default").json()["ranking"]
    winner = [x for x in rank if x["games"] > 0]
    assert winner and winner[0]["elo"] > 1000  # left model gained


def test_vote_replayed_nonce_rejected(client):
    d = client.get("/api/next?mode=default").json()
    body = {"token": "voter-2", "mode": "default", "prompt_id": d["prompt_id"],
            "left_id": d["left_id"], "right_id": d["right_id"], "choice": "left",
            "dwell_ms": 3000, "both_played": True, "turnstile_token": "x",
            "pair_nonce": d["pair_nonce"]}
    assert client.post("/api/vote", json=body).json()["ok"] is True
    # same nonce again -> logged-but-not-clean / rejected as replay
    r2 = client.post("/api/vote", json=body)
    assert r2.json()["ok"] is False


def test_vote_dwell_too_short_logs_but_not_clean(client):
    d = client.get("/api/next?mode=default").json()
    body = {"token": "voter-3", "mode": "default", "prompt_id": d["prompt_id"],
            "left_id": d["left_id"], "right_id": d["right_id"], "choice": "left",
            "dwell_ms": 200, "both_played": True, "turnstile_token": "x",
            "pair_nonce": d["pair_nonce"]}
    r = client.post("/api/vote", json=body)
    assert r.json()["ok"] is True            # accepted + logged
    assert r.json()["clean"] is False        # but not scored
    rank = client.get("/api/ranking?mode=default").json()["ranking"]
    assert all(x["games"] == 0 for x in rank)  # no Elo movement


def test_admin_flag_requires_token(client):
    assert client.post("/admin/flag_token", json={"token": "x"}).status_code == 401
    r = client.post("/admin/flag_token",
                    json={"token": "x"}, headers={"X-Admin-Token": "letmein"})
    assert r.status_code == 200


def test_reference_redirects_in_cloning(client):
    r = client.get("/reference?mode=cloning", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"].endswith("/windows-cloning/_reference.wav")


def test_vote_voter_field_lands_in_token_column(client):
    import arena.app as appmod
    d = client.get("/api/next?mode=default").json()
    body = {"voter": "real-voter", "mode": "default", "prompt_id": d["prompt_id"],
            "left_id": d["left_id"], "right_id": d["right_id"], "choice": "left",
            "dwell_ms": 3000, "both_played": True, "turnstile_token": "x",
            "pair_nonce": d["pair_nonce"]}
    assert client.post("/api/vote", json=body).json()["ok"] is True
    cur = appmod._conn.execute("SELECT token FROM votes ORDER BY id DESC LIMIT 1")
    assert cur.fetchone()["token"] == "real-voter"   # NOT collapsed to "anon"


def test_distinct_voters_each_get_a_clean_vote(client):
    # Two different voters voting back-to-back must BOTH score clean. If the token
    # collapsed to a shared "anon", the second would be burst-rate-limited (not clean).
    # No sleep needed: the per-issuance nonce salt makes two same-second /api/next
    # calls for the same pair mint DISTINCT nonces, so the 2nd vote is not a replay.
    results = []
    for v in ("alice", "bob"):
        d = client.get("/api/next?mode=default").json()
        body = {"voter": v, "mode": "default", "prompt_id": d["prompt_id"],
                "left_id": d["left_id"], "right_id": d["right_id"], "choice": "left",
                "dwell_ms": 3000, "both_played": True, "turnstile_token": "x",
                "pair_nonce": d["pair_nonce"]}
        results.append(client.post("/api/vote", json=body).json()["clean"])
    assert results == [True, True]
