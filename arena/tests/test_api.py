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
    # post-vote reveal metadata; cloning slugs deliberately absent -> slug fallback
    "models": {
        "alpha": {"name": "Alpha TTS 2", "url": "https://hf.test/org/alpha-2"},
        "beta": {"name": "Beta-TTS", "url": "https://hf.test/org/beta"},
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


def _vote_body(d, voter="rev-voter", mode="default", choice="left"):
    return {"voter": voter, "mode": mode, "prompt_id": d["prompt_id"],
            "left_id": d["left_id"], "right_id": d["right_id"], "choice": choice,
            "dwell_ms": 3000, "both_played": True, "turnstile_token": "x",
            "pair_nonce": d["pair_nonce"]}


@pytest.mark.parametrize("state,stored,clean", [
    ("ok", 1, True), ("failed", 0, False), ("unreachable", 2, True),
])
def test_turnstile_state_controls_clean_vote_and_storage(client, monkeypatch, state, stored, clean):
    import arena.app as appmod

    async def verify_state(secret, token, remoteip, http_client):
        assert token == "x"
        assert http_client is not None
        return state

    monkeypatch.setattr(appmod.turnstile, "verify_state", verify_state)
    d = client.get("/api/next?mode=default").json()
    response = client.post("/api/vote", json=_vote_body(d))
    assert response.status_code == 200
    assert response.json()["clean"] is clean
    row = appmod._conn.execute("SELECT turnstile_ok, elo_clean FROM votes").fetchone()
    assert row["turnstile_ok"] == stored
    assert row["elo_clean"] == int(clean)
    ranking = client.get("/api/ranking?mode=default").json()
    assert ranking["votes"] == int(clean)
    assert all(r["games"] == int(clean) for r in ranking["ranking"])


def test_vote_response_reveals_names_and_urls(client):
    # pre-vote payload stays blind...
    d = client.get("/api/next?mode=default").json()
    assert "reveal" not in d and "Alpha TTS 2" not in str(d)
    # ...the post-vote response carries both models' display names + links
    rev = client.post("/api/vote", json=_vote_body(d)).json()["reveal"]
    names = {rev["left"]["name"], rev["right"]["name"]}
    assert names == {"Alpha TTS 2", "Beta-TTS"}
    urls = {rev["left"]["url"], rev["right"]["url"]}
    assert urls == {"https://hf.test/org/alpha-2", "https://hf.test/org/beta"}


def test_vote_reveal_falls_back_to_slug_without_meta(client):
    # cloning slugs are not in the manifest's models map -> name=slug, url=None
    d = client.get("/api/next?mode=cloning").json()
    rev = client.post("/api/vote", json=_vote_body(d, mode="cloning")).json()["reveal"]
    assert {rev["left"]["name"], rev["right"]["name"]} == {"echo", "indextts"}
    assert rev["left"]["url"] is None and rev["right"]["url"] is None


def test_rejected_vote_carries_no_reveal(client):
    d = client.get("/api/next?mode=default").json()
    body = _vote_body(d)
    body["pair_nonce"] = "forged"
    r = client.post("/api/vote", json=body)
    assert r.status_code == 400 and "reveal" not in r.json()


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


def test_hosted_pool_is_its_own_lens(tmp_path, monkeypatch):
    """A hosted-default block becomes its own mode: pairs only from hosted clips, and the
    ranking carries display names."""
    import copy
    man = copy.deepcopy(MANIFEST)
    man["modes"]["hosted-default"] = {"reference_url": None, "clips": [
        {"model": "hosted:vui_luna", "prompt": 1, "url": "https://gh.test/tts-bench/hosted/vui_luna/default/p1_t1.wav"},
        {"model": "hosted:rime_coda", "prompt": 1, "url": "https://gh.test/tts-bench/hosted/rime_coda/default/p1_t1.wav"},
    ]}
    man["models"]["hosted:vui_luna"] = {"name": "VUI Labs Luna", "url": None}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(man), encoding="utf-8")
    monkeypatch.setenv("ARENA_MANIFEST", str(mpath))
    monkeypatch.setenv("ARENA_DB", str(tmp_path / "arena.db"))
    monkeypatch.setenv("TURNSTILE_SECRET", "")
    monkeypatch.setenv("HMAC_SECRET", "test-hmac")
    import arena.app as appmod
    import importlib
    importlib.reload(appmod)
    with TestClient(appmod.app) as c:
        assert '"hosted-default"' in c.get("/").text
        for _ in range(5):
            d = c.get("/api/next?mode=hosted-default").json()
            urls = {c.get(d["left_url"], follow_redirects=False).headers["location"],
                    c.get(d["right_url"], follow_redirects=False).headers["location"]}
            assert all("/hosted/" in u for u in urls)
        names = {r["name"] for r in c.get("/api/ranking?mode=hosted-default").json()["ranking"]}
        assert names == {"VUI Labs Luna", "hosted:rime_coda"}


def _mixed_client(tmp_path, monkeypatch, rate):
    import copy
    man = copy.deepcopy(MANIFEST)
    man["modes"]["hosted-default"] = {"reference_url": None, "clips": [
        {"model": "hosted:vui_luna", "prompt": 1, "url": "https://gh.test/tts-bench/hosted/vui_luna/default/p1_t1.wav"},
        {"model": "hosted:rime_coda", "prompt": 1, "url": "https://gh.test/tts-bench/hosted/rime_coda/default/p1_t1.wav"},
    ]}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(man), encoding="utf-8")
    monkeypatch.setenv("ARENA_MANIFEST", str(mpath))
    monkeypatch.setenv("ARENA_DB", str(tmp_path / "arena.db"))
    monkeypatch.setenv("TURNSTILE_SECRET", "")
    monkeypatch.setenv("HMAC_SECRET", "test-hmac")
    monkeypatch.setenv("ARENA_MIX_RATE", rate)
    import arena.app as appmod
    import importlib
    importlib.reload(appmod)
    return appmod


def test_mixed_round_pairs_hosted_with_local_and_skips_elo(tmp_path, monkeypatch):
    appmod = _mixed_client(tmp_path, monkeypatch, "1")
    with TestClient(appmod.app) as c:
        d = c.get("/api/next?mode=default").json()
        assert d["vote_mode"] == "mixed-default"
        urls = [c.get(d[k], follow_redirects=False).headers["location"] for k in ("left_url", "right_url")]
        assert sorted("/hosted/" in u for u in urls) == [False, True]   # one hosted, one local
        before = c.get("/api/ranking?mode=default").json()
        body = {"voter": "mix-voter", "mode": "mixed-default", "choice": "left",
                "prompt_id": d["prompt_id"], "left_id": d["left_id"], "right_id": d["right_id"],
                "pair_nonce": d["pair_nonce"], "dwell_ms": 5000, "both_played": True}
        r = c.post("/api/vote", json=body).json()
        assert r["ok"] and r["clean"]
        assert {r["reveal"]["left"]["model"], r["reveal"]["right"]["model"]} & {"hosted:vui_luna", "hosted:rime_coda"}
        assert c.get("/api/ranking?mode=default").json() == before          # local board untouched
        assert c.get("/api/ranking?mode=hosted-default").json()["votes"] == 0
        with appmod._db_lock:
            modes = [row[0] for row in appmod._conn.execute("SELECT mode FROM votes").fetchall()]
        assert modes == ["mixed-default"]


def test_mixed_rounds_off_at_zero_rate(tmp_path, monkeypatch):
    appmod = _mixed_client(tmp_path, monkeypatch, "0")
    with TestClient(appmod.app) as c:
        assert all(c.get("/api/next?mode=default").json()["vote_mode"] == "default" for _ in range(20))


def test_mixed_nonce_cannot_be_replayed_as_default(tmp_path, monkeypatch):
    appmod = _mixed_client(tmp_path, monkeypatch, "1")
    with TestClient(appmod.app) as c:
        d = c.get("/api/next?mode=default").json()
        body = {"voter": "v", "mode": "default", "choice": "left", "prompt_id": d["prompt_id"],
                "left_id": d["left_id"], "right_id": d["right_id"], "pair_nonce": d["pair_nonce"],
                "dwell_ms": 5000, "both_played": True}
        assert c.post("/api/vote", json=body).status_code == 400   # signed for mixed-default only
