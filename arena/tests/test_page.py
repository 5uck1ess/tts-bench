import json
import importlib


def _render(monkeypatch, tmp_path):
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps({
        "base_url": "https://gh.test/x/", "prompts": {"1": ["en", "hi"]},
        "modes": {"default": {"reference_url": None, "clips": [
            {"model": "a", "prompt": 1, "url": "https://gh.test/x/windows-default/a_cpu_p1.wav"},
            {"model": "b", "prompt": 1, "url": "https://gh.test/x/windows-default/b_cpu_p1.wav"}]},
            "cloning": {"reference_url": None, "clips": []}},
    }), encoding="utf-8")
    monkeypatch.setenv("ARENA_MANIFEST", str(mpath))
    monkeypatch.setenv("ARENA_DB", str(tmp_path / "a.db"))
    monkeypatch.setenv("TURNSTILE_SITEKEY", "0xSITEKEY")
    import arena.app as appmod
    importlib.reload(appmod)
    from fastapi.testclient import TestClient
    with TestClient(appmod.app) as c:
        return c.get("/").text


def test_page_has_token_turnstile_and_gates(monkeypatch, tmp_path):
    html = _render(monkeypatch, tmp_path)
    assert "arena_token" in html                 # persistent anon identity
    assert "0xSITEKEY" in html                    # injected sitekey
    assert "turnstile" in html.lower()            # widget
    assert "both_played" in html                  # play gate field
    assert "dwell" in html.lower()                # dwell timer
    assert "pair_nonce" in html                   # nonce echo
    assert "/api/vote" in html and "/api/next" in html


def test_page_drops_v1_excluded_controls(monkeypatch, tmp_path):
    html = _render(monkeypatch, tmp_path)
    assert "switch rater" not in html.lower()     # no rater prompt
    assert "undo last" not in html.lower()        # no undo in public v1
