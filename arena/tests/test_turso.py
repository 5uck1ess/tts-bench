import pytest
from arena import turso


def test_connect_requires_url():
    with pytest.raises(ValueError):
        turso.connect("", "token")


def test_http_url_rewrites_websocket_schemes():
    # libsql:// and wss:// -> https:// (host preserved); plain ws:// -> http://.
    assert turso._http_url("libsql://x.turso.io") == "https://x.turso.io"
    assert turso._http_url("wss://x.turso.io") == "https://x.turso.io"
    assert turso._http_url("ws://localhost:8080") == "http://localhost:8080"
    assert turso._http_url("https://x.turso.io") == "https://x.turso.io"
    assert turso._http_url("file:local.db") == "file:local.db"


def test_connect_forces_http_transport(monkeypatch):
    # connect() must hand libsql-client an https:// URL so it uses the HTTP hrana
    # transport, not the websocket one that current Turso rejects at handshake.
    captured = {}

    class _FakeModule:
        @staticmethod
        def create_client_sync(url, auth_token):
            captured["url"] = url
            captured["token"] = auth_token
            return _FakeClient()

    monkeypatch.setattr(turso, "_import_libsql", lambda: _FakeModule)
    turso.connect("libsql://tts-arena-5uck1ess.aws-us-east-1.turso.io", "tok")
    assert captured["url"] == "https://tts-arena-5uck1ess.aws-us-east-1.turso.io"
    assert captured["token"] == "tok"


def test_connect_reports_missing_dependency_clearly(monkeypatch):
    # Simulate libsql-client not installed: connect should raise a helpful error.
    monkeypatch.setattr(turso, "_import_libsql", lambda: (_ for _ in ()).throw(
        ImportError("No module named 'libsql_client'")))
    with pytest.raises(RuntimeError) as ei:
        turso.connect("libsql://db.turso.io", "tok")
    assert "libsql-client" in str(ei.value)


class _FakeRS:
    columns = []
    rows = []


class _FakeClient:
    """Records every statement the facade issues, to pin executescript splitting."""
    def __init__(self):
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append(sql)
        return _FakeRS()


def test_executescript_splits_into_non_empty_statements():
    fc = _FakeClient()
    conn = turso.TursoConnection(fc)
    conn.executescript("CREATE TABLE a (x);\nCREATE INDEX i ON a(x);\n")
    assert len(fc.statements) == 2          # trailing blank not issued
    assert fc.statements[0].startswith("CREATE TABLE a")
    assert fc.statements[1].startswith("CREATE INDEX i")


def test_executescript_runs_full_db_schema():
    # The production schema must split cleanly through the Turso facade: 3 tables
    # (votes, tokens, rankings) + 3 indexes = 6 statements, none empty.
    from arena import db
    fc = _FakeClient()
    conn = turso.TursoConnection(fc)
    conn.executescript(db.SCHEMA)
    assert len(fc.statements) == 6
    assert any("CREATE TABLE IF NOT EXISTS votes" in s for s in fc.statements)
    assert any("CREATE TABLE IF NOT EXISTS rankings" in s for s in fc.statements)
    assert all(s.strip() for s in fc.statements)
