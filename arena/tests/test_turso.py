import pytest
from arena import turso


def test_connect_requires_url():
    with pytest.raises(ValueError):
        turso.connect("", "token")


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
