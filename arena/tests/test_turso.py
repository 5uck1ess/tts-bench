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
