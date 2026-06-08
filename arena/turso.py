"""Production datastore adapter: run db.py's SQLite SQL against Turso/libSQL.

libsql-client is SQLite-compatible at the SQL level, so the statements in
arena/db.py run unchanged. This adapter wraps a libSQL client in the same
execute/commit surface sqlite3.Connection exposes (the subset db.py uses).
Lazy import: nothing here loads libsql-client until connect() is called.
"""


def _import_libsql():
    import libsql_client  # noqa: F401  (isolated for monkeypatch in tests)
    return libsql_client


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class TursoConnection:
    """Minimal sqlite3-compatible facade over a libSQL sync client.

    Supports the calls arena/db.py makes: execute(sql, params) -> cursor with
    fetchone/fetchall returning Row-like mappings, executescript(sql), commit().
    """

    def __init__(self, client):
        self._client = client

    def execute(self, sql: str, params=()):
        rs = self._client.execute(sql, list(params))
        cols = rs.columns
        rows = [{c: v for c, v in zip(cols, row)} for row in rs.rows]
        return _Cursor(rows)

    def executescript(self, script: str):
        for stmt in (s.strip() for s in script.split(";")):
            if stmt:
                self._client.execute(stmt)

    def commit(self):
        pass  # libSQL autocommits each statement


def connect(url: str, token: str) -> TursoConnection:
    """Open a Turso connection. Raises RuntimeError with a fix hint if the
    libsql-client dependency is missing."""
    if not url:
        raise ValueError("Turso URL is required")
    try:
        libsql_client = _import_libsql()
    except ImportError as e:
        raise RuntimeError(
            "Turso backend requested but libsql-client is not installed. "
            "Add `libsql-client` to arena/requirements.txt (it ships in the "
            f"Space image). Original error: {e}") from e
    client = libsql_client.create_client_sync(url=url, auth_token=token)
    return TursoConnection(client)
