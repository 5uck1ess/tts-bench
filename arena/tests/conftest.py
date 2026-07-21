import asyncio
import inspect
import pytest


_DEPLOYMENT_ENV = (
    "ADMIN_TOKEN",
    "ARENA_DB",
    "ARENA_LANGS",
    "ARENA_MANIFEST",
    "GH_PAGES_BASE",
    "HMAC_SECRET",
    "NONCE_MAX_AGE_S",
    "TURNSTILE_SECRET",
    "TURNSTILE_SITEKEY",
    "TURSO_TOKEN",
    "TURSO_URL",
)


@pytest.fixture(autouse=True)
def isolate_deployment_environment(monkeypatch):
    """Keep tests independent of local/prod arena credentials.

    export_votes intentionally loads arena/.env for CLI convenience. Disabling
    dotenv here also prevents that process-global load from leaking into tests
    that reload arena.app later in the same pytest process.
    """
    for key in _DEPLOYMENT_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")


def pytest_collection_modifyitems(items):
    for item in items:
        if isinstance(item, pytest.Function) and inspect.iscoroutinefunction(item.function):
            item.add_marker("asyncio")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    func = pyfuncitem.obj
    if inspect.iscoroutinefunction(func):
        kwargs = {n: pyfuncitem.funcargs[n] for n in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(func(**kwargs))
        return True
