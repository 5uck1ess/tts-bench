import logging

import httpx
import pytest
from arena.turnstile import verify


def _client(payload, capture=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["url"] = str(request.url)
            capture["body"] = request.content.decode()
        return httpx.Response(200, json=payload)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_success():
    async with _client({"success": True}) as client:
        assert await verify("secret", "tok", "1.2.3.4", client) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["invalid-input-response", "invalid-input-secret"])
async def test_failure(caplog, code):
    async with _client({"success": False, "error-codes": [code]}) as client:
        assert await verify("secret", "tok", "1.2.3.4", client) is False
    assert any(record.name == "arena.turnstile" and record.levelno == logging.WARNING
               and code in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_empty_secret_disables_and_passes():
    # no client call should be needed
    assert await verify("", "tok", "1.2.3.4", client=None) is True


@pytest.mark.asyncio
async def test_network_error_is_false(caplog):
    def boom(request):
        raise httpx.ConnectError("down")
    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as client:
        assert await verify("secret", "tok", "1.2.3.4", client) is False
    assert any(record.name == "arena.turnstile" and record.levelno == logging.WARNING
               and record.exc_info and isinstance(record.exc_info[1], httpx.ConnectError)
               for record in caplog.records)
    assert "down" in caplog.text


@pytest.mark.asyncio
async def test_invalid_json_is_false_and_logged(caplog):
    def handler(request):
        return httpx.Response(200, text="not JSON")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await verify("secret", "tok", "1.2.3.4", client) is False
    assert any(record.name == "arena.turnstile" and record.levelno == logging.WARNING
               and record.exc_info and isinstance(record.exc_info[1], ValueError)
               for record in caplog.records)


@pytest.mark.asyncio
async def test_remote_ip_is_omitted_when_unknown():
    capture = {}
    async with _client({"success": True}, capture) as client:
        assert await verify("secret", "tok", None, client) is True
    assert "remoteip" not in capture["body"]


@pytest.mark.asyncio
async def test_remote_ip_is_forwarded_when_provided():
    capture = {}
    async with _client({"success": True}, capture) as client:
        assert await verify("secret", "tok", "1.2.3.4", client) is True
    assert "remoteip=1.2.3.4" in capture["body"]
