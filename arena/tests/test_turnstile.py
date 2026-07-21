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
async def test_failure():
    async with _client({"success": False, "error-codes": ["invalid-input-response"]}) as client:
        assert await verify("secret", "tok", "1.2.3.4", client) is False


@pytest.mark.asyncio
async def test_empty_secret_disables_and_passes():
    # no client call should be needed
    assert await verify("", "tok", "1.2.3.4", client=None) is True


@pytest.mark.asyncio
async def test_network_error_is_false():
    def boom(request):
        raise httpx.ConnectError("down")
    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as client:
        assert await verify("secret", "tok", "1.2.3.4", client) is False


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
