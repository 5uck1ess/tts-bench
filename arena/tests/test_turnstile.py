import logging

import httpx
import pytest
from arena.turnstile import SITEVERIFY, verify, verify_state


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


@pytest.mark.asyncio
async def test_state_disabled():
    assert await verify_state("", "", None, None) == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("token,has_client", [("", True), ("tok", False), ("", False)])
async def test_state_missing_input(token, has_client):
    def unexpected_request(request):
        pytest.fail("Missing input must not make a request")
    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected_request)) as client:
        assert await verify_state("secret", token, None,
                                  client if has_client else None) == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("success,expected", [(True, "ok"), (False, "failed")])
async def test_state_verdict_and_request(success, expected, caplog):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"success": success, "error-codes": ["invalid-input-response"]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await verify_state("secret", "tok", "1.2.3.4", client) == expected
    assert len(requests) == 1
    assert str(requests[0].url) == SITEVERIFY
    assert requests[0].content == b"secret=secret&response=tok&remoteip=1.2.3.4"
    assert set(requests[0].extensions["timeout"].values()) == {10.0}
    if not success:
        assert "invalid-input-response" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout])
@pytest.mark.parametrize("second_verdict", [None, True, False])
async def test_state_transport_retry(error, second_verdict, caplog):
    requests = []
    def handler(request):
        requests.append(request)
        if len(requests) == 1 or second_verdict is None:
            raise error("verifier unavailable")
        return httpx.Response(200, json={"success": second_verdict})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        expected = "unreachable" if second_verdict is None else "ok" if second_verdict else "failed"
        assert await verify_state("secret", "tok", None, client) == expected
    assert len(requests) == 2
    assert all(set(r.extensions["timeout"].values()) == {10.0} for r in requests)
    assert requests[0].content == requests[1].content == b"secret=secret&response=tok"
    assert any(r.exc_info and isinstance(r.exc_info[1], error) for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", ["not JSON", "null", "[]"])
async def test_state_bad_json_unreachable_without_retry(payload, caplog):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, text=payload)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await verify_state("secret", "tok", None, client) == "unreachable"
    assert len(requests) == 1
    assert any(r.exc_info and isinstance(r.exc_info[1], ValueError) for r in caplog.records)
