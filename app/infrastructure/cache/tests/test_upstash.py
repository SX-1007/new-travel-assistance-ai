"""Upstash REST client with a fake session and clock: request, errors, cooldowns."""

import pytest
import requests

from app.infrastructure.cache.upstash import (
    AUTH_FAILURE_COOLDOWN,
    COMMAND_LIMIT_COOLDOWN,
    TRANSPORT_FAILURE_COOLDOWN,
    CacheCoolingDown,
    CacheUnavailableError,
    UpstashRestClient,
)


URL = "https://example-12345.upstash.io"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, invalid_json=False):
        self.status_code = status_code
        self._payload = payload
        self._invalid_json = invalid_json

    def json(self):
        if self._invalid_json:
            raise ValueError("not JSON")
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def client(*responses):
    session, clock = FakeSession(*responses), Clock()
    return UpstashRestClient(url=f" {URL}/ ", token=" tok ", session=session, clock=clock), session, clock


def test_posts_the_command_with_the_token_only_in_a_header():
    upstash, session, _ = client(FakeResponse(payload={"result": "OK"}))

    assert upstash.execute(["SET", "k", "v", "EX", 3600]) == "OK"

    url, kwargs = session.calls[0]
    assert url == URL
    assert kwargs["json"] == ["SET", "k", "v", "EX", "3600"]
    assert kwargs["headers"] == {"Authorization": "Bearer tok"}
    assert max(kwargs["timeout"]) <= 5
    assert "tok" not in repr(upstash) and "upstash.io" not in repr(upstash)


@pytest.mark.parametrize("result", [None, 0, "text", ["cached", "{}"]])
def test_returns_any_result_including_null(result):
    upstash, _, _ = client(FakeResponse(payload={"result": result}))

    assert upstash.execute(["GET", "k"]) == result


def test_rejects_a_bad_url_or_empty_token():
    for url in ("http://example.upstash.io", "example.upstash.io", "https://"):
        with pytest.raises(ValueError):
            UpstashRestClient(url=url, token="t")
    with pytest.raises(ValueError):
        UpstashRestClient(url=URL, token=" ")


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(payload={"error": "ERR wrong number of arguments"}, status_code=400),
        FakeResponse(payload={"error": "OOM command not allowed when used memory > 'maxmemory'"}, status_code=400),
        FakeResponse(payload={"error": "ERR max request size exceeded"}, status_code=400),
        FakeResponse(invalid_json=True),
        FakeResponse(payload={"no": "result"}),
        FakeResponse(status_code=500, payload={"result": "OK"}),
    ],
)
def test_failed_commands_raise_without_a_cooldown(response):
    upstash, session, _ = client(response)

    with pytest.raises(CacheUnavailableError) as raised:
        upstash.execute(["GET", "k"])

    assert not isinstance(raised.value, CacheCoolingDown)
    assert upstash.seconds_until_available() == 0


@pytest.mark.parametrize(
    ("failure", "cooldown"),
    [
        (requests.ConnectionError(), TRANSPORT_FAILURE_COOLDOWN),
        (requests.ReadTimeout(), TRANSPORT_FAILURE_COOLDOWN),
        (FakeResponse(401, payload={"error": "WRONGPASS invalid or missing auth token."}), AUTH_FAILURE_COOLDOWN),
        (
            FakeResponse(400, payload={"error": "ERR max requests limit exceeded. Limit: 500000, Usage: 500000"}),
            COMMAND_LIMIT_COOLDOWN,
        ),
        (FakeResponse(400, payload={"error": "ERR max daily request limit exceeded"}), COMMAND_LIMIT_COOLDOWN),
    ],
)
def test_repeating_failures_stop_commands_until_the_cooldown_ends(failure, cooldown):
    upstash, session, clock = client(failure, FakeResponse(payload={"result": "PONG"}))

    with pytest.raises(CacheUnavailableError):
        upstash.execute(["PING"])

    with pytest.raises(CacheCoolingDown):
        upstash.execute(["PING"])

    assert len(session.calls) == 1
    assert upstash.seconds_until_available() == cooldown

    clock.now += cooldown
    assert upstash.execute(["PING"]) == "PONG"


def test_close_only_closes_a_session_it_created():
    upstash, session, _ = client(FakeResponse(payload={"result": "OK"}))
    upstash.close()
    assert not session.closed

    UpstashRestClient(url=URL, token="t").close()
