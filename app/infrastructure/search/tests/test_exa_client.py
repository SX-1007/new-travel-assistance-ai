"""Exa HTTP client with a fake session and clock: request, errors, cooldowns."""

from datetime import datetime, timezone

import pytest
import requests

from app.infrastructure.search.exa import (
    AUTH_FAILURE_COOLDOWN,
    CREDIT_EXHAUSTED_COOLDOWN,
    DEFAULT_RATE_LIMIT_COOLDOWN,
    ExaClient,
    ExaCoolingDown,
    ExaUnavailableError,
)


MID_MONTH = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc).timestamp()


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, invalid_json=False):
        self.status_code = status_code
        self._payload = {"results": []} if payload is None else payload
        self.headers = headers or {}
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
    def __init__(self, now=MID_MONTH):
        self.now = now

    def __call__(self):
        return self.now


def client(*responses, clock=None):
    session = FakeSession(*responses)
    return ExaClient(api_key=" key-1 ", session=session, clock=clock or Clock()), session


def test_posts_the_body_with_the_key_only_in_a_header():
    exa, session = client(FakeResponse(payload={"results": [{"url": "https://a.go.jp"}]}))

    payload = exa.search({"query": "q", "numResults": 3})

    url, kwargs = session.calls[0]
    assert url == "https://api.exa.ai/search"
    assert kwargs["json"] == {"query": "q", "numResults": 3}
    assert kwargs["headers"] == {"x-api-key": "key-1"}
    assert kwargs["timeout"] > 10
    assert payload["results"] == [{"url": "https://a.go.jp"}]
    assert "key-1" not in repr(exa)


def test_rejects_an_empty_key_or_timeout():
    with pytest.raises(ValueError):
        ExaClient(api_key="  ")
    with pytest.raises(ValueError):
        ExaClient(api_key="k", timeout_seconds=0)


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(invalid_json=True),
        FakeResponse(payload=["not", "a", "dict"]),
        FakeResponse(payload={"error": "no results list"}),
        FakeResponse(payload={"results": "not a list"}),
    ],
)
def test_unreadable_answers_raise(response):
    exa, _ = client(response)

    with pytest.raises(ExaUnavailableError):
        exa.search({"query": "q"})


@pytest.mark.parametrize("failure", [FakeResponse(500), FakeResponse(503), FakeResponse(400), requests.ConnectionError()])
def test_transient_failures_raise_without_a_cooldown(failure):
    exa, session = client(failure, FakeResponse())

    with pytest.raises(ExaUnavailableError) as raised:
        exa.search({"query": "q"})

    assert not isinstance(raised.value, ExaCoolingDown)
    assert exa.seconds_until_available() == 0
    exa.search({"query": "q"})
    assert len(session.calls) == 2


def test_no_credit_stops_calls_until_the_cooldown_ends():
    clock = Clock()
    exa, session = client(FakeResponse(402), FakeResponse(), clock=clock)

    with pytest.raises(ExaUnavailableError):
        exa.search({"query": "q"})

    with pytest.raises(ExaCoolingDown):
        exa.search({"query": "q"})

    assert len(session.calls) == 1
    assert exa.seconds_until_available() == CREDIT_EXHAUSTED_COOLDOWN

    clock.now += CREDIT_EXHAUSTED_COOLDOWN
    exa.search({"query": "q"})
    assert len(session.calls) == 2


def test_no_credit_is_tried_again_when_a_new_utc_month_starts():
    an_hour_before = datetime(2026, 12, 31, 23, 0, tzinfo=timezone.utc).timestamp()
    exa, _ = client(FakeResponse(402), clock=Clock(an_hour_before))

    with pytest.raises(ExaUnavailableError):
        exa.search({"query": "q"})

    assert exa.seconds_until_available() == 3600


def test_a_rejected_key_cools_down():
    exa, _ = client(FakeResponse(401))

    with pytest.raises(ExaUnavailableError):
        exa.search({"query": "q"})

    assert exa.seconds_until_available() == AUTH_FAILURE_COOLDOWN


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ({"Retry-After": "3"}, 3),
        ({}, DEFAULT_RATE_LIMIT_COOLDOWN),
        ({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, DEFAULT_RATE_LIMIT_COOLDOWN),
        ({"Retry-After": "-5"}, DEFAULT_RATE_LIMIT_COOLDOWN),
        ({"Retry-After": "nan"}, DEFAULT_RATE_LIMIT_COOLDOWN),
        ({"Retry-After": "86400"}, 300),
    ],
)
def test_rate_limit_follows_a_bounded_retry_after(header, expected):
    exa, _ = client(FakeResponse(429, headers=header))

    with pytest.raises(ExaUnavailableError):
        exa.search({"query": "q"})

    assert exa.seconds_until_available() == expected


def test_a_shorter_refusal_never_cuts_a_longer_cooldown():
    # Two questions at once: the inner one runs out of credit while the
    # outer one is still in flight and then comes back rate limited.
    class OverlappingSession(FakeSession):
        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if len(self.calls) == 1:
                with pytest.raises(ExaUnavailableError):
                    exa.search({"query": "inner"})
                return FakeResponse(429, headers={"Retry-After": "1"})
            return FakeResponse(402)

    exa = ExaClient(api_key="k", session=OverlappingSession(), clock=Clock())

    with pytest.raises(ExaUnavailableError):
        exa.search({"query": "outer"})

    assert exa.seconds_until_available() == CREDIT_EXHAUSTED_COOLDOWN


def test_close_only_closes_a_session_it_created():
    exa, session = client(FakeResponse())
    exa.close()
    assert not session.closed

    own = ExaClient(api_key="k")
    own.close()
