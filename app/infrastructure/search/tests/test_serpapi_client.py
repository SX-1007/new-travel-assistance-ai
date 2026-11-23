"""SerpApiClient against a real local HTTP server, so urllib3's own retry
logic is what decides how many requests are sent."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.infrastructure.search import serpapi
from app.infrastructure.search.serpapi import (
    SerpApiClient,
    SerpApiCoolingDown,
    SerpApiKeyHealth,
    SerpApiQuotaExhausted,
    SerpApiUnavailableError,
)
from app.shared.exceptions import ProviderQuotaExhausted


class _Server:
    """Answers each request with the next (status, headers, body) in turn."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                server.requests += 1
                status, headers, body = server.replies.pop(0) if server.replies else (200, {}, {})
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/search"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve(monkeypatch):
    servers = []

    def start(*replies):
        server = _Server(replies)
        servers.append(server)
        monkeypatch.setattr(serpapi, "_SERPAPI_URL", server.url)
        return server

    yield start

    for server in servers:
        server.close()


OUT_OF_SEARCHES = {"error": "Your account has run out of searches."}


def test_a_used_up_quota_makes_exactly_one_request_even_with_retry_after(serve):
    server = serve((429, {"Retry-After": "1"}, OUT_OF_SEARCHES))
    client = SerpApiClient(api_key="k")

    with pytest.raises(SerpApiQuotaExhausted) as caught:
        client.search({"q": "x"})

    assert server.requests == 1
    # Old callers that only catch the failure type still catch it.
    assert isinstance(caught.value, SerpApiUnavailableError)
    assert isinstance(caught.value, ProviderQuotaExhausted)


def test_the_next_call_makes_no_network_request(serve):
    server = serve((429, {}, OUT_OF_SEARCHES))
    health = SerpApiKeyHealth()
    flights = SerpApiClient(api_key="k", health=health)
    places = SerpApiClient(api_key="k", health=health)

    with pytest.raises(SerpApiQuotaExhausted):
        flights.search({"q": "x"})

    # Every client of the key rests, without sending anything.
    for client in (flights, places):
        with pytest.raises(SerpApiQuotaExhausted):
            client.search({"q": "x"})

    assert server.requests == 1
    assert health.seconds_until_available() == pytest.approx(serpapi.SEARCHES_EXHAUSTED_COOLDOWN, abs=5)


def test_out_of_searches_in_a_200_body_is_also_a_used_up_quota(serve):
    server = serve((200, {}, OUT_OF_SEARCHES))

    with pytest.raises(SerpApiQuotaExhausted):
        SerpApiClient(api_key="k").search({"q": "x"})

    assert server.requests == 1


def test_a_throughput_429_rests_for_its_retry_after(serve):
    serve((429, {"Retry-After": "120"}, {"error": "Too many requests."}))
    health = SerpApiKeyHealth()

    with pytest.raises(SerpApiQuotaExhausted):
        SerpApiClient(api_key="k", health=health).search({"q": "x"})

    assert health.seconds_until_available() == pytest.approx(120, abs=5)


def test_server_errors_are_still_retried(serve):
    server = serve((503, {}, {}), (200, {}, {"organic_results": []}))

    assert SerpApiClient(api_key="k").search({"q": "x"}) == {"organic_results": []}
    assert server.requests == 2


def test_a_rejected_key_rests_as_an_ordinary_failure(serve):
    server = serve((401, {}, {"error": "Invalid API key."}))
    client = SerpApiClient(api_key="k")

    with pytest.raises(SerpApiUnavailableError) as first:
        client.search({"q": "x"})

    with pytest.raises(SerpApiCoolingDown):
        client.search({"q": "x"})

    assert not isinstance(first.value, ProviderQuotaExhausted)
    assert server.requests == 1


def test_a_no_results_error_body_is_returned_to_the_caller(serve):
    serve((200, {}, {"error": "Google hasn't returned any results for this query."}))

    payload = SerpApiClient(api_key="k").search({"q": "x"})

    assert payload["error"].startswith("Google hasn't")


def test_the_key_is_never_in_the_representation():
    assert "secret" not in repr(SerpApiClient(api_key="secret"))
