"""Request body limit and security headers (``api/http_hardening.py``)."""

from __future__ import annotations

import json

from fastapi import FastAPI, Response
from fastapi.testclient import TestClient
from pydantic import BaseModel

import app.main as main
from app.api.http_hardening import MAX_REQUEST_BODY_BYTES, RequestBodyLimit, SecurityHeaders


class Message(BaseModel):
    text: str


def client(*, max_bytes=100, hsts=False):
    app = FastAPI()
    calls = []

    @app.post("/echo")
    async def echo(message: Message):
        calls.append(message.text)
        return {"length": len(message.text)}

    @app.get("/cached")
    async def cached(response: Response):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return {}

    @app.get("/docs-like")
    async def docs_like():
        return {}

    app.add_middleware(RequestBodyLimit, max_bytes=max_bytes)
    app.add_middleware(SecurityHeaders, hsts=hsts)
    return TestClient(app), calls


# ── Body limit ──────────────────────────────────────────────────────


def test_a_body_within_the_limit_reaches_the_route():
    api, calls = client()

    response = api.post("/echo", json={"text": "hello"})

    assert response.status_code == 200 and calls == ["hello"]


def test_a_declared_length_over_the_limit_is_refused_before_reading():
    api, calls = client()

    response = api.post("/echo", json={"text": "x" * 200})

    assert response.status_code == 413
    assert response.json() == {"detail": "The request is too large."}
    assert calls == []


def test_a_chunked_body_is_counted_as_it_arrives():
    # No Content-Length: the header check can't see it, the count must.
    api, calls = client()

    def chunks():
        yield b'{"text": "'
        for _ in range(20):
            yield b"x" * 10
        yield b'"}'

    response = api.post("/echo", content=chunks(), headers={"Content-Type": "application/json"})

    assert response.status_code == 413
    assert calls == []


def test_the_limit_fits_the_largest_chat_message_even_fully_escaped():
    api, calls = client(max_bytes=MAX_REQUEST_BODY_BYTES)
    # 8,192 emoji (the schema's maximum), each sent as a JSON surrogate pair.
    body = json.dumps({"text": "😀" * 8192}, ensure_ascii=True).encode()

    response = api.post("/echo", content=body, headers={"Content-Type": "application/json"})

    assert response.status_code == 200 and len(calls) == 1


def test_the_app_refuses_an_oversized_body():
    api = TestClient(main.create_app())

    response = api.post("/api/chat/message", content=b"x" * (MAX_REQUEST_BODY_BYTES + 1))

    assert response.status_code == 413


# ── Security headers ────────────────────────────────────────────────


def test_every_response_gets_the_security_headers():
    api, _ = client()

    headers = api.post("/echo", json={"text": "hi"}).headers

    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert "Strict-Transport-Security" not in headers


def test_refusals_get_them_too():
    api, _ = client()

    assert api.post("/echo", json={"text": "x" * 200}).headers["X-Content-Type-Options"] == "nosniff"


def test_a_header_the_route_set_is_kept():
    api, _ = client()

    assert api.get("/cached").headers["Cache-Control"] == "no-store, max-age=0"


def test_hsts_only_when_asked():
    api, _ = client(hsts=True)

    assert api.get("/cached").headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"


def test_production_sends_hsts_and_development_does_not(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CORS_ORIGINS", "https://travel.example.com")
    assert "Strict-Transport-Security" in TestClient(main.create_app()).get("/health").headers

    monkeypatch.setenv("ENVIRONMENT", "development")
    assert "Strict-Transport-Security" not in TestClient(main.create_app()).get("/health").headers


def test_the_development_docs_page_can_load_its_scripts(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    response = TestClient(main.create_app()).get("/docs")

    assert response.status_code == 200
    assert "Content-Security-Policy" not in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
