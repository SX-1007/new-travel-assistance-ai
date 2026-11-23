"""The development / production switch, and the thread pools.

``ENVIRONMENT`` and ``CORS_ORIGINS`` are set in the process environment
here, which wins over any ``.env`` file, so the tests don't depend on it.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.bootstrap.container import check_deployment_settings
from app.infrastructure.config.environment import app_environment
from app.infrastructure.config.settings import Settings


FRONTEND = "https://travel.example.com"


def build(monkeypatch, environment, origins=None):
    monkeypatch.setenv("ENVIRONMENT", environment)

    if origins is not None:
        monkeypatch.setenv("CORS_ORIGINS", origins)

    return TestClient(main.create_app())


def preflight(api, origin, headers="authorization,content-type"):
    return api.options(
        "/api/chat/message",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": headers,
        },
    )


# ── ENVIRONMENT ─────────────────────────────────────────────────────


def test_development_is_the_default_and_keeps_the_docs(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setattr("app.infrastructure.config.environment._ENV_FILE", "no-such.env")

    assert app_environment() == "development"

    api = build(monkeypatch, "development")
    assert api.get("/docs").status_code == 200
    assert api.get("/openapi.json").status_code == 200


def test_an_unknown_environment_stops_the_app(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "prod")

    with pytest.raises(ValueError, match="ENVIRONMENT"):
        main.create_app()


def test_production_publishes_no_api_description(monkeypatch):
    api = build(monkeypatch, "production", FRONTEND)

    for path in ("/docs", "/redoc", "/openapi.json"):
        assert api.get(path).status_code == 404
    assert api.get("/health").status_code == 200


# ── CORS ────────────────────────────────────────────────────────────


def test_production_allows_only_the_frontend_without_credentials(monkeypatch):
    api = build(monkeypatch, "production", FRONTEND)

    allowed = preflight(api, FRONTEND)
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == FRONTEND
    assert "access-control-allow-credentials" not in allowed.headers

    assert "access-control-allow-origin" not in preflight(api, "http://localhost:5173").headers
    # A header the frontend never sends is refused.
    assert preflight(api, FRONTEND, headers="authorization,x-anything").status_code == 400


def test_retry_after_is_readable_by_the_frontend(monkeypatch):
    api = build(monkeypatch, "production", FRONTEND)

    response = api.get("/health", headers={"Origin": FRONTEND})

    assert "Retry-After" in response.headers["access-control-expose-headers"]


@pytest.mark.parametrize("origins", ["", "http://travel.example.com", "*"])
def test_production_refuses_to_start_without_https_frontend_origins(monkeypatch, origins):
    with pytest.raises(RuntimeError, match="CORS_ORIGINS"):
        build(monkeypatch, "production", origins)


def test_development_still_allows_any_localhost_port(monkeypatch):
    api = build(monkeypatch, "development", "")

    response = preflight(api, "http://localhost:5174", headers="authorization,x-anything")

    assert response.headers["access-control-allow-origin"] == "http://localhost:5174"


# ── Start-up checks ─────────────────────────────────────────────────


def settings(**values):
    defaults = dict(
        ENVIRONMENT="production",
        AUTH_INSECURE_DEV_TOKENS=False,
        UPSTASH_REDIS_REST_URL="https://example.upstash.io",
        UPSTASH_REDIS_REST_TOKEN="token",
    )
    return Settings.model_construct(**{**defaults, **values})


def test_production_refuses_development_sign_in_and_missing_limits():
    check_deployment_settings(settings())

    with pytest.raises(RuntimeError, match="AUTH_INSECURE_DEV_TOKENS"):
        check_deployment_settings(settings(AUTH_INSECURE_DEV_TOKENS=True))

    with pytest.raises(RuntimeError, match="UPSTASH"):
        check_deployment_settings(settings(UPSTASH_REDIS_REST_TOKEN=""))


def test_development_allows_both():
    check_deployment_settings(settings(ENVIRONMENT="development", AUTH_INSECURE_DEV_TOKENS=True, UPSTASH_REDIS_REST_URL=""))


# ── Threads ─────────────────────────────────────────────────────────


def test_startup_gives_the_shared_pool_its_own_threads(monkeypatch):
    seen = {}

    def readiness():
        seen["thread"] = threading.current_thread().name
        return {"postgres": True}

    async def nothing():
        return None

    monkeypatch.setattr(main, "init_resources", lambda: None)
    monkeypatch.setattr(main, "init_graph", nothing)
    monkeypatch.setattr(main, "verify_startup_health", lambda: True)
    monkeypatch.setattr(main, "shutdown_graph", nothing)
    monkeypatch.setattr(main, "shutdown_container", lambda: None)
    monkeypatch.setattr(main, "readiness", readiness)

    with TestClient(main.create_app()) as api:
        assert api.get("/health/ready").status_code == 200

    assert seen["thread"].startswith("shared")
    assert main.SHARED_THREADS == 32


def test_planning_runs_on_its_own_pool():
    from concurrent.futures import ThreadPoolExecutor

    from app.modules.itinerary.tests.test_planning_end_to_end import GOOD_IDEAS, FakeIdeas, World, state

    class RecordingIdeas(FakeIdeas):
        def propose(self, request):
            self.thread = threading.current_thread().name
            return super().propose(request)

    ideas = RecordingIdeas(GOOD_IDEAS)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="planning-test") as pool:
        update = World(ideas=ideas).run(state(), executor=pool)

    assert update["planning_outcome"] == "validated"
    assert ideas.thread.startswith("planning-test")


def test_the_container_gives_planning_a_small_pool_of_its_own():
    from app.bootstrap import container

    pool = container.get_planning_executor()

    try:
        assert pool._max_workers == container.PLANNING_THREADS == container.MAX_PLANS_RUNNING + 1
    finally:
        pool.shutdown(wait=False)
        container.get_planning_executor.cache_clear()


# ── .env.example ────────────────────────────────────────────────────


def test_the_env_example_lists_every_setting():
    # A new machine's checklist: a setting added to Settings must be added
    # there too.
    import re
    from pathlib import Path

    example = Path(__file__).resolve().parents[3] / ".env.example"
    listed = {
        match.group(1)
        for line in example.read_text(encoding="utf-8").splitlines()
        if (match := re.match(r"#?\s*([A-Z][A-Z0-9_]*)=", line))
    }

    assert set(Settings.model_fields) <= listed
    assert {"CORS_ORIGINS", "LOG_LEVEL", "APP_VERSION"} <= listed
