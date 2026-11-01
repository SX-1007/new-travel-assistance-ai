"""Profile and geocode routes, and the assembled app's probes and routes."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.main as main
from app.api.errors import register_exception_handlers
from app.api.providers import identity_verifier, onboarding, place_pin_locator
from app.api.routers import geocode, profile
from app.infrastructure.auth.dev_identity import InsecureDevIdentityVerifier
from app.modules.places.exceptions import PlaceSearchUnavailable
from app.modules.places.models import PlaceCandidate
from app.modules.profile.exceptions import ProfileUnavailable
from app.modules.profile.models import OnboardingProfile


USER = {"Authorization": "Bearer user-1"}
ANA = OnboardingProfile(name="Ana", home_country="Malaysia", home_state="Selangor")


class Onboarding:
    def __init__(self, profile=None, error=None):
        self.profile = profile
        self.error = error
        self.saved = []

    def get(self, user_id):
        if self.error:
            raise self.error
        return self.profile

    def save(self, user_id, profile):
        if self.error:
            raise self.error
        self.saved.append((user_id, profile))


class Locator:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def locate(self, query, *, near=None):
        self.calls.append((query, near))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def client(*, onboarding_service=None, locator=None):
    api = FastAPI()
    register_exception_handlers(api)
    api.include_router(profile.router, prefix="/api/profile")
    api.include_router(geocode.router, prefix="/api/geocode")
    api.dependency_overrides[onboarding] = lambda: onboarding_service or Onboarding()
    api.dependency_overrides[place_pin_locator] = lambda: locator or Locator(None)
    api.dependency_overrides[identity_verifier] = InsecureDevIdentityVerifier
    return TestClient(api, raise_server_exceptions=False)


# ── Profile ─────────────────────────────────────────────────────────


def test_profile_reports_onboarding_with_only_its_fields():
    done = client(onboarding_service=Onboarding(ANA)).get("/api/profile/", headers=USER)
    fresh = client(onboarding_service=Onboarding(None)).get("/api/profile/", headers=USER)

    assert done.json() == {
        "status": "success",
        "onboarded": True,
        "profile": {"name": "Ana", "home_country": "Malaysia", "home_state": "Selangor"},
    }
    assert done.headers["cache-control"] == "no-store, max-age=0"
    assert fresh.json() == {"status": "success", "onboarded": False, "profile": {}}


def test_saving_a_profile():
    service = Onboarding()
    body = {"name": " Ana ", "origin_country": "Malaysia", "origin_state": "Selangor"}

    response = client(onboarding_service=service).post("/api/profile/", json=body, headers=USER)

    assert response.json() == {"status": "success", "user_id": "user-1"}
    assert service.saved == [("user-1", ANA)]


def test_profile_errors():
    bad_country = {"name": "Ana", "origin_country": "Atlantis", "origin_state": "X"}
    down = client(onboarding_service=Onboarding(error=ProfileUnavailable()))

    assert client().post("/api/profile/", json=bad_country, headers=USER).status_code == 422
    assert client().get("/api/profile/").status_code == 401
    for response in (
        down.get("/api/profile/", headers=USER),
        down.post("/api/profile/", json={**bad_country, "origin_country": "Japan"}, headers=USER),
    ):
        assert response.status_code == 503 and response.headers["retry-after"] == "5"


# ── Geocode ─────────────────────────────────────────────────────────


def test_a_found_place_returns_its_coordinates():
    locator = Locator(PlaceCandidate(title="Tokyo Tower", latitude=35.65, longitude=139.74, address="Shibakoen"))

    body = client(locator=locator).get("/api/geocode/?q=Tokyo Tower&lat=35.6&lng=139.7", headers=USER).json()

    assert body == {
        "status": "success",
        "found": True,
        "name": "Tokyo Tower",
        "address": "Shibakoen",
        "lat": 35.65,
        "lng": 139.74,
    }
    assert locator.calls == [("Tokyo Tower", (35.6, 139.7))]


def test_a_miss_or_failure_is_not_found_without_null_fields():
    miss = client(locator=Locator(None)).get("/api/geocode/?q=Nowhere&lat=1", headers=USER).json()
    down = client(locator=Locator(PlaceSearchUnavailable())).get("/api/geocode/?q=Nowhere", headers=USER).json()

    assert miss == {"status": "success", "found": False}
    assert down == {"status": "error", "found": False}


def test_geocode_input_is_validated():
    assert client().get("/api/geocode/?q=ab", headers=USER).status_code == 422
    assert client().get("/api/geocode/?q=Tokyo&lat=91&lng=0", headers=USER).status_code == 422
    assert client().get("/api/geocode/?q=Tokyo").status_code == 401


# ── The assembled app ───────────────────────────────────────────────


def test_main_registers_every_route_without_starting_resources():
    paths = {route.path for route in main.app.routes}

    assert {
        "/api/chat/message",
        "/api/chat/history",
        "/api/form/submit",
        "/api/profile/",
        "/api/geocode/",
        "/health",
        "/health/ready",
    } <= paths


def test_liveness_and_readiness_probes(monkeypatch):
    # No ``with``: the lifespan (and so any real storage) never starts.
    api = TestClient(main.app)
    assert api.get("/health").json() == {"status": "ok"}

    monkeypatch.setattr(main, "readiness", lambda: {"postgres": True, "travel_vectors": True, "checkpoints": True})
    assert api.get("/health/ready").status_code == 200

    monkeypatch.setattr(main, "readiness", lambda: {"postgres": True, "travel_vectors": True, "checkpoints": False})
    not_ready = api.get("/health/ready")
    assert not_ready.status_code == 503 and not_ready.json()["checks"]["checkpoints"] is False


def test_the_assembled_app_maps_errors(monkeypatch):
    # A route that needs the graph before startup finished: 503, not a crash.
    monkeypatch.setitem(main.app.dependency_overrides, identity_verifier, InsecureDevIdentityVerifier)
    api = TestClient(main.app, raise_server_exceptions=False)

    response = api.post("/api/chat/message", json={"session_id": "s", "user_message": "hi"}, headers=USER)

    assert response.status_code == 503 and response.headers["retry-after"] == "5"


def test_startup_and_shutdown_run_in_order_and_release_on_failure(monkeypatch):
    calls = []

    async def init_graph():
        calls.append("init_graph")

    async def shutdown_graph():
        calls.append("shutdown_graph")

    monkeypatch.setattr(main, "init_resources", lambda: calls.append("init_resources"))
    monkeypatch.setattr(main, "init_graph", init_graph)
    monkeypatch.setattr(main, "verify_startup_health", lambda: calls.append("health") or True)
    monkeypatch.setattr(main, "shutdown_graph", shutdown_graph)
    monkeypatch.setattr(main, "shutdown_container", lambda: calls.append("shutdown_container"))

    with TestClient(main.create_app()) as api:
        assert api.get("/health").status_code == 200

    assert calls == ["init_resources", "init_graph", "health", "shutdown_graph", "shutdown_container"]

    calls.clear()

    def broken():
        raise RuntimeError("no database")

    monkeypatch.setattr(main, "init_resources", broken)

    try:
        with TestClient(main.create_app()):
            raise AssertionError("startup should have failed")
    except RuntimeError:
        pass

    assert calls == ["shutdown_graph", "shutdown_container"]


def test_validation_errors_read_as_one_sentence_the_client_can_show():
    bad_country = {"name": "Ana", "origin_country": "Atlantis", "origin_state": "X"}

    body = client().post("/api/profile/", json=bad_country, headers=USER).json()

    assert body["detail"] == "origin_country: origin_country must be a recognised country name or code"
    assert body["errors"] == [
        {"loc": ["body", "origin_country"], "msg": body["errors"][0]["msg"], "type": "value_error"}
    ]
    # Submitted values are never echoed back.
    assert "Atlantis" not in str(body)
