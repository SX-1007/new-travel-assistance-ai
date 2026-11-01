"""Sign-in on every route: the bearer-token dependency with fake verifiers."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import CurrentUser
from app.api.errors import register_exception_handlers
from app.api.providers import identity_verifier
from app.modules.identity.exceptions import IdentityVerificationUnavailable, InvalidIdentityToken


class Verifier:
    """Accepts ``good-<uid>``; any other token is invalid."""

    def __init__(self, error=None):
        self.error = error
        self.tokens = []

    def verify(self, token):
        self.tokens.append(token)
        if self.error:
            raise self.error
        if not token.startswith("good-"):
            raise InvalidIdentityToken("bad")
        return token.removeprefix("good-")


def client(verifier):
    api = FastAPI()
    register_exception_handlers(api)

    @api.get("/whoami")
    def whoami(user_id: CurrentUser):
        return {"user_id": user_id}

    api.dependency_overrides[identity_verifier] = lambda: verifier
    return TestClient(api, raise_server_exceptions=False)


def test_a_verified_token_gives_the_route_its_user():
    verifier = Verifier()

    response = client(verifier).get("/whoami", headers={"Authorization": "Bearer good-uid-1"})

    assert response.json() == {"user_id": "uid-1"}
    assert verifier.tokens == ["good-uid-1"]


def test_the_scheme_is_case_insensitive_and_spaces_are_trimmed():
    response = client(Verifier()).get("/whoami", headers={"Authorization": "  bearer   good-uid-1 "})

    assert response.json() == {"user_id": "uid-1"}


def test_no_token_asks_for_sign_in():
    verifier = Verifier()

    response = client(verifier).get("/whoami")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"] == "Sign-in required."
    assert verifier.tokens == []


@pytest.mark.parametrize("header", ["X-User-ID", "x-uid", "x-auth-user-id"])
def test_the_old_user_id_headers_are_no_longer_trusted(header):
    response = client(Verifier()).get("/whoami", headers={header: "user-1"})

    assert response.status_code == 401


@pytest.mark.parametrize(
    "header",
    ["Basic dXNlcjpwYXNz", "Bearer", "Bearer    ", "Token good-uid-1", "Bearer " + "x" * 4097],
)
def test_malformed_headers_are_refused_before_any_check(header):
    verifier = Verifier()

    response = client(verifier).get("/whoami", headers={"Authorization": header})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Bearer error="invalid_token"'
    assert verifier.tokens == []


def test_a_rejected_token_means_sign_in_again():
    response = client(Verifier()).get("/whoami", headers={"Authorization": "Bearer forged"})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Bearer error="invalid_token"'
    assert "sign in again" in response.json()["detail"]


def test_a_check_that_cannot_run_is_retryable_not_a_sign_out():
    verifier = Verifier(error=IdentityVerificationUnavailable("keys down"))

    response = client(verifier).get("/whoami", headers={"Authorization": "Bearer good-uid-1"})

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
