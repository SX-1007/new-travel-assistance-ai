"""Firebase ID-token verifier and the development stand-in, with no network."""

from __future__ import annotations

import json

import pytest
from firebase_admin import auth

import app.infrastructure.auth.firebase_identity as firebase_identity
from app.infrastructure.auth.dev_identity import InsecureDevIdentityVerifier
from app.infrastructure.auth.firebase_identity import FirebaseIdTokenVerifier
from app.modules.identity.exceptions import IdentityVerificationUnavailable, InvalidIdentityToken


class FirebaseFakes:
    """Stands in for firebase_admin's app registry and token check."""

    def __init__(self, monkeypatch, *, claims=None, error=None):
        self.apps = {}
        self.calls = []
        self.claims = claims if claims is not None else {"uid": "uid-1"}
        self.error = error
        monkeypatch.setattr(firebase_identity.firebase_admin, "get_app", self.get_app)
        monkeypatch.setattr(firebase_identity.firebase_admin, "initialize_app", self.initialize_app)
        monkeypatch.setattr(firebase_identity.credentials, "Certificate", lambda value: ("cert", value))
        monkeypatch.setattr(firebase_identity.auth, "verify_id_token", self.verify_id_token)

    def get_app(self, name):
        if name not in self.apps:
            raise ValueError("no app")
        return self.apps[name]

    def initialize_app(self, credential, options, name):
        self.options = options
        self.apps[name] = ("app", name, credential)
        return self.apps[name]

    def verify_id_token(self, token, app, clock_skew_seconds):
        self.calls.append((token, app, clock_skew_seconds))
        if self.error:
            raise self.error
        return self.claims


CREDENTIAL = json.dumps({"type": "service_account", "project_id": "demo"})


def test_a_valid_token_returns_its_uid_using_a_named_app(monkeypatch):
    fakes = FirebaseFakes(monkeypatch)
    verifier = FirebaseIdTokenVerifier(credential_json=CREDENTIAL)

    assert verifier.verify("token-1") == "uid-1"
    assert verifier.verify("token-2") == "uid-1"

    # Its own named app, never the default one; built once.
    assert list(fakes.apps) == ["identity"]
    assert fakes.apps["identity"][2] == ("cert", {"type": "service_account", "project_id": "demo"})
    assert [call[0] for call in fakes.calls] == ["token-1", "token-2"]
    assert fakes.calls[0][2] == 10
    # A stalled key fetch fails in seconds (a 503), not firebase-admin's 120 s.
    assert fakes.options == {"httpTimeout": 5}


def test_an_existing_named_app_is_reused(monkeypatch):
    fakes = FirebaseFakes(monkeypatch)
    fakes.apps["identity"] = "already-there"

    FirebaseIdTokenVerifier(credential_json=CREDENTIAL).verify("t")

    assert fakes.calls[0][1] == "already-there"


@pytest.mark.parametrize(
    "error",
    [
        auth.InvalidIdTokenError("bad"),
        auth.ExpiredIdTokenError("expired", cause=None),
        auth.RevokedIdTokenError("revoked"),
        auth.UserDisabledError("disabled"),
        ValueError("not a JWT"),
    ],
)
def test_rejected_tokens_are_invalid(monkeypatch, error):
    FirebaseFakes(monkeypatch, error=error)

    with pytest.raises(InvalidIdentityToken):
        FirebaseIdTokenVerifier(credential_json=CREDENTIAL).verify("t")


@pytest.mark.parametrize("claims", [{}, {"uid": ""}, {"uid": 7}])
def test_a_token_without_a_uid_is_invalid(monkeypatch, claims):
    FirebaseFakes(monkeypatch, claims=claims)

    with pytest.raises(InvalidIdentityToken):
        FirebaseIdTokenVerifier(credential_json=CREDENTIAL).verify("t")


def _password_claims(**extra):
    return {"uid": "uid-1", "firebase": {"sign_in_provider": "password"}, **extra}


@pytest.mark.parametrize("claims", [_password_claims(), _password_claims(email_verified=False)])
def test_an_email_account_must_be_verified(monkeypatch, claims):
    FirebaseFakes(monkeypatch, claims=claims)

    with pytest.raises(InvalidIdentityToken):
        FirebaseIdTokenVerifier(credential_json=CREDENTIAL).verify("t")


def test_a_verified_email_account_is_accepted(monkeypatch):
    FirebaseFakes(monkeypatch, claims=_password_claims(email_verified=True))

    assert FirebaseIdTokenVerifier(credential_json=CREDENTIAL).verify("t") == "uid-1"


def test_a_google_account_needs_no_separate_verification(monkeypatch):
    FirebaseFakes(monkeypatch, claims={"uid": "uid-1", "firebase": {"sign_in_provider": "google.com"}})

    assert FirebaseIdTokenVerifier(credential_json=CREDENTIAL).verify("t") == "uid-1"


def test_unreachable_signing_keys_are_unavailable_not_invalid(monkeypatch):
    FirebaseFakes(monkeypatch, error=auth.CertificateFetchError("down", cause=None))

    with pytest.raises(IdentityVerificationUnavailable):
        FirebaseIdTokenVerifier(credential_json=CREDENTIAL).verify("t")


def test_a_broken_credential_is_unavailable(monkeypatch):
    FirebaseFakes(monkeypatch)
    monkeypatch.setattr(firebase_identity.credentials, "Certificate", _reject)

    with pytest.raises(IdentityVerificationUnavailable):
        FirebaseIdTokenVerifier(credential_json="{not json").verify("t")


def _reject(value):
    raise ValueError("bad certificate")


def test_an_empty_credential_is_refused_at_construction():
    with pytest.raises(ValueError):
        FirebaseIdTokenVerifier(credential_json="  ")


def test_the_dev_verifier_takes_a_plain_user_id_only():
    verifier = InsecureDevIdentityVerifier()

    assert verifier.verify(" user-1 ") == "user-1"
    assert verifier.verify("smoke-test-1028") == "smoke-test-1028"

    for token in ("", "a b", "x" * 129, "eyJ<script>", "user/../admin"):
        with pytest.raises(InvalidIdentityToken):
            verifier.verify(token)
