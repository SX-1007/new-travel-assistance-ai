"""Firebase Authentication ID tokens, checked with firebase-admin.

The verifier owns a named Firebase app, so it never shares or depends on a
default app another part of the process might create. Google's signing keys are
fetched once and cached by firebase-admin; a check needs no other network
call. Revocation is not checked (that would be one more call per request), so
a disabled account keeps working until its token expires, at most one hour.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Final

import firebase_admin
from firebase_admin import auth, credentials

from app.modules.identity.exceptions import IdentityVerificationUnavailable, InvalidIdentityToken
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

_APP_NAME: Final = "identity"
# Tolerates a few seconds of clock drift between this machine and Google's.
_CLOCK_SKEW_SECONDS: Final = 10
# For fetching Google's signing keys. firebase-admin's default is 120 s per
# attempt, and a stalled connection would hold the request that long.
_HTTP_TIMEOUT_SECONDS: Final = 5
# Firebase's sign_in_provider for an email-and-password account.
_PASSWORD_PROVIDER: Final = "password"


class FirebaseIdTokenVerifier:
    """Verify Firebase ID tokens; the app is created on first use."""

    def __init__(self, *, credential_json: str) -> None:
        if not credential_json or not credential_json.strip():
            raise ValueError("credential_json must not be empty")

        self._credential_json = credential_json.strip()
        self._app: firebase_admin.App | None = None
        self._lock = threading.Lock()

    def verify(self, token: str) -> str:
        app = self._firebase_app()

        try:
            claims = auth.verify_id_token(token, app=app, clock_skew_seconds=_CLOCK_SKEW_SECONDS)
        except auth.CertificateFetchError as exc:
            safe_log(logger, logging.WARNING, "identity.firebase.keys_unavailable")
            raise IdentityVerificationUnavailable("Firebase signing keys are unreachable.") from exc
        except (auth.InvalidIdTokenError, auth.UserDisabledError, ValueError) as exc:
            # ExpiredIdTokenError and RevokedIdTokenError are InvalidIdTokenError.
            raise InvalidIdentityToken("The sign-in token is not valid.") from exc

        user_id = claims.get("uid")

        if not isinstance(user_id, str) or not user_id:
            raise InvalidIdentityToken("The sign-in token names no user.")

        # Anyone can create a password account for an address they don't own,
        # so one counts only after its owner opens the verification link.
        # Google accounts are verified by Google.
        provider = (claims.get("firebase") or {}).get("sign_in_provider")

        if provider == _PASSWORD_PROVIDER and claims.get("email_verified") is not True:
            raise InvalidIdentityToken("The account's email address is not verified.")

        return user_id

    def _firebase_app(self) -> firebase_admin.App:
        if self._app is not None:
            return self._app

        with self._lock:
            if self._app is not None:
                return self._app

            try:
                self._app = firebase_admin.get_app(_APP_NAME)
            except ValueError:
                self._app = firebase_admin.initialize_app(
                    self._certificate(),
                    options={"httpTimeout": _HTTP_TIMEOUT_SECONDS},
                    name=_APP_NAME,
                )

            return self._app

    def _certificate(self) -> credentials.Certificate:
        # The credential is a file path or the raw JSON itself.
        try:
            if os.path.isfile(self._credential_json):
                return credentials.Certificate(self._credential_json)

            return credentials.Certificate(json.loads(self._credential_json))
        except (OSError, ValueError) as exc:
            safe_log(logger, logging.ERROR, "identity.firebase.credentials_invalid")
            raise IdentityVerificationUnavailable("The Firebase credential is invalid.") from exc


__all__ = [
    "FirebaseIdTokenVerifier",
]
