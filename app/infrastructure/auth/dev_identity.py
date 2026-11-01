"""Local-development stand-in: the bearer token is taken as the user id.

Anyone can claim any user with it, so it is off unless
``AUTH_INSECURE_DEV_TOKENS`` is set, and start-up logs a warning when it is.
It exists for scripts and smoke tests that cannot sign in through a browser.
"""

from __future__ import annotations

import re
from typing import Final

from app.modules.identity.exceptions import InvalidIdentityToken


_USER_ID: Final = re.compile(r"^[A-Za-z0-9_.@-]{1,128}$")


class InsecureDevIdentityVerifier:
    def verify(self, token: str) -> str:
        user_id = token.strip()

        if not _USER_ID.match(user_id):
            raise InvalidIdentityToken("A development token must be a plain user id.")

        return user_id


__all__ = [
    "InsecureDevIdentityVerifier",
]
