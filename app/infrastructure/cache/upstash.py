"""Redis commands over Upstash's REST API, with its own session and cooldowns.

Each command is one HTTPS POST of a JSON array (``["GET", "key"]``) that
answers ``{"result": ...}`` or ``{"error": "..."}``. A kept-alive session
matters: the first request pays a TLS handshake (about 0.5 s from here),
later ones about 40 ms.

A cache must never slow a request down much, so timeouts are short and a
failure that the next command would repeat (Upstash unreachable, token
rejected, monthly command limit reached) makes the client refuse commands
itself for a while instead of waiting on each one.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Final
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

# (connect, read) seconds.
_TIMEOUT_SECONDS: Final = (3.05, 3.0)
# Upstash unreachable or timing out: skip it briefly, then try again.
TRANSPORT_FAILURE_COOLDOWN: Final = 30.0
# A rejected token needs a person to fix it.
AUTH_FAILURE_COOLDOWN: Final = 600.0
# The free plan's command allowance (500,000 a month) or a daily limit.
COMMAND_LIMIT_COOLDOWN: Final = 900.0

_COMMAND_LIMIT: Final = re.compile(r"max (?:daily )?requests? limit", re.IGNORECASE)
_STORAGE_FULL: Final = re.compile(r"\bOOM\b|max ?memory|max data size", re.IGNORECASE)

# One retry, only when the connection couldn't be made: nothing reached
# Upstash, so a write can't be applied twice.
_RETRY_STRATEGY: Final = Retry(total=1, connect=1, read=0, status=0, other=0, raise_on_status=False)


class CacheUnavailableError(RuntimeError):
    """Raised when a cache command did not run or its answer can't be read."""


class CacheCoolingDown(CacheUnavailableError):
    """Raised instead of sending a command while an earlier failure applies."""


def _build_http_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=2, pool_maxsize=10, max_retries=_RETRY_STRATEGY)
    session.mount("https://", adapter)
    return session


class UpstashRestClient:
    """One Upstash database, token and HTTP session; thread-safe."""

    def __init__(
        self,
        *,
        url: str,
        token: str,
        session: requests.Session | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        url = url.strip().rstrip("/")
        token = token.strip()

        if urlparse(url).scheme != "https" or not urlparse(url).hostname:
            raise ValueError("url must be an https:// Upstash REST URL")

        if not token:
            raise ValueError("token must not be empty")

        self._url = url
        self._headers = {"Authorization": f"Bearer {token}"}
        self._session = session or _build_http_session()
        self._owns_session = session is None
        self._clock = clock
        self._lock = threading.Lock()
        self._available_at = 0.0

    def __repr__(self) -> str:
        # Neither the URL (it names the database) nor the token.
        return "UpstashRestClient()"

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def seconds_until_available(self) -> float:
        with self._lock:
            return max(self._available_at - self._clock(), 0.0)

    def execute(self, command: Sequence[str | int]) -> Any:
        """The command's result. Raises ``CacheCoolingDown`` without sending
        during a cooldown, and ``CacheUnavailableError`` on any failure."""

        if self.seconds_until_available() > 0:
            raise CacheCoolingDown("The cache is cooling down after an earlier failure.")

        try:
            response = self._session.post(
                self._url,
                json=[str(part) for part in command],
                headers=self._headers,
                timeout=_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            self._cool_down(TRANSPORT_FAILURE_COOLDOWN, "cache.upstash.transport_failed")
            raise CacheUnavailableError("Cache request failed.") from exc

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if response.status_code == 401:
            self._cool_down(AUTH_FAILURE_COOLDOWN, "cache.upstash.token_rejected", status=401)
            raise CacheUnavailableError("Cache token rejected.")

        if isinstance(payload, dict) and isinstance(payload.get("error"), str):
            self._command_failed(payload["error"], response.status_code)
            raise CacheUnavailableError("Cache command failed.")

        if response.status_code != 200 or not isinstance(payload, dict) or "result" not in payload:
            safe_log(logger, logging.WARNING, "cache.upstash.invalid_response", status_code=response.status_code)
            raise CacheUnavailableError("Cache returned an invalid response.")

        return payload["result"]

    def _command_failed(self, error: str, status: int) -> None:
        # The message is Upstash's own text and may quote the key; only its
        # kind is logged.
        if _COMMAND_LIMIT.search(error):
            self._cool_down(COMMAND_LIMIT_COOLDOWN, "cache.upstash.command_limit_reached", status=status)
        elif _STORAGE_FULL.search(error):
            # Reads still work; only writes fail until entries expire.
            safe_log(logger, logging.ERROR, "cache.upstash.storage_full", status_code=status)
        else:
            safe_log(logger, logging.WARNING, "cache.upstash.command_failed", status_code=status)

    def _cool_down(self, seconds: float, event: str, *, status: int | None = None) -> None:
        with self._lock:
            self._available_at = max(self._available_at, self._clock() + seconds)

        safe_log(logger, logging.WARNING, event, status_code=status)


__all__ = [
    "AUTH_FAILURE_COOLDOWN",
    "COMMAND_LIMIT_COOLDOWN",
    "TRANSPORT_FAILURE_COOLDOWN",
    "CacheCoolingDown",
    "CacheUnavailableError",
    "UpstashRestClient",
]
