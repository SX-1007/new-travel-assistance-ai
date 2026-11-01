"""Exa search HTTP client, with its own key, session and cooldowns.

Exa's free plan is prepaid: $10 of credit each calendar month, 10 requests
per second. When the credit is used up every request answers 402 until the
month resets, so the client remembers refusals that will repeat (no credit,
bad key, rate limit) and refuses calls itself until they can succeed.
Callers then go straight to their backup instead of waiting on a request
that is bound to fail.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any, Final

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

_EXA_SEARCH_URL: Final = "https://api.exa.ai/search"
# Above Exa's own page-fetch limit (10 s by default), so a search that
# refreshes cached pages isn't cut off while Exa is still answering.
_REQUEST_TIMEOUT_SECONDS: Final = 20.0

# Out of credit (or over the key's own budget, also 402). The free credit
# returns on the 1st of the month, in a time zone Exa doesn't state, so the
# client tries again at most this often, and at the start of each UTC month.
CREDIT_EXHAUSTED_COOLDOWN: Final = 6 * 3600.0
# A missing, invalid or revoked key needs a person to fix it.
AUTH_FAILURE_COOLDOWN: Final = 600.0
# Rate limited without a usable Retry-After header.
DEFAULT_RATE_LIMIT_COOLDOWN: Final = 10.0
_MAX_RATE_LIMIT_COOLDOWN: Final = 300.0

# Only connection failures are retried: the request never reached Exa, so
# it can't have been billed. Anything Exa answered is final, and the caller
# falls back instead of paying for a second attempt.
_RETRY_STRATEGY: Final = Retry(
    total=2,
    connect=2,
    read=0,
    status=0,
    other=0,
    backoff_factor=0.5,
    raise_on_status=False,
)


class ExaUnavailableError(RuntimeError):
    """Raised when Exa cannot return a valid response."""


class ExaCoolingDown(ExaUnavailableError):
    """Raised instead of calling Exa while an earlier refusal still applies."""


def _build_http_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=_RETRY_STRATEGY)
    session.mount("https://", adapter)
    return session


class ExaClient:
    """One Exa key and its own HTTP session; thread-safe."""

    def __init__(
        self,
        *,
        api_key: str,
        session: requests.Session | None = None,
        timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        key = api_key.strip()

        if not key:
            raise ValueError("api_key must not be empty")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        self._api_key = key
        self._session = session or _build_http_session()
        self._owns_session = session is None
        self._timeout_seconds = float(timeout_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._available_at = 0.0

    def __repr__(self) -> str:
        # The key is never part of the representation.
        return "ExaClient()"

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def seconds_until_available(self) -> float:
        with self._lock:
            return max(self._available_at - self._clock(), 0.0)

    def search(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """The response JSON. Raises ``ExaCoolingDown`` without calling Exa
        during a cooldown, and ``ExaUnavailableError`` on any HTTP, transport
        or parsing failure (the key goes in a header and is never logged)."""

        if self.seconds_until_available() > 0:
            raise ExaCoolingDown("Exa is cooling down after an earlier refusal.")

        try:
            response = self._session.post(
                _EXA_SEARCH_URL,
                json=dict(body),
                headers={"x-api-key": self._api_key},
                timeout=self._timeout_seconds,
            )
        except requests.RequestException as exc:
            safe_log(logger, logging.WARNING, "provider.exa.transport_failed")
            raise ExaUnavailableError("Exa request failed.") from exc

        if response.status_code != 200:
            self._after_refusal(response)
            safe_log(logger, logging.WARNING, "provider.exa.http_failed", status_code=response.status_code)
            raise ExaUnavailableError("Exa request failed.")

        try:
            payload = response.json()
        except ValueError as exc:
            safe_log(logger, logging.WARNING, "provider.exa.invalid_json")
            raise ExaUnavailableError("Exa returned invalid JSON.") from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            safe_log(logger, logging.WARNING, "provider.exa.invalid_payload")
            raise ExaUnavailableError("Exa returned an invalid payload.")

        return payload

    def _after_refusal(self, response: requests.Response) -> None:
        """Start a cooldown for refusals that the next call would repeat.

        Server errors and rejected requests (400) don't start one: the next
        question may well succeed.
        """

        status = response.status_code
        now = self._clock()

        if status == 402:
            cooldown = min(CREDIT_EXHAUSTED_COOLDOWN, _seconds_until_next_utc_month(now))
            event = "provider.exa.credit_exhausted"
        elif status in {401, 403}:
            cooldown = AUTH_FAILURE_COOLDOWN
            event = "provider.exa.key_rejected"
        elif status == 429:
            cooldown = _retry_after_seconds(response.headers.get("Retry-After"))
            event = "provider.exa.rate_limited"
        else:
            return

        with self._lock:
            self._available_at = max(self._available_at, now + cooldown)

        safe_log(logger, logging.WARNING, event, status_code=status)


def _seconds_until_next_utc_month(now: float) -> float:
    current = datetime.fromtimestamp(now, timezone.utc)
    year, month = (current.year + 1, 1) if current.month == 12 else (current.year, current.month + 1)
    return datetime(year, month, 1, tzinfo=timezone.utc).timestamp() - now


def _retry_after_seconds(value: str | None) -> float:
    """Retry-After in seconds, bounded. The HTTP-date form isn't read and
    gets the default instead."""

    try:
        seconds = float(value) if value is not None else DEFAULT_RATE_LIMIT_COOLDOWN
    except ValueError:
        seconds = DEFAULT_RATE_LIMIT_COOLDOWN

    if seconds != seconds or seconds <= 0:  # NaN or nonsense
        seconds = DEFAULT_RATE_LIMIT_COOLDOWN

    return min(seconds, _MAX_RATE_LIMIT_COOLDOWN)


__all__ = [
    "AUTH_FAILURE_COOLDOWN",
    "CREDIT_EXHAUSTED_COOLDOWN",
    "DEFAULT_RATE_LIMIT_COOLDOWN",
    "ExaClient",
    "ExaCoolingDown",
    "ExaUnavailableError",
]
