"""SerpAPI HTTP client, shared by every SerpAPI-backed provider.

Each provider gets its own instance (and HTTP session), so one feature's
traffic or failures never hold up another's.

A SerpAPI plan has a monthly number of searches (and an hourly throughput
limit); when either is reached SerpAPI answers 429, or an error saying the
account has run out of searches. That is not "SerpAPI failed": retrying
can't help, and the caller must not show an empty result as "nothing
found". So the client:

* never retries a 429 (it still retries server errors and dropped
  connections);
* raises ``SerpApiQuotaExhausted`` for it, a subtype of
  ``SerpApiUnavailableError`` (callers that only know the failure type keep
  working) and of the shared ``ProviderQuotaExhausted``;
* records a cooldown in the key's ``SerpApiKeyHealth``, shared by every
  client of that key, and refuses calls itself until it ends, without any
  network call (the same idea as ``infrastructure/llm/health.py``).

A rejected key (401/403) also starts a cooldown, as an ordinary failure.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Final, Literal

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.shared.exceptions import ProviderQuotaExhausted
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

_SERPAPI_URL: Final = "https://serpapi.com/search"
_REQUEST_TIMEOUT_SECONDS: Final = 15.0

# Out of the plan's searches. The plan renews on the account's own billing
# day, which the API doesn't say, so the key is tried again at most hourly.
SEARCHES_EXHAUSTED_COOLDOWN: Final = 3600.0
# Over the hourly throughput limit, without a usable Retry-After header.
DEFAULT_RATE_LIMIT_COOLDOWN: Final = 60.0
_MAX_RATE_LIMIT_COOLDOWN: Final = 3600.0
# A missing, invalid or deleted key needs a person to fix it.
AUTH_FAILURE_COOLDOWN: Final = 600.0

# SerpAPI's wording: "Your account has run out of searches."
_OUT_OF_SEARCHES: Final = re.compile(r"run out of searches|out of searches|searches? (?:limit|quota)", re.IGNORECASE)

class _NoRateLimitRetry(Retry):
    """urllib3 also retries any 429 that carries Retry-After, whatever the
    status list says; only 503 may be retried that way here."""

    RETRY_AFTER_STATUS_CODES = frozenset({503})


# Connection problems and server errors are retried, honouring Retry-After;
# a 429 never is: it means the quota or throughput is used up.
_RETRY_STRATEGY: Final = _NoRateLimitRetry(
    total=3,
    connect=3,
    read=2,
    status=3,
    backoff_factor=1.0,
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
    respect_retry_after_header=True,
    raise_on_status=False,
)

CooldownReason = Literal["quota", "auth"]


class SerpApiUnavailableError(RuntimeError):
    """Raised when SerpAPI cannot return a valid response."""


class SerpApiQuotaExhausted(SerpApiUnavailableError, ProviderQuotaExhausted):
    """The key's searches or throughput are used up (429, or "out of
    searches"), or a cooldown from that is still running."""


class SerpApiCoolingDown(SerpApiUnavailableError):
    """Raised without calling SerpAPI while a rejected-key cooldown runs."""


class SerpApiKeyHealth:
    """Cooldown for one SerpAPI key, shared by every client using it
    (the container hands out one per key); thread-safe."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._available_at = 0.0
        self._reason: CooldownReason | None = None

    def cooling_down(self) -> CooldownReason | None:
        """Why the key is resting now, or None when it may be called."""

        with self._lock:
            return self._reason if self._available_at > self._clock() else None

    def seconds_until_available(self) -> float:
        with self._lock:
            return max(self._available_at - self._clock(), 0.0)

    def cool_down(self, seconds: float, reason: CooldownReason) -> None:
        with self._lock:
            until = self._clock() + seconds

            if until > self._available_at:
                self._available_at = until
                self._reason = reason


def _build_http_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=_RETRY_STRATEGY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class SerpApiClient:
    """One SerpAPI key and its own HTTP session; thread-safe."""

    def __init__(
        self,
        *,
        api_key: str,
        session: requests.Session | None = None,
        timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
        health: SerpApiKeyHealth | None = None,
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
        self._health = health or SerpApiKeyHealth()

    def __repr__(self) -> str:
        # The key is never part of the representation.
        return "SerpApiClient()"

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def search(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The response JSON.

        Raises ``SerpApiQuotaExhausted`` when the key's searches are used up
        (also, without calling SerpAPI, while that cooldown runs), and
        ``SerpApiUnavailableError`` on any other HTTP, transport or parsing
        failure. The key is added here and never logged.
        """

        resting = self._health.cooling_down()

        if resting == "quota":
            raise SerpApiQuotaExhausted("SerpAPI searches are used up for now.")

        if resting == "auth":
            raise SerpApiCoolingDown("SerpAPI rejected the key recently.")

        request_params = {**params, "api_key": self._api_key}

        try:
            response = self._session.get(_SERPAPI_URL, params=request_params, timeout=self._timeout_seconds)
        except requests.RequestException as exc:
            safe_log(logger, logging.WARNING, "provider.serpapi.transport_failed")
            raise SerpApiUnavailableError("SerpAPI request failed.") from exc

        try:
            payload = response.json()
        except ValueError:
            payload = None

        error = payload.get("error") if isinstance(payload, dict) else None

        if response.status_code == 429 or (isinstance(error, str) and _OUT_OF_SEARCHES.search(error)):
            self._quota_exhausted(response, error)

        if response.status_code in {401, 403}:
            self._health.cool_down(AUTH_FAILURE_COOLDOWN, "auth")
            safe_log(logger, logging.WARNING, "provider.serpapi.key_rejected", status_code=response.status_code)
            raise SerpApiUnavailableError("SerpAPI rejected the key.")

        if response.status_code != 200:
            safe_log(logger, logging.WARNING, "provider.serpapi.http_failed", status_code=response.status_code)
            raise SerpApiUnavailableError("SerpAPI request failed.")

        if payload is None:
            safe_log(logger, logging.WARNING, "provider.serpapi.invalid_json")
            raise SerpApiUnavailableError("SerpAPI returned invalid JSON.")

        if not isinstance(payload, dict):
            raise SerpApiUnavailableError("SerpAPI returned an invalid payload.")

        return payload

    def _quota_exhausted(self, response: requests.Response, error: Any) -> None:
        out_of_searches = isinstance(error, str) and bool(_OUT_OF_SEARCHES.search(error))
        cooldown = (
            SEARCHES_EXHAUSTED_COOLDOWN
            if out_of_searches
            else _retry_after_seconds(response.headers.get("Retry-After"))
        )
        self._health.cool_down(cooldown, "quota")
        # The error text is SerpAPI's own; only its kind is logged.
        safe_log(
            logger,
            logging.WARNING,
            "provider.serpapi.searches_exhausted" if out_of_searches else "provider.serpapi.rate_limited",
            status_code=response.status_code,
        )
        raise SerpApiQuotaExhausted("SerpAPI searches are used up for now.")


def _retry_after_seconds(value: str | None) -> float:
    """Retry-After in seconds, bounded; the HTTP-date form gets the default."""

    try:
        seconds = float(value) if value is not None else DEFAULT_RATE_LIMIT_COOLDOWN
    except ValueError:
        seconds = DEFAULT_RATE_LIMIT_COOLDOWN

    if seconds != seconds or seconds <= 0:  # NaN or nonsense
        seconds = DEFAULT_RATE_LIMIT_COOLDOWN

    return min(seconds, _MAX_RATE_LIMIT_COOLDOWN)


__all__ = [
    "AUTH_FAILURE_COOLDOWN",
    "DEFAULT_RATE_LIMIT_COOLDOWN",
    "SEARCHES_EXHAUSTED_COOLDOWN",
    "SerpApiClient",
    "SerpApiCoolingDown",
    "SerpApiKeyHealth",
    "SerpApiQuotaExhausted",
    "SerpApiUnavailableError",
]
