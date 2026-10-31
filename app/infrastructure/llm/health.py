"""Skip a model backend while it cannot answer.

When a provider says a key's quota is used up, or that the key is invalid,
every further call to it fails the same way until the quota resets or the key
is fixed. Trying it first on every call wastes a request and a limiter slot,
and delays the backup that will answer anyway. ``BackendHealth`` remembers
such a failure and refuses calls until the cooldown ends; the next call after
that tries the backend again.

Transient failures (timeouts, server errors, unreadable output) never start a
cooldown: the next call should try the backend again.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import openai
from google.api_core import exceptions as google_errors
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from app.shared.observability import log_event


logger = logging.getLogger(__name__)

# Gemini's daily quotas reset at midnight Pacific time.
GEMINI_QUOTA_TIMEZONE = ZoneInfo("America/Los_Angeles")
# Per-minute quota without a usable "retry in" hint.
DEFAULT_RATE_LIMIT_COOLDOWN = 60.0
# Never trust a hint shorter than this: a backend that just refused would
# otherwise be retried on the very next call.
MIN_COOLDOWN = 10.0
# An invalid or blocked key needs a person to fix it; check again now and then.
AUTH_FAILURE_COOLDOWN = 600.0
# OpenRouter's shared free models are often busy for a while.
OPENROUTER_BUSY_COOLDOWN = 30.0

_RETRY_HINT = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_GEMINI_INVALID_KEY = re.compile(r"api key not valid|api_key_invalid|permission denied", re.IGNORECASE)


class BackendCoolingDown(RuntimeError):
    """Raised instead of calling a backend that is cooling down."""


class BackendHealth:
    """Cooldown state for one provider, key and model; thread-safe."""

    def __init__(self, *, provider: str, model: str, clock: Callable[[], float] = time.time) -> None:
        # The key is deliberately not stored: this object appears in logs.
        self.provider = provider
        self.model = model
        self._clock = clock
        self._lock = threading.Lock()
        self._available_at = 0.0
        # Bumped by every cooldown. A call that started before the latest one
        # (it was already in flight) must not clear it when it succeeds.
        self._cooldowns = 0

    def __repr__(self) -> str:
        return f"BackendHealth(provider={self.provider!r}, model={self.model!r})"

    def seconds_until_available(self) -> float:
        with self._lock:
            return max(self._available_at - self._clock(), 0.0)

    def ensure_available(self) -> int:
        """Raise ``BackendCoolingDown`` while the cooldown lasts; otherwise
        return the token to pass to ``record_success`` when the call is done."""

        with self._lock:
            remaining = self._available_at - self._clock()
            token = self._cooldowns

        if remaining > 0:
            raise BackendCoolingDown(f"{self.provider} {self.model} is cooling down for {remaining:.0f} s")

        return token

    def record_success(self, token: int) -> None:
        """Clear an expired cooldown, unless the call began before the latest one."""

        with self._lock:
            if token != self._cooldowns or self._available_at == 0:
                return

            self._available_at = 0.0

        self._log("recovered")

    def record_failure(self, error: BaseException) -> None:
        """Start a cooldown when ``error`` means the backend can't answer for now."""

        decision = cooldown_for(error, provider=self.provider, now=self._clock())

        if decision is None:
            return

        seconds, reason = decision

        with self._lock:
            self._available_at = max(self._available_at, self._clock() + seconds)
            self._cooldowns += 1

        self._log("cooling_down", reason=reason, seconds=seconds)

    def _log(self, outcome: str, *, reason: str | None = None, seconds: float | None = None) -> None:
        log_event(
            logger,
            "llm.backend_health",
            stage=f"llm.{self.provider}",
            outcome=outcome,
            issue_codes=[reason] if reason else [],
            reason=f"{self.model}" + (f" for {seconds:.0f} s" if seconds is not None else ""),
            level=logging.WARNING if outcome == "cooling_down" else logging.INFO,
        )


def cooldown_for(error: BaseException, *, provider: str, now: float) -> tuple[float, str] | None:
    """(seconds, reason code) to skip a backend after ``error``, or None."""

    if provider == "gemini":
        return _gemini_cooldown(error, now)

    if provider == "openrouter":
        return _openrouter_cooldown(error, now)

    return None


def _gemini_cooldown(error: BaseException, now: float) -> tuple[float, str] | None:
    text = str(error)

    if isinstance(error, google_errors.ResourceExhausted):
        if "perday" in text.replace("_", "").casefold():
            return max(_seconds_to_next_midnight(now, GEMINI_QUOTA_TIMEZONE), MIN_COOLDOWN), "llm.quota_daily"

        return _hinted_cooldown(text), "llm.quota_per_minute"

    if isinstance(error, (google_errors.PermissionDenied, google_errors.Unauthenticated)):
        return AUTH_FAILURE_COOLDOWN, "llm.key_rejected"

    # The Gemini client wraps InvalidArgument; only a key problem is lasting.
    if isinstance(error, ChatGoogleGenerativeAIError) and _GEMINI_INVALID_KEY.search(text):
        return AUTH_FAILURE_COOLDOWN, "llm.key_rejected"

    return None


def _openrouter_cooldown(error: BaseException, now: float) -> tuple[float, str] | None:
    if isinstance(error, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return AUTH_FAILURE_COOLDOWN, "llm.key_rejected"

    if not isinstance(error, openai.RateLimitError):
        return None

    reset = _openrouter_reset(error, now)

    # "temporarily rate-limited upstream": the shared free model is busy.
    if "upstream" in str(error).casefold():
        return OPENROUTER_BUSY_COOLDOWN, "llm.provider_busy"

    if reset is not None:
        return max(reset, MIN_COOLDOWN), "llm.quota"

    return DEFAULT_RATE_LIMIT_COOLDOWN, "llm.quota"


def _openrouter_reset(error: openai.RateLimitError, now: float) -> float | None:
    """Seconds until OpenRouter's limit resets, from its ``X-RateLimit-Reset``
    header (epoch milliseconds)."""

    try:
        reset_ms = float(error.response.headers.get("X-RateLimit-Reset"))
    except (AttributeError, TypeError, ValueError):
        return None

    seconds = reset_ms / 1000 - now
    return seconds if seconds > 0 else None


def _hinted_cooldown(text: str) -> float:
    match = _RETRY_HINT.search(text)
    seconds = float(match.group(1)) if match else DEFAULT_RATE_LIMIT_COOLDOWN
    return max(seconds, MIN_COOLDOWN)


def _seconds_to_next_midnight(now: float, zone: ZoneInfo) -> float:
    local = datetime.fromtimestamp(now, zone)
    midnight = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), zone)
    return midnight.timestamp() - now


__all__ = [
    "AUTH_FAILURE_COOLDOWN",
    "BackendCoolingDown",
    "BackendHealth",
    "cooldown_for",
]
