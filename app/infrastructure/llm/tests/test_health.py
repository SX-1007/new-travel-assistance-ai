"""When a backend is skipped, for how long, and when it is tried again."""

from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import openai
import pytest
from google.api_core import exceptions as google_errors
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from app.infrastructure.llm.health import (
    AUTH_FAILURE_COOLDOWN,
    BackendCoolingDown,
    BackendHealth,
    cooldown_for,
)


PACIFIC = ZoneInfo("America/Los_Angeles")
PER_MINUTE = (
    "429 You exceeded your current quota. quota_id: "
    "GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 22.555s."
)
PER_DAY = "429 You exceeded your current quota. quota_id: GenerateRequestsPerDayPerProjectPerModel-FreeTier"


def at(text):
    return datetime.fromisoformat(text).timestamp()


def openrouter_429(message, reset_at=None):
    headers = {"X-RateLimit-Reset": str(int(reset_at * 1000))} if reset_at else {}
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat"))
    return openai.RateLimitError(message, response=response, body=None)


def openrouter_401():
    response = httpx.Response(401, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat"))
    return openai.AuthenticationError("No auth credentials found", response=response, body=None)


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


# ── Which failures start a cooldown ────────────────────────────────


def test_a_per_minute_quota_follows_googles_retry_hint():
    assert cooldown_for(google_errors.ResourceExhausted(PER_MINUTE), provider="gemini", now=0) == (
        22.555,
        "llm.quota_per_minute",
    )


def test_a_tiny_or_missing_hint_still_waits_a_moment():
    tiny = google_errors.ResourceExhausted("429 quota exceeded. Please retry in 0.4s.")
    bare = google_errors.ResourceExhausted("429 quota exceeded")

    assert cooldown_for(tiny, provider="gemini", now=0)[0] == 10.0
    assert cooldown_for(bare, provider="gemini", now=0)[0] == 60.0


def test_a_daily_quota_waits_until_midnight_pacific():
    now = at("2026-10-31T10:00:00-07:00")

    seconds, reason = cooldown_for(google_errors.ResourceExhausted(PER_DAY), provider="gemini", now=now)

    assert reason == "llm.quota_daily"
    assert datetime.fromtimestamp(now + seconds, PACIFIC).isoformat() == "2026-11-01T00:00:00-07:00"


def test_the_daily_reset_follows_daylight_saving():
    # US clocks go back on 1 Nov 2026, so that Pacific day lasts 25 hours.
    now = at("2026-11-01T00:30:00-07:00")

    seconds, _ = cooldown_for(google_errors.ResourceExhausted(PER_DAY), provider="gemini", now=now)

    assert seconds == pytest.approx(24.5 * 3600)
    assert datetime.fromtimestamp(now + seconds, PACIFIC).isoformat() == "2026-11-02T00:00:00-08:00"


@pytest.mark.parametrize(
    "error",
    [
        ChatGoogleGenerativeAIError("Invalid argument provided to Gemini: 400 API key not valid. Please pass a valid API key."),
        google_errors.PermissionDenied("403 The caller does not have permission"),
        google_errors.Unauthenticated("401 Request had invalid authentication credentials"),
    ],
)
def test_a_rejected_gemini_key_is_left_alone_for_a_while(error):
    assert cooldown_for(error, provider="gemini", now=0) == (AUTH_FAILURE_COOLDOWN, "llm.key_rejected")


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("read timed out"),
        google_errors.InternalServerError("500 internal"),
        google_errors.ServiceUnavailable("503 overloaded"),
        ChatGoogleGenerativeAIError("Invalid argument provided to Gemini: 400 Request contains an invalid argument."),
        ValueError("could not parse the structured output"),
    ],
)
def test_transient_failures_never_start_a_cooldown(error):
    assert cooldown_for(error, provider="gemini", now=0) is None


def test_a_busy_free_openrouter_model_is_skipped_briefly():
    error = openrouter_429("google/gemma-4-31b-it:free is temporarily rate-limited upstream.")

    assert cooldown_for(error, provider="openrouter", now=0) == (30.0, "llm.provider_busy")


def test_an_openrouter_limit_waits_for_its_reset_header():
    now = at("2026-10-31T10:00:00+00:00")
    error = openrouter_429("Rate limit exceeded: free-models-per-day", reset_at=now + 3600)

    assert cooldown_for(error, provider="openrouter", now=now) == (pytest.approx(3600), "llm.quota")


def test_a_rejected_openrouter_key_is_left_alone_for_a_while():
    assert cooldown_for(openrouter_401(), provider="openrouter", now=0) == (AUTH_FAILURE_COOLDOWN, "llm.key_rejected")


# ── The cooldown itself ────────────────────────────────────────────


def test_a_cooling_backend_is_refused_until_the_cooldown_ends():
    clock = Clock(1_000.0)
    health = BackendHealth(provider="gemini", model="gemini-test", clock=clock)

    health.ensure_available()
    health.record_failure(google_errors.ResourceExhausted(PER_MINUTE))

    with pytest.raises(BackendCoolingDown):
        health.ensure_available()

    clock.now += 23
    health.ensure_available()


def test_a_transient_failure_leaves_the_backend_available():
    health = BackendHealth(provider="gemini", model="gemini-test", clock=Clock(0.0))

    health.record_failure(TimeoutError("read timed out"))

    assert health.seconds_until_available() == 0


def test_a_shorter_cooldown_never_cuts_a_longer_one():
    clock = Clock(at("2026-10-31T10:00:00-07:00"))
    health = BackendHealth(provider="gemini", model="gemini-test", clock=clock)

    health.record_failure(google_errors.ResourceExhausted(PER_DAY))
    health.record_failure(google_errors.ResourceExhausted(PER_MINUTE))

    assert health.seconds_until_available() == pytest.approx(14 * 3600)


def test_a_call_after_the_cooldown_clears_it_and_logs_the_recovery(caplog):
    clock = Clock(0.0)
    health = BackendHealth(provider="gemini", model="gemini-test", clock=clock)
    health.record_failure(google_errors.ResourceExhausted(PER_MINUTE))
    clock.now += 23

    with caplog.at_level("INFO"):
        health.record_success(health.ensure_available())

    assert "recovered" in caplog.text


def test_a_call_already_in_flight_cannot_clear_a_newer_cooldown():
    # Found live: a burst of 18 calls, 2 hit the quota, and the 16 that were
    # already running finished afterwards and wiped the cooldown.
    clock = Clock(0.0)
    health = BackendHealth(provider="gemini", model="gemini-test", clock=clock)

    in_flight = health.ensure_available()
    health.record_failure(google_errors.ResourceExhausted(PER_MINUTE))
    health.record_success(in_flight)

    assert health.seconds_until_available() == pytest.approx(22.555)

    with pytest.raises(BackendCoolingDown):
        health.ensure_available()

