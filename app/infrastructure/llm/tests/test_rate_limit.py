"""Sliding-window pacing for Gemini's per-minute quota, on a fake clock."""

import asyncio

import pytest

from app.infrastructure.llm import rate_limit
from app.infrastructure.llm.rate_limit import SlidingWindowRateLimiter


class FakeTime:
    """A clock that only moves when the limiter sleeps."""

    def __init__(self):
        self.now = 1_000.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 3))
        self.now += seconds

    async def async_sleep(self, seconds):
        self.sleep(seconds)


@pytest.fixture
def fake_time(monkeypatch):
    fake = FakeTime()
    monkeypatch.setattr(rate_limit.time, "sleep", fake.sleep)
    monkeypatch.setattr(rate_limit.asyncio, "sleep", fake.async_sleep)
    return fake


def limiter(fake_time, max_requests=3, window_seconds=60.0):
    return SlidingWindowRateLimiter(max_requests=max_requests, window_seconds=window_seconds, clock=fake_time.clock)


def test_calls_within_the_quota_never_wait(fake_time):
    pacer = limiter(fake_time)

    assert all(pacer.acquire() for _ in range(3))
    # Unlike a token bucket, the first call is not delayed either.
    assert fake_time.sleeps == []


def test_a_call_over_the_quota_waits_for_the_oldest_slot(fake_time):
    pacer = limiter(fake_time)

    for _ in range(3):
        pacer.acquire()
        fake_time.now += 10

    assert pacer.acquire()
    # Calls started at 0, 10 and 20 s; the 4th may start at 60 s, 30 s later.
    assert fake_time.sleeps == [30.0]


def test_no_window_ever_holds_more_than_the_quota(fake_time):
    pacer = limiter(fake_time, max_requests=15)
    started = []

    for _ in range(40):
        pacer.acquire()
        started.append(fake_time.now)

    for first in started:
        assert sum(first <= t < first + 60 for t in started) <= 15


def test_non_blocking_acquire_reports_a_full_window(fake_time):
    pacer = limiter(fake_time, max_requests=1)

    assert pacer.acquire(blocking=False)
    assert not pacer.acquire(blocking=False)
    assert fake_time.sleeps == []


def test_sync_and_async_calls_share_one_window(fake_time):
    pacer = limiter(fake_time, max_requests=2)

    pacer.acquire()
    assert asyncio.run(pacer.aacquire())
    assert asyncio.run(pacer.aacquire())

    assert fake_time.sleeps == [60.0]


@pytest.mark.parametrize("kwargs", [{"max_requests": 0}, {"max_requests": 5, "window_seconds": 0}])
def test_invalid_limits_are_refused(kwargs):
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(**kwargs)

