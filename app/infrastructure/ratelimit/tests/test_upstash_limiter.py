"""The fixed-window counter: its Lua script is run here by a fake with the
same meaning (INCR, PTTL, PEXPIRE); the real script runs in Upstash."""

import pytest

from app.infrastructure.cache.upstash import CacheCoolingDown
from app.infrastructure.ratelimit.upstash_limiter import (
    RateLimitStoreUnavailable,
    UpstashFixedWindowLimiter,
)


class FakeRedis:
    def __init__(self):
        self.now_ms = 0
        self.counts = {}
        self.expires = {}
        self.commands = []
        self.fail = None

    def execute(self, command):
        self.commands.append(list(command))
        if self.fail:
            raise self.fail
        name, script, numkeys, key, window_ms = command
        assert name == "EVAL" and numkeys == 1 and "INCR" in script
        if key in self.expires and self.expires[key] <= self.now_ms:
            self.counts.pop(key)
            self.expires.pop(key)
        self.counts[key] = self.counts.get(key, 0) + 1
        if key not in self.expires:
            self.expires[key] = self.now_ms + int(window_ms)
        return [self.counts[key], self.expires[key] - self.now_ms]


def test_counts_per_rule_and_user_within_one_window():
    redis = FakeRedis()
    limiter = UpstashFixedWindowLimiter(client=redis)

    first = limiter.hit("chat", "user-1", window_seconds=3600)
    redis.now_ms = 600_000
    second = limiter.hit("chat", "user-1", window_seconds=3600)
    other_user = limiter.hit("chat", "user-2", window_seconds=3600)
    other_rule = limiter.hit("pin", "user-1", window_seconds=60)

    assert (first.count, second.count, other_user.count, other_rule.count) == (1, 2, 1, 1)
    # The window started with the first request and doesn't move.
    assert second.seconds_left == 3000


def test_a_new_window_starts_when_the_old_one_expires():
    redis = FakeRedis()
    limiter = UpstashFixedWindowLimiter(client=redis)

    limiter.hit("pin", "u", window_seconds=60)
    redis.now_ms = 60_000

    assert limiter.hit("pin", "u", window_seconds=60).count == 1


def test_user_ids_are_hashed_into_the_key():
    redis = FakeRedis()

    UpstashFixedWindowLimiter(client=redis).hit("form", "alice@example.com", window_seconds=60)

    key = redis.commands[0][3]
    assert key.startswith("travel:ratelimit:form:") and "alice" not in key


def test_a_store_failure_or_odd_answer_is_reported():
    redis = FakeRedis()
    redis.fail = CacheCoolingDown("resting")

    with pytest.raises(RateLimitStoreUnavailable):
        UpstashFixedWindowLimiter(client=redis).hit("chat", "u", window_seconds=60)

    class Odd:
        def execute(self, command):
            return "OK"

    with pytest.raises(RateLimitStoreUnavailable):
        UpstashFixedWindowLimiter(client=Odd()).hit("chat", "u", window_seconds=60)
