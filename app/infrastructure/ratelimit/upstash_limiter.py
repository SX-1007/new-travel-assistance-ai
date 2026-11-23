"""Fixed-window request counters in Upstash Redis.

One counter per (rule, user) and window: the first request in a window
creates the key with the window's length as its expiry, each request adds
one, and the window ends when the key expires. One Lua call does the add
and the expiry together, so a counter can never be left without an expiry.

The user id is hashed into the key: Redis never holds raw user ids.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Final, Protocol, Sequence

from app.infrastructure.cache.upstash import CacheUnavailableError


_KEY_PREFIX: Final = "travel:ratelimit"

# INCR, and set the window's expiry if the key has none (new key, or one
# somehow left without it). Returns {count, milliseconds left}.
_HIT_SCRIPT: Final = """
local count = redis.call('INCR', KEYS[1])
local ttl = redis.call('PTTL', KEYS[1])
if ttl < 0 then
  redis.call('PEXPIRE', KEYS[1], ARGV[1])
  ttl = tonumber(ARGV[1])
end
return {count, ttl}
"""


class RateLimitStoreUnavailable(RuntimeError):
    """The counter store could not be reached or answered nonsense."""


class _RedisCommands(Protocol):
    def execute(self, command: Sequence[str | int]) -> object: ...


@dataclass(frozen=True, slots=True)
class WindowCount:
    """Requests so far in the current window, and seconds until it ends."""

    count: int
    seconds_left: int


class UpstashFixedWindowLimiter:
    """Counts requests per rule and user; thread-safe (the client is)."""

    def __init__(self, *, client: _RedisCommands, prefix: str = _KEY_PREFIX) -> None:
        self._client = client
        self._prefix = prefix

    def close(self) -> None:
        close = getattr(self._client, "close", None)

        if callable(close):
            close()

    def hit(self, rule: str, subject: str, *, window_seconds: int) -> WindowCount:
        """Count one request; raises ``RateLimitStoreUnavailable``."""

        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        key = f"{self._prefix}:{rule}:{_digest(subject)}"

        try:
            result = self._client.execute(["EVAL", _HIT_SCRIPT, 1, key, window_seconds * 1000])
        except CacheUnavailableError as exc:
            raise RateLimitStoreUnavailable("Rate-limit store unavailable.") from exc

        if not isinstance(result, list) or len(result) != 2:
            raise RateLimitStoreUnavailable("Rate-limit store answered an unexpected value.")

        try:
            count, ttl_ms = int(result[0]), int(result[1])
        except (TypeError, ValueError) as exc:
            raise RateLimitStoreUnavailable("Rate-limit store answered an unexpected value.") from exc

        return WindowCount(count=count, seconds_left=max(math.ceil(ttl_ms / 1000), 1))


def _digest(subject: str) -> str:
    return hashlib.sha256(subject.encode("utf-8")).hexdigest()[:32]


__all__ = [
    "RateLimitStoreUnavailable",
    "UpstashFixedWindowLimiter",
    "WindowCount",
]
