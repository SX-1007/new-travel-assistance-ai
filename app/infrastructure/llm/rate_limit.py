"""Client-side pacing for Gemini's requests-per-minute quota.

Google counts requests per minute for each project and model. Without pacing,
a burst (one chat turn makes 4-5 calls) goes over the free tier's limit and
the user sees a failed turn. With it, the extra calls wait for the next free
slot instead.

A sliding window matches how the quota is counted: at most ``max_requests``
calls start within any ``window_seconds``. A token bucket (LangChain's
``InMemoryRateLimiter``) either delays the first call or lets a burst through
on top of the steady rate.

The window is per process. With several worker processes, or other apps on
the same project, set the limit below the quota.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable

from langchain_core.rate_limiters import BaseRateLimiter

from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 60.0


class SlidingWindowRateLimiter(BaseRateLimiter):
    """At most ``max_requests`` acquisitions in any ``window_seconds``.

    Thread-safe and usable from sync and async code at once: LangChain calls
    ``acquire`` for ``invoke`` (often from worker threads) and ``aacquire``
    for ``ainvoke`` on the event loop, and both share one window.
    """

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be at least 1")

        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._started: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self, *, blocking: bool = True) -> bool:
        waited = False

        while (wait := self._try_reserve()) > 0:
            if not blocking:
                return False

            waited = self._log_wait(waited)
            time.sleep(wait)

        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        waited = False

        while (wait := self._try_reserve()) > 0:
            if not blocking:
                return False

            waited = self._log_wait(waited)
            await asyncio.sleep(wait)

        return True

    def _try_reserve(self) -> float:
        """Take a slot and return 0, or return the seconds until one frees up."""

        with self._lock:
            now = self._clock()

            while self._started and self._started[0] <= now - self.window_seconds:
                self._started.popleft()

            if len(self._started) < self.max_requests:
                self._started.append(now)
                return 0.0

            # Another waiter may take the slot first; the caller then waits again.
            return max(self._started[0] + self.window_seconds - now, 0.01)

    @staticmethod
    def _log_wait(already_logged: bool) -> bool:
        if not already_logged:
            safe_log(logger, logging.INFO, "llm.rate_limit_wait")

        return True


__all__ = [
    "DEFAULT_WINDOW_SECONDS",
    "SlidingWindowRateLimiter",
]
