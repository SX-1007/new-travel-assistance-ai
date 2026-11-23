"""Per-user request limits and the limit on trips planned at once.

Limits (fixed windows, counted per signed-in user):

* trip form submissions: 10 a day (a budget check or a confirmation each);
* new trip plans: 3 a day (counted only when planning actually starts);
* chat messages: 30 an hour;
* map pin lookups: 20 a minute (each can be a paid search).

Over a limit the request gets 429 with ``Retry-After`` (``api/errors.py``).
When the counter store can't be reached, production refuses form, plan and
chat requests (503) rather than let them through uncounted; development
lets them through. Map pins always go through: a missing pin costs little
and the page never fails on one.

``PlanSlots`` allows one running form submission per user and at most
``max_running`` trips being planned at once by this process. Like the
session lock (B3) it is per process; several workers would need a shared
lock.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Final, Protocol

from app.shared.exceptions import (
    PlanAlreadyRunning,
    PlanningCapacityFull,
    RateLimitExceeded,
    RateLimitUnavailable,
)
from app.shared.observability import log_event


logger = logging.getLogger(__name__)

_MINUTE: Final = 60
_HOUR: Final = 3600
_DAY: Final = 86400


@dataclass(frozen=True, slots=True)
class LimitRule:
    name: str
    limit: int
    window_seconds: int
    # For the message: "<limit> <what> <per>".
    what: str
    per: str
    # Refuse when the counter store is down (production only).
    fail_closed: bool = True


FORM_SUBMITS: Final = LimitRule("form", 10, _DAY, "trip form submissions", "a day")
PLANS: Final = LimitRule("plan", 3, _DAY, "new trip plans", "a day")
CHAT_MESSAGES: Final = LimitRule("chat", 30, _HOUR, "chat messages", "an hour")
MAP_PINS: Final = LimitRule("pin", 20, _MINUTE, "map pin lookups", "a minute", fail_closed=False)


class WindowCounter(Protocol):
    """Counts one request; raises any exception when the store fails."""

    def hit(self, rule: str, subject: str, *, window_seconds: int) -> "CountLike": ...


class CountLike(Protocol):
    count: int
    seconds_left: int


class RequestLimits:
    """Checks the per-user limits. Blocking (one store call): run in a thread."""

    def __init__(self, counter: WindowCounter | None, *, fail_closed: bool) -> None:
        # No counter: limits are off (development without Upstash).
        self._counter = counter
        self._fail_closed = fail_closed

    @property
    def enabled(self) -> bool:
        return self._counter is not None

    def check(self, rule: LimitRule, user_id: str) -> None:
        """Count this request; raise if it is over the rule's limit."""

        if self._counter is None:
            return

        try:
            window = self._counter.hit(rule.name, user_id, window_seconds=rule.window_seconds)
        except Exception as exc:
            refuse = self._fail_closed and rule.fail_closed
            log_event(
                logger,
                "rate_limit.unavailable",
                stage=rule.name,
                outcome="refused" if refuse else "allowed",
                reason=type(exc).__name__,
                level=logging.WARNING,
            )

            if refuse:
                raise RateLimitUnavailable from exc

            return

        if window.count > rule.limit:
            log_event(logger, "rate_limit.exceeded", stage=rule.name, outcome="refused")
            raise RateLimitExceeded(
                f"You've reached the limit of {rule.limit} {rule.what} {rule.per}. "
                f"Please try again in {_duration(window.seconds_left)}.",
                retry_after_seconds=window.seconds_left,
            )


class PlanSlots:
    """One running trip submission per user; at most ``max_running`` trips
    being planned at once. Event-loop only: a check and its claim have no
    ``await`` between them, so they can't interleave."""

    def __init__(self, *, max_running: int) -> None:
        if max_running < 1:
            raise ValueError("max_running must be at least 1")

        self._max_running = max_running
        self._users: set[str] = set()
        self._running = 0

    @asynccontextmanager
    async def for_user(self, user_id: str) -> AsyncIterator[None]:
        if user_id in self._users:
            raise PlanAlreadyRunning

        self._users.add(user_id)

        try:
            yield
        finally:
            self._users.discard(user_id)

    @asynccontextmanager
    async def planning(self) -> AsyncIterator[None]:
        if self._running >= self._max_running:
            raise PlanningCapacityFull

        self._running += 1

        try:
            yield
        finally:
            self._running -= 1


def _duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} s"

    minutes = -(-seconds // 60)

    if minutes < 60:
        return f"{minutes} min"

    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


__all__ = [
    "CHAT_MESSAGES",
    "FORM_SUBMITS",
    "LimitRule",
    "MAP_PINS",
    "PLANS",
    "PlanSlots",
    "RequestLimits",
    "WindowCounter",
]
