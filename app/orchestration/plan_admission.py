"""Admission for a planning run that a chat turn starts.

A new trip from the form is admitted by the API before the graph runs
(``api/routers/form.py``: one submission per user, a free planning slot,
then the daily plan limit). A chat turn can start a full re-plan too (new
dates or cities, an accepted budget), so the planning node asks the same
admission before spending anything. ``GraphRuntime`` marks a new-trip run
as already admitted, so it is never counted twice.

The orchestration layer only knows the ``PlanAdmission`` protocol; the
implementation (Upstash counters, per-process slots) is wired in by the
container.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Final, Protocol

from app.shared.exceptions import (
    PlanAlreadyRunning,
    PlanningCapacityFull,
    RateLimitExceeded,
    RateLimitUnavailable,
)


# ``configurable`` key set by ``GraphRuntime`` on runs the API admitted.
PLAN_ADMITTED_KEY: Final = "plan_admitted"

LIMIT_REACHED_ISSUE: Final = "planning.limit.reached"
ALREADY_RUNNING_ISSUE: Final = "planning.limit.already_running"
CAPACITY_FULL_ISSUE: Final = "planning.capacity.full"
LIMITS_UNAVAILABLE_ISSUE: Final = "planning.limits.unavailable"

_UNCHANGED: Final = "Your trip plan has not been changed."


class PlanAdmission(Protocol):
    """Holds the right to plan for one run.

    Entering raises ``RateLimitExceeded``, ``RateLimitUnavailable``,
    ``PlanAlreadyRunning`` or ``PlanningCapacityFull``; leaving frees the
    slot. Hold it until the planning work has really stopped.
    """

    def admit(self, user_id: str) -> AbstractAsyncContextManager[None]: ...


@dataclass(frozen=True, slots=True)
class PlanRefusal:
    issue_code: str
    reply: str


def plan_admitted(config: Mapping[str, Any] | None) -> bool:
    configurable = config.get("configurable") if isinstance(config, Mapping) else None
    return isinstance(configurable, Mapping) and configurable.get(PLAN_ADMITTED_KEY) is True


def refusal_for(exc: BaseException) -> PlanRefusal | None:
    """The issue code and reply for an admission refusal; None for other errors."""

    if isinstance(exc, RateLimitExceeded):
        # The limiter's own sentence: it names the limit and the wait.
        return PlanRefusal(LIMIT_REACHED_ISSUE, f"{exc.message} {_UNCHANGED}")

    if isinstance(exc, PlanAlreadyRunning):
        return PlanRefusal(
            ALREADY_RUNNING_ISSUE,
            "Another of your trips is being planned right now, so I can't re-plan this one "
            # Not "once that plan is ready": the reply rules read it as a
            # claim about a plan this turn didn't make.
            f"at the same time. {_UNCHANGED} Please try again when it has finished.",
        )

    if isinstance(exc, PlanningCapacityFull):
        return PlanRefusal(
            CAPACITY_FULL_ISSUE,
            f"Several trips are being planned right now, so I can't re-plan yours yet. {_UNCHANGED} "
            "Please try again in a minute.",
        )

    if isinstance(exc, RateLimitUnavailable):
        return PlanRefusal(
            LIMITS_UNAVAILABLE_ISSUE,
            f"I can't check your planning allowance right now, so I didn't re-plan. {_UNCHANGED} "
            "Please try again in a minute.",
        )

    return None


ADMISSION_ERRORS: Final = (RateLimitExceeded, RateLimitUnavailable, PlanAlreadyRunning, PlanningCapacityFull)


__all__ = [
    "ADMISSION_ERRORS",
    "ALREADY_RUNNING_ISSUE",
    "CAPACITY_FULL_ISSUE",
    "LIMITS_UNAVAILABLE_ISSUE",
    "LIMIT_REACHED_ISSUE",
    "PLAN_ADMITTED_KEY",
    "PlanAdmission",
    "PlanRefusal",
    "plan_admitted",
    "refusal_for",
]
