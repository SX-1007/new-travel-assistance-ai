"""Planning transaction: booking → activities → routes → validation.

One call builds a complete itinerary candidate and returns it only if the
deterministic validator qualifies it. A rejected candidate is repaired, not
rebuilt:

* Stages that passed are kept. Each validator issue is traced to the
  earliest stage that can fix it (and, where the path says so, to its day),
  and the next attempt starts at that stage.
* Activity issues re-plan only the affected days; route issues re-route only
  the affected days. Days whose activities changed are always re-routed.
* Booking issues are retried only when a retry can change the result: once
  with a fresh search if the bookings came from the budget check, and once
  if a search found nothing (provider outages also look like "nothing").
  Otherwise the same cached search results would fail the same way.
* A stage that can't succeed is not followed by the next one: a missing
  booking or an empty day already disqualifies the candidate, so no activity
  or routing calls are spent on it.
* The transaction stops early on unplannable input, and when an attempt
  ends with exactly the same issues as the one before it.

The final candidate always passes the full validator; retries never weaken
the checks.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Final, Literal, TypeVar

from app.modules.itinerary.models import (
    TripRequirements,
    ValidationIssue,
    ValidationReport,
)
from app.shared.observability import (
    LogContext,
    log_event,
)
from app.modules.itinerary.ports import (
    BookingGap,
    PlanningInputError,
    PlanningStages,
)
from app.modules.itinerary.validators import validate_itinerary_candidate


logger = logging.getLogger(__name__)

PLANNING_MAX_ATTEMPTS: Final = 3

_T = TypeVar("_T")

CandidateValidator = Callable[[TripRequirements, Any, Any], ValidationReport]

PlanningStatus = Literal["validated", "unavailable"]

StopReason = Literal[
    "validated",
    "invalid_input",
    "booking_unrecoverable",
    "no_progress",
    "attempts_exhausted",
]


class Stage(IntEnum):
    """Planning stages in run order; a retry reruns its stage and every later one."""

    BOOKING = 1
    ACTIVITIES = 2
    ROUTES = 3

    @property
    def label(self) -> str:
        return self.name.lower()


_GAP_ISSUES: Final[Mapping[BookingGap, str]] = {
    "outbound_flight": "flight.outbound.missing",
    "return_flight": "flight.return.missing",
    "hotel": "hotel.night.missing",
}

_ROUTE_PREFIXES: Final = ("map.", "maps.", "itinerary.route.")
_ACTIVITY_PREFIXES: Final = ("activity.", "day.activities.")
_ACTIVITY_CODES: Final = frozenset({"itinerary.city.coverage.missing"})

_ITINERARY_INDEX: Final = re.compile(r"^itinerary\[(\d+)\]")
_MAP_DAY: Final = re.compile(r"^maps\[(\d+)\]")


@dataclass(frozen=True, slots=True)
class PlanningResult:
    """Outcome of one planning transaction."""

    status: PlanningStatus
    attempts: int
    stop_reason: StopReason
    # Report of the last attempt (empty when validated).
    report: ValidationReport
    # Every attempt's issues, in order, for diagnostics.
    issues: tuple[ValidationIssue, ...]
    itinerary: list[dict[str, Any]] | None = None
    maps: dict[int, dict[str, Any]] | None = None
    # How many times each stage ran, e.g. {"booking": 1, "activities": 2}.
    stage_runs: Mapping[str, int] = field(default_factory=dict)

    @property
    def validated(self) -> bool:
        return self.status == "validated"


@dataclass(frozen=True, slots=True)
class _Step:
    """Where the next attempt starts."""

    stage: Stage
    # ACTIVITIES only; None re-plans every day.
    activity_days: frozenset[int] | None = None
    # BOOKING only; False forces a fresh search.
    reuse_evidence: bool = True


class _StageFailed(Exception):
    def __init__(self, stage: Stage, retry: _Step, error: Exception) -> None:
        super().__init__(stage.label)
        self.stage = stage
        self.retry = retry
        self.error = error


class ItineraryPlanningService:
    """Build one validated itinerary candidate from three planning stages."""

    def __init__(
        self,
        *,
        max_attempts: int = PLANNING_MAX_ATTEMPTS,
        validator: CandidateValidator = validate_itinerary_candidate,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self._max_attempts = max_attempts
        self._validator = validator
        self._clock = clock

    def build(
        self,
        requirements: TripRequirements,
        stages: PlanningStages,
        *,
        max_attempts: int | None = None,
        log_context: LogContext | None = None,
    ) -> PlanningResult:
        """Run the transaction; ``max_attempts`` caps this call only."""

        limit = self._max_attempts if max_attempts is None else max_attempts

        if limit < 1:
            raise ValueError("max_attempts must be at least 1")

        context = (log_context or LogContext()).with_country(
            requirements.destination_country_code
        )

        return _Transaction(
            requirements,
            stages,
            validator=self._validator,
            context=context,
            clock=self._clock,
        ).run(limit)


def classify_issues(
    issues: Sequence[ValidationIssue],
    itinerary: Sequence[Mapping[str, Any]],
) -> dict[Stage, frozenset[int] | None]:
    """Earliest stage that can fix each issue, with the days it concerns.

    ``None`` means every day (or the issue names no day). Anything not
    clearly an activity or route issue belongs to booking, the stage every
    other one is built on.
    """

    found: dict[Stage, set[int] | None] = {}

    for issue in issues:
        stage = _stage_for(issue.code)
        day = None if stage is Stage.BOOKING else _day_for(issue.path, itinerary)

        if stage in found and found[stage] is None:
            continue

        if day is None:
            found[stage] = None
        else:
            found.setdefault(stage, set()).add(day)  # type: ignore[union-attr]

    return {
        stage: None if days is None else frozenset(days)
        for stage, days in found.items()
    }


def _stage_for(code: str) -> Stage:
    if code.startswith(_ROUTE_PREFIXES):
        return Stage.ROUTES

    if code.startswith(_ACTIVITY_PREFIXES) or code in _ACTIVITY_CODES:
        return Stage.ACTIVITIES

    return Stage.BOOKING


def _day_for(path: str, itinerary: Sequence[Mapping[str, Any]]) -> int | None:
    if match := _MAP_DAY.match(path):
        return int(match.group(1))

    if match := _ITINERARY_INDEX.match(path):
        index = int(match.group(1))

        if index < len(itinerary):
            day = itinerary[index].get("day")

            if isinstance(day, int) and not isinstance(day, bool) and day > 0:
                return day

    return None


class _Transaction:
    """Mutable state of one ``build`` call; never shared between calls."""

    def __init__(
        self,
        requirements: TripRequirements,
        stages: PlanningStages,
        *,
        validator: CandidateValidator,
        context: LogContext,
        clock: Callable[[], float],
    ) -> None:
        self._requirements = requirements
        self._stages = stages
        self._validator = validator
        self._context = context
        self._clock = clock

        self._itinerary: list[dict[str, Any]] = []
        self._maps: dict[int, dict[str, Any]] = {}
        # Days whose route is missing or out of date; None means every day.
        self._stale_routes: set[int] | None = None
        self._reused_evidence = False
        self._booking_gaps: tuple[BookingGap, ...] = ()
        self._gap_retry_used = False
        self._stage_runs: Counter[str] = Counter()

    # ── Attempt loop ────────────────────────────────────────────────

    def run(self, limit: int) -> PlanningResult:
        started = self._clock()
        step = _Step(Stage.BOOKING)
        reports: list[ValidationReport] = []
        previous: frozenset[tuple[str, str]] | None = None

        for attempt in range(1, limit + 1):
            self._log("planning.attempt", attempt, step.stage.label, "started")

            try:
                report = self._attempt(attempt, step)

            except PlanningInputError as exc:
                reports.append(_input_error_report(exc))
                return self._finish(attempt, reports, "invalid_input", started)

            except _StageFailed as failure:
                report = _stage_error_report(failure)
                reports.append(report)
                next_step: _Step | StopReason = failure.retry

            else:
                reports.append(report)

                if report.qualified:
                    return self._finish(attempt, reports, "validated", started)

                next_step = self._next_step(report)

            if isinstance(next_step, str):
                return self._finish(attempt, reports, next_step, started)

            fingerprint = frozenset((issue.code, issue.path) for issue in report.issues)

            if fingerprint == previous:
                return self._finish(attempt, reports, "no_progress", started)

            previous = fingerprint

            if attempt < limit:
                self._log(
                    "planning.retry",
                    attempt,
                    next_step.stage.label,
                    "scheduled",
                    issue_codes=[issue.code for issue in report.issues],
                )

            step = next_step

        return self._finish(limit, reports, "attempts_exhausted", started)

    def _attempt(self, attempt: int, step: _Step) -> ValidationReport:
        activity_days = step.activity_days

        if step.stage is Stage.BOOKING:
            reuse = step.reuse_evidence
            skeleton = self._run(
                attempt,
                Stage.BOOKING,
                _Step(Stage.BOOKING, reuse_evidence=False),
                lambda: self._stages.skeleton(reuse_evidence=reuse),
            )
            self._itinerary = list(skeleton.itinerary)
            self._maps = {}
            self._stale_routes = None
            self._reused_evidence = skeleton.reused_evidence
            self._booking_gaps = tuple(skeleton.missing)
            activity_days = None

            if skeleton.missing:
                # Activities and routes can't rescue a missing booking.
                return _gap_report(skeleton.missing)

        if step.stage <= Stage.ACTIVITIES:
            days = activity_days
            activities = self._run(
                attempt,
                Stage.ACTIVITIES,
                _Step(Stage.ACTIVITIES, activity_days=days),
                lambda: self._stages.activities(self._itinerary, days=days),
            )
            self._itinerary = list(activities.itinerary)
            self._mark_stale(days)

            if activities.missing_days:
                # An empty day is rejected anyway; don't route this candidate.
                return _empty_day_report(activities.missing_days, self._itinerary)

        route_days = None if self._stale_routes is None else frozenset(self._stale_routes)

        if route_days is None or route_days:
            routes = self._run(
                attempt,
                Stage.ROUTES,
                _Step(Stage.ROUTES),
                lambda: self._stages.routes(self._itinerary, days=route_days),
            )
            self._itinerary = list(routes.itinerary)
            self._maps = (
                dict(routes.maps)
                if route_days is None
                else {**self._maps, **routes.maps}
            )
            self._stale_routes = set()

        return self._validate(attempt)

    def _next_step(self, report: ValidationReport) -> _Step | StopReason:
        """Cheapest retry that can fix ``report``, or why none can."""

        by_stage = classify_issues(report.issues, self._itinerary)

        if Stage.BOOKING in by_stage:
            if self._reused_evidence:
                return _Step(Stage.BOOKING, reuse_evidence=False)

            if self._booking_gaps and not self._gap_retry_used:
                self._gap_retry_used = True
                return _Step(Stage.BOOKING, reuse_evidence=False)

            return "booking_unrecoverable"

        if Stage.ROUTES in by_stage:
            self._mark_stale(by_stage[Stage.ROUTES])

        if Stage.ACTIVITIES in by_stage:
            return _Step(Stage.ACTIVITIES, activity_days=by_stage[Stage.ACTIVITIES])

        return _Step(Stage.ROUTES)

    # ── Stage execution ─────────────────────────────────────────────

    def _run(
        self,
        attempt: int,
        stage: Stage,
        retry: _Step,
        call: Callable[[], _T],
    ) -> _T:
        started = self._clock()
        self._stage_runs[stage.label] += 1

        try:
            result = call()

        except PlanningInputError:
            self._log(
                "planning.stage",
                attempt,
                stage.label,
                "failed",
                issue_codes=["planning.input.invalid"],
                started=started,
            )
            raise

        except Exception as exc:
            self._log(
                "planning.stage",
                attempt,
                stage.label,
                "failed",
                issue_codes=[f"planning.{stage.label}.exception"],
                started=started,
            )
            raise _StageFailed(stage, retry, exc) from exc

        self._log("planning.stage", attempt, stage.label, "completed", started=started)
        return result

    def _validate(self, attempt: int) -> ValidationReport:
        started = self._clock()
        report = self._validator(
            self._requirements,
            self._itinerary,
            self._maps,
        )
        self._log(
            "planning.validation",
            attempt,
            "validation",
            "approved" if report.qualified else "rejected",
            issue_codes=[issue.code for issue in report.issues],
            started=started,
        )
        return report

    def _mark_stale(self, days: frozenset[int] | None) -> None:
        if days is None:
            self._stale_routes = None
        elif self._stale_routes is not None:
            self._stale_routes |= days

    # ── Result ──────────────────────────────────────────────────────

    def _finish(
        self,
        attempts: int,
        reports: Sequence[ValidationReport],
        reason: StopReason,
        started: float,
    ) -> PlanningResult:
        validated = reason == "validated"
        report = reports[-1] if reports else ValidationReport()
        issues = tuple(issue for item in reports for issue in item.issues)

        self._log(
            "planning.complete",
            attempts,
            "transaction",
            "validated" if validated else "unavailable",
            issue_codes=[issue.code for issue in issues],
            started=started,
            reason=reason,
        )

        return PlanningResult(
            status="validated" if validated else "unavailable",
            attempts=attempts,
            stop_reason=reason,
            report=report,
            issues=issues,
            itinerary=self._itinerary if validated else None,
            maps=self._maps if validated else None,
            stage_runs=dict(self._stage_runs),
        )

    def _log(
        self,
        event: str,
        attempt: int,
        stage: str,
        outcome: str,
        *,
        issue_codes: Sequence[str] = (),
        started: float | None = None,
        reason: str | None = None,
    ) -> None:
        log_event(
            logger,
            event,
            context=self._context,
            attempt=attempt,
            stage=stage,
            outcome=outcome,
            issue_codes=issue_codes,
            elapsed_ms=None if started is None else (self._clock() - started) * 1000,
            reason=reason,
        )


# ── Synthetic reports ───────────────────────────────────────────────
# Same codes the validator uses, so callers see one vocabulary.


def _gap_report(gaps: Sequence[BookingGap]) -> ValidationReport:
    return ValidationReport(
        issues=tuple(
            ValidationIssue(
                code=_GAP_ISSUES[gap],
                path=f"booking.{gap}",
                message=f"No {gap.replace('_', ' ')} could be found for this trip.",
            )
            for gap in gaps
        )
    )


def _empty_day_report(
    missing_days: Sequence[int],
    itinerary: Sequence[Mapping[str, Any]],
) -> ValidationReport:
    index_by_day = {
        day.get("day"): index for index, day in enumerate(itinerary)
    }

    return ValidationReport(
        issues=tuple(
            ValidationIssue(
                code="day.activities.empty",
                path=(
                    f"itinerary[{index_by_day[day]}].activities"
                    if day in index_by_day
                    else "itinerary"
                ),
                message=f"No verified activity could be found for day {day}.",
            )
            for day in missing_days
        )
    )


def _stage_error_report(failure: _StageFailed) -> ValidationReport:
    label = failure.stage.label

    return ValidationReport(
        issues=(
            ValidationIssue(
                code=f"planning.{label}.exception",
                path=f"stage.{label}",
                message=f"The {label} stage failed with {type(failure.error).__name__}.",
            ),
        )
    )


def _input_error_report(error: PlanningInputError) -> ValidationReport:
    return ValidationReport(
        issues=(
            ValidationIssue(
                code="planning.input.invalid",
                path="trip",
                message=str(error)[:200] or "The trip details can't be planned.",
            ),
        )
    )


__all__ = [
    "PLANNING_MAX_ATTEMPTS",
    "ItineraryPlanningService",
    "PlanningResult",
    "PlanningStatus",
    "Stage",
    "StopReason",
    "classify_issues",
]
