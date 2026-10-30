"""Planning transaction retry rules with fake stages and a scripted validator."""

import logging

import pytest

from app.modules.itinerary.models import (
    ExpectedDay,
    TripRequirements,
    ValidationIssue,
    ValidationReport,
)
from app.shared.observability import LogContext, log_event
from app.modules.itinerary.ports import (
    ActivityDraft,
    PlanningInputError,
    RouteDraft,
    SkeletonDraft,
)
from app.modules.itinerary.service import (
    ItineraryPlanningService,
    Stage,
    classify_issues,
)


REQUIREMENTS = TripRequirements(
    origin_country="Malaysia",
    destination_country="Japan",
    destination_country_code="JP",
    cities=("Tokyo",),
    expected_days=(
        ExpectedDay(day=1, date="2026-11-01"),
        ExpectedDay(day=2, date="2026-11-02"),
        ExpectedDay(day=3, date="2026-11-03"),
    ),
    num_people=1,
    base_currency="MYR",
    destination_currency="JPY",
    total_budget=600.0,
    budget_allocation={
        "transportation": 100.0,
        "accommodation": 100.0,
        "food": 100.0,
        "activity": 100.0,
        "shopping": 100.0,
        "emergency_fund": 100.0,
    },
)


def issue(code, path="itinerary"):
    return ValidationIssue(code=code, path=path, message="test")


def report(*issues):
    return ValidationReport(issues=tuple(issues))


OK = report()


class Stages:
    """Records every call; each stage's outcomes can be scripted per call."""

    def __init__(self, *, skeletons=None, activity_results=None, route_errors=()):
        self.calls = []
        self.skeletons = list(skeletons or [])
        self.activity_results = list(activity_results or [])
        self.route_errors = list(route_errors)

    def skeleton(self, *, reuse_evidence):
        self.calls.append(("booking", reuse_evidence))
        outcome = self.skeletons.pop(0) if self.skeletons else SkeletonDraft(itinerary=_days())
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def activities(self, itinerary, *, days):
        self.calls.append(("activities", days))
        outcome = self.activity_results.pop(0) if self.activity_results else None
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, tuple):
            return ActivityDraft(itinerary=list(itinerary), missing_days=outcome)
        return ActivityDraft(itinerary=list(itinerary))

    def routes(self, itinerary, *, days):
        self.calls.append(("routes", days))
        if self.route_errors:
            error = self.route_errors.pop(0)
            if error is not None:
                raise error
        routed = [day["day"] for day in itinerary if days is None or day["day"] in days]
        return RouteDraft(itinerary=list(itinerary), maps={n: {"day": n} for n in routed})


def _days():
    return [{"day": n, "activities": []} for n in (1, 2, 3)]


class Validator:
    def __init__(self, *reports):
        self.reports = list(reports)
        self.seen_maps = []

    def __call__(self, requirements, itinerary, maps):
        self.seen_maps.append(dict(maps))
        return self.reports.pop(0) if self.reports else OK


def build(stages, validator, **kwargs):
    return ItineraryPlanningService(validator=validator).build(REQUIREMENTS, stages, **kwargs)


# ── Happy path ──────────────────────────────────────────────────────


def test_first_qualified_candidate_is_returned():
    stages = Stages()
    result = build(stages, Validator(OK))

    assert result.validated and result.stop_reason == "validated"
    assert result.attempts == 1
    assert stages.calls == [("booking", True), ("activities", None), ("routes", None)]
    assert sorted(result.maps) == [1, 2, 3]
    assert result.stage_runs == {"booking": 1, "activities": 1, "routes": 1}


# ── Targeted retries ────────────────────────────────────────────────


def test_route_issue_reroutes_only_that_day():
    stages = Stages()
    validator = Validator(report(issue("map.route.invalid", "maps[2].features[1]")), OK)

    result = build(stages, validator)

    assert result.validated and result.attempts == 2
    assert stages.calls[3:] == [("routes", frozenset({2}))]
    # Maps of the days that passed are kept.
    assert sorted(validator.seen_maps[1]) == [1, 2, 3]


def test_activity_issue_replans_and_reroutes_only_that_day():
    stages = Stages()
    validator = Validator(
        report(issue("activity.city.mismatch", "itinerary[0].activities[1].location.requested_city")),
        OK,
    )

    result = build(stages, validator)

    assert result.validated
    assert stages.calls[3:] == [("activities", frozenset({1})), ("routes", frozenset({1}))]


def test_activity_and_route_issues_on_different_days_are_combined():
    stages = Stages()
    validator = Validator(
        report(
            issue("day.activities.unverified", "itinerary[2].activities"),
            issue("map.points.mismatch", "maps[1]"),
        ),
        OK,
    )

    build(stages, validator)

    assert stages.calls[3:] == [("activities", frozenset({3})), ("routes", frozenset({1, 3}))]


def test_city_coverage_issue_replans_every_day():
    stages = Stages()
    validator = Validator(report(issue("itinerary.city.coverage.missing", "itinerary.city_coverage[0]")), OK)

    build(stages, validator)

    assert stages.calls[3:] == [("activities", None), ("routes", None)]


def test_empty_days_skip_routing_and_are_replanned():
    stages = Stages(activity_results=[(2,), None])
    validator = Validator(OK)

    result = build(stages, validator)

    assert result.validated and result.attempts == 2
    assert stages.calls == [
        ("booking", True),
        ("activities", None),
        ("activities", frozenset({2})),
        ("routes", None),
    ]
    assert [i.code for i in result.issues] == ["day.activities.empty"]


# ── Booking ─────────────────────────────────────────────────────────


def test_booking_issue_from_reused_evidence_retries_with_a_fresh_search():
    stages = Stages(skeletons=[SkeletonDraft(itinerary=_days(), reused_evidence=True)])
    validator = Validator(report(issue("hotel.country.mismatch", "itinerary[0].hotel")), OK)

    result = build(stages, validator)

    assert result.validated
    assert [c for c in stages.calls if c[0] == "booking"] == [("booking", True), ("booking", False)]


def test_booking_issue_from_a_search_stops_at_once():
    stages = Stages()
    validator = Validator(report(issue("flight.date.mismatch", "itinerary[0].flight[0]")))

    result = build(stages, validator)

    assert not result.validated
    assert result.stop_reason == "booking_unrecoverable"
    assert result.attempts == 1


def test_missing_booking_is_researched_once_then_stops():
    gap = SkeletonDraft(itinerary=_days(), missing=("hotel",))
    stages = Stages(skeletons=[gap, gap, gap])

    result = build(stages, Validator())

    assert result.stop_reason == "booking_unrecoverable"
    assert result.attempts == 2
    assert stages.calls == [("booking", True), ("booking", False)]
    assert result.report.issues[0].code == "hotel.night.missing"


def test_missing_booking_found_on_the_fresh_search():
    gap = SkeletonDraft(itinerary=_days(), missing=("outbound_flight",))
    stages = Stages(skeletons=[gap])

    result = build(stages, Validator(OK))

    assert result.validated and result.attempts == 2


# ── Stopping ────────────────────────────────────────────────────────


def test_identical_issues_twice_stop_as_no_progress():
    stages = Stages()
    same = report(issue("map.route.invalid", "maps[1]"))

    result = build(stages, Validator(same, same, OK))

    assert result.stop_reason == "no_progress"
    assert result.attempts == 2


def test_attempts_are_capped():
    stages = Stages()
    validator = Validator(
        report(issue("map.route.invalid", "maps[1]")),
        report(issue("map.route.invalid", "maps[2]")),
        report(issue("map.route.invalid", "maps[3]")),
    )

    result = build(stages, validator, max_attempts=2)

    assert result.stop_reason == "attempts_exhausted"
    assert result.attempts == 2
    assert [i.path for i in result.issues] == ["maps[1]", "maps[2]"]


def test_input_error_stops_immediately():
    stages = Stages(skeletons=[PlanningInputError("dest_currency_code is required")])

    result = build(stages, Validator())

    assert result.stop_reason == "invalid_input"
    assert result.attempts == 1
    assert result.report.issues[0].code == "planning.input.invalid"


def test_stage_exception_retries_from_that_stage_only():
    stages = Stages(route_errors=[RuntimeError("boom"), None])

    result = build(stages, Validator(OK))

    assert result.validated and result.attempts == 2
    assert stages.calls == [
        ("booking", True),
        ("activities", None),
        ("routes", None),
        ("routes", None),
    ]
    assert [i.code for i in result.issues] == ["planning.routes.exception"]


def test_repeated_stage_exception_stops():
    stages = Stages(activity_results=[RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])

    result = build(stages, Validator())

    assert result.stop_reason == "no_progress"
    assert result.attempts == 2
    assert stages.calls.count(("booking", True)) == 1


def test_invalid_attempt_limits_are_rejected():
    with pytest.raises(ValueError):
        ItineraryPlanningService(max_attempts=0)
    with pytest.raises(ValueError):
        build(Stages(), Validator(), max_attempts=0)


# ── Issue classification ────────────────────────────────────────────


def test_classify_issues_maps_paths_to_days():
    itinerary = [{"day": 1}, {"day": 2}, {"day": 3}]
    found = classify_issues(
        [
            issue("activity.duplicate", "itinerary[1].activities[0]"),
            issue("map.day.missing", "maps[3]"),
            issue("itinerary.route.invalid", "maps[1].route"),
            issue("day.total_cost.mismatch", "itinerary[0].day_total_cost"),
        ],
        itinerary,
    )

    assert found == {
        Stage.ACTIVITIES: frozenset({2}),
        Stage.ROUTES: frozenset({1, 3}),
        Stage.BOOKING: None,
    }


def test_issue_without_a_day_widens_to_every_day():
    found = classify_issues(
        [issue("map.route.invalid", "maps[1]"), issue("maps.invalid", "maps")],
        [{"day": 1}],
    )
    assert found == {Stage.ROUTES: None}


# ── Logging ─────────────────────────────────────────────────────────


def test_log_event_is_bounded_and_readable(caplog):
    target = logging.getLogger("test.planning")

    with caplog.at_level(logging.INFO, logger="test.planning"):
        log_event(
            target,
            "planning.stage",
            context=LogContext(request_id="r 1\nINJECTED", session_id=None, destination_country_code="jp"),
            stage="routes",
            outcome="failed",
            attempt=500,
            issue_codes=["a", "a", *[f"c{n}" for n in range(20)]],
            elapsed_ms=12.7,
        )

    record = caplog.records[-1]
    message = record.getMessage()
    assert record.levelno == logging.WARNING
    assert "\n" not in message
    assert "request_id=r_1_INJECTED" in message
    assert "session_id=unknown country=JP attempt=99" in message
    assert "elapsed_ms=12" in message
    assert message.split("issues=")[1].count(",") == 7
    assert record.event_fields["stage"] == "routes"


def test_broken_handler_never_raises():
    class Broken(logging.Handler):
        def emit(self, record):
            raise RuntimeError("handler down")

        def handleError(self, record):
            raise RuntimeError("still down")

    target = logging.getLogger("test.planning.broken")
    target.addHandler(Broken())
    target.propagate = False
    try:
        log_event(target, "planning.stage", outcome="completed")
    finally:
        target.handlers.clear()
        target.propagate = True
