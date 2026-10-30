"""Stage contracts for the itinerary planning transaction.

The transaction only knows these three stages. The orchestration layer binds
them to the booking, activity and routing services for one trip, so this
module never imports those services or ``AgentState``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


BookingGap = Literal["outbound_flight", "return_flight", "hotel"]


class PlanningInputError(ValueError):
    """The trip itself can't be planned (missing currency, bad dates...).

    Retrying can't fix this, so the transaction stops at once.
    """


@dataclass(frozen=True, slots=True)
class SkeletonDraft:
    """Day skeleton with flights and hotel, before activities."""

    itinerary: list[dict[str, Any]]
    # True when the bookings came from the budget check instead of a search.
    reused_evidence: bool = False
    # Bookings no search could provide.
    missing: tuple[BookingGap, ...] = ()


@dataclass(frozen=True, slots=True)
class ActivityDraft:
    """Itinerary with activities, plus the days that are still empty."""

    itinerary: list[dict[str, Any]]
    missing_days: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class RouteDraft:
    """Itinerary with route summaries, plus maps for the routed days only."""

    itinerary: list[dict[str, Any]]
    maps: dict[int, dict[str, Any]] = field(default_factory=dict)


class PlanningStages(Protocol):
    """The three provider-backed planning stages for one trip.

    ``days=None`` means every day. A stage may raise ``PlanningInputError``
    for an unplannable trip; any other exception is treated as a failed run
    of that stage.
    """

    def skeleton(self, *, reuse_evidence: bool) -> SkeletonDraft:
        ...

    def activities(
        self,
        itinerary: Sequence[dict[str, Any]],
        *,
        days: frozenset[int] | None,
    ) -> ActivityDraft:
        ...

    def routes(
        self,
        itinerary: Sequence[dict[str, Any]],
        *,
        days: frozenset[int] | None,
    ) -> RouteDraft:
        ...


__all__ = [
    "ActivityDraft",
    "BookingGap",
    "PlanningInputError",
    "PlanningStages",
    "RouteDraft",
    "SkeletonDraft",
]
