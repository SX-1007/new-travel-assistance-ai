"""Activity-planning domain models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.shared.types import NonEmptyText


PlaceType = Literal["attraction", "restaurant"]
Pacing = Literal["relaxed", "moderate", "packed"]


class PlaceIdea(BaseModel):
    """One AI-proposed place for a specific day. Not yet provider-verified."""

    model_config = ConfigDict(frozen=True)

    name: NonEmptyText
    place_type: PlaceType
    city: str = ""
    day: int = Field(ge=1)
    order: int = Field(ge=1)
    suggested_time: str = ""
    estimated_cost: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ActivityPreferences:
    """The traveller preferences that shape idea generation."""

    interests: tuple[str, ...] = ()
    dietary: tuple[str, ...] = ()
    pacing: Pacing = "moderate"


@dataclass(frozen=True, slots=True)
class IdeaRequest:
    """Everything an idea generator needs; free of provider details."""

    country: str
    cities: tuple[str, ...]
    dates: tuple[str, ...]
    num_people: int
    hotel_name: str
    preferences: ActivityPreferences
    activity_budget: float
    food_budget: float
    # The traveller's pacing cap: no day gets more places than this.
    max_per_day: int
    # Most places per day, aligned with ``dates``; 0 is a travel day (the
    # traveller is not at the destination). Empty: ``max_per_day`` each.
    day_limits: tuple[int, ...] = ()
    # Free hours at the destination per day, aligned with ``dates`` (None:
    # unknown). Empty: not given.
    day_hours: tuple[float | None, ...] = ()


@dataclass(frozen=True, slots=True)
class ActivityPlanRequest:
    """Fill a day skeleton (flights/hotels already chosen) with activities."""

    itinerary: Sequence[Mapping[str, Any]]
    country: str
    cities: tuple[str, ...]
    num_people: int
    activity_budget: float
    food_budget: float
    preferences: ActivityPreferences = field(default_factory=ActivityPreferences)
    # Empty: plan every day. Otherwise re-plan only these days and keep every
    # other day's activities exactly as they are.
    replan_days: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ActivityPlanResult:
    """Updated itinerary plus diagnostics for the caller."""

    itinerary: list[dict[str, Any]]
    # Days that still have no verified activity after every rescue round.
    missing_days: tuple[int, ...]
    rescue_rounds_used: int
    searches_used: int


__all__ = [
    "ActivityPlanRequest",
    "ActivityPlanResult",
    "ActivityPreferences",
    "IdeaRequest",
    "Pacing",
    "PlaceIdea",
    "PlaceType",
]
