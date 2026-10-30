"""Itinerary-domain models.

This module composes booking and routing data contracts only. It does not import
booking/routing services or infrastructure implementations, so service modules
remain independently replaceable.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import math
from typing import Any, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.modules.booking.models import (
    FlightResult,
    HotelResult,
)
from app.modules.budget.models import BudgetCategory
from app.modules.routing.models import RouteInfo
from app.shared.types import (
    CountryCode,
    NonEmptyText,
    ensure_finite_number,
)


MAX_ACTIVITY_CITY_DISTANCE_KM = 75.0

_BUDGET_TOTAL_TOLERANCE = 0.05

_CANONICAL_BUDGET_CATEGORIES = frozenset(
    get_args(BudgetCategory)
)


class FrozenDict(dict[str, float]):
    """JSON-serializable budget mapping that rejects public mutation."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrozenDict is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self) -> "FrozenDict":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "FrozenDict":
        return self


class ExpectedDay(BaseModel):
    """One exact calendar day required by the planning contract."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    day: int = Field(ge=1)
    date: str

    @field_validator("date")
    @classmethod
    def _date_is_iso_calendar_day(
        cls,
        value: str,
    ) -> str:
        try:
            datetime.strptime(
                value,
                "%Y-%m-%d",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "date must use YYYY-MM-DD"
            ) from exc

        return value


class TripRequirements(BaseModel):
    """Immutable itinerary-domain contract, independent of orchestration state."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    origin_country: NonEmptyText
    destination_country: NonEmptyText
    destination_country_code: CountryCode

    cities: tuple[NonEmptyText, ...] = Field(
        min_length=1,
    )

    expected_days: tuple[ExpectedDay, ...] = Field(
        min_length=1,
    )

    num_people: int = Field(
        ge=1,
        strict=True,
    )

    base_currency: str = Field(
        pattern=r"^[A-Z]{3}$",
    )

    destination_currency: str = Field(
        pattern=r"^[A-Z]{3}$",
    )

    total_budget: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    budget_allocation: dict[str, float]

    @field_validator(
        "budget_allocation",
        mode="before",
    )
    @classmethod
    def _allocation_values_are_numeric(
        cls,
        value: Any,
    ) -> Any:
        if not isinstance(value, Mapping):
            raise ValueError(
                "budget allocation must be a mapping"
            )

        for category, amount in value.items():
            if not isinstance(category, str):
                raise ValueError(
                    "budget allocation categories must be strings"
                )

            if (
                isinstance(amount, bool)
                or not isinstance(amount, (int, float))
            ):
                raise ValueError(
                    f"budget allocation for {category} "
                    "must be a finite number"
                )

            if not math.isfinite(float(amount)):
                raise ValueError(
                    f"budget allocation for {category} "
                    "must be a finite number"
                )

        return value

    @model_validator(mode="after")
    def _allocation_is_canonical_and_balanced(
        self,
    ) -> "TripRequirements":
        actual_categories = set(
            self.budget_allocation
        )

        if actual_categories != _CANONICAL_BUDGET_CATEGORIES:
            missing = sorted(
                _CANONICAL_BUDGET_CATEGORIES
                - actual_categories
            )

            unexpected = sorted(
                actual_categories
                - _CANONICAL_BUDGET_CATEGORIES
            )

            details: list[str] = []

            if missing:
                details.append(
                    f"missing={missing}"
                )

            if unexpected:
                details.append(
                    f"unexpected={unexpected}"
                )

            raise ValueError(
                "budget allocation must contain exactly "
                "the canonical categories"
                + (
                    f" ({', '.join(details)})"
                    if details
                    else ""
                )
            )

        validated: dict[str, float] = {}

        for category, amount in (
            self.budget_allocation.items()
        ):
            number = float(amount)

            if number < 0:
                raise ValueError(
                    "budget allocation values "
                    "must be non-negative"
                )

            validated[category] = number

        if not math.isclose(
            sum(validated.values()),
            float(self.total_budget),
            rel_tol=0.0,
            abs_tol=_BUDGET_TOTAL_TOLERANCE,
        ):
            raise ValueError(
                "budget allocation must equal "
                "the total converted budget"
            )

        object.__setattr__(
            self,
            "budget_allocation",
            FrozenDict(validated),
        )

        return self


class ValidationIssue(BaseModel):
    """One deterministic itinerary qualification failure."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    code: str
    path: str
    message: str


class ValidationReport(BaseModel):
    """Immutable result of deterministic itinerary validation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def qualified(self) -> bool:
        return not self.issues


class ActivityLocation(BaseModel):
    """A provider-verified activity location safe for public rendering."""

    model_config = ConfigDict(
        extra="forbid",
    )

    place_name: NonEmptyText
    country_code: CountryCode
    requested_city: NonEmptyText
    verified_locality: NonEmptyText

    latitude: float = Field(
        ge=-90,
        le=90,
        allow_inf_nan=False,
    )

    longitude: float = Field(
        ge=-180,
        le=180,
        allow_inf_nan=False,
    )

    distance_from_requested_city_km: float | None = Field(
        default=None,
        ge=0,
        le=MAX_ACTIVITY_CITY_DISTANCE_KM,
        allow_inf_nan=False,
    )

    @field_validator(
        "latitude",
        "longitude",
        mode="before",
    )
    @classmethod
    def _coordinates_are_finite(
        cls,
        value: Any,
        info: Any,
    ) -> Any:
        return ensure_finite_number(
            value,
            field_name=info.field_name,
        )

    @model_validator(mode="after")
    def _locality_proves_requested_city(
        self,
    ) -> "ActivityLocation":
        requested = " ".join(
            self.requested_city.split()
        ).casefold()

        verified = " ".join(
            self.verified_locality.split()
        ).casefold()

        if verified == requested:
            return self

        if (
            self.distance_from_requested_city_km
            is not None
        ):
            return self

        raise ValueError(
            "verified locality must match the requested city "
            "or carry valid nearby-city distance evidence"
        )


class ActivityResult(BaseModel):
    """A destination-verified activity in a public itinerary."""

    model_config = ConfigDict(
        extra="forbid",
    )

    name: NonEmptyText

    type: Literal[
        "attraction",
        "restaurant",
    ]

    address: NonEmptyText

    estimated_cost: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    order: int = Field(
        ge=1,
        strict=True,
    )

    location: ActivityLocation

    description: str | None = None
    category: str | None = None

    rating: float | None = Field(
        default=None,
        ge=0,
        le=5,
        allow_inf_nan=False,
    )

    thumbnail: str | None = None
    suggested_time: str | None = None
    is_estimated: bool | None = None

    @field_validator(
        "estimated_cost",
        "rating",
        mode="before",
    )
    @classmethod
    def _numbers_are_finite(
        cls,
        value: Any,
        info: Any,
    ) -> Any:
        if (
            value is None
            and info.field_name == "rating"
        ):
            return value

        return ensure_finite_number(
            value,
            field_name=info.field_name,
        )


class DailyItinerary(BaseModel):
    """One day's composed trip plan."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    day: int = Field(
        ge=1,
    )

    date: str | None = None

    flight: list[FlightResult] | None = None

    hotel: HotelResult | None = None

    activities: list[ActivityResult] = Field(
        default_factory=list,
    )

    route: RouteInfo | None = None

    day_total_cost: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
    )


__all__ = [
    "MAX_ACTIVITY_CITY_DISTANCE_KM",
    "ExpectedDay",
    "TripRequirements",
    "ValidationIssue",
    "ValidationReport",
    "ActivityLocation",
    "ActivityResult",
    "DailyItinerary",
]