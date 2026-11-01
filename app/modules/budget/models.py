"""Budget-domain models. HTTP response shapes live in app.api.schemas.budget."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from app.shared.types import NonEmptyText


BudgetCategory = Literal[
    "transportation",
    "accommodation",
    "food",
    "activity",
    "shopping",
    "emergency_fund",
]


BudgetAdjustmentModifier = Literal[
    "small_increase",
    "small_decrease",
    "large_increase",
    "large_decrease",
]


class BudgetAdjustment(BaseModel):
    """
    One explicit budget-category change
    requested by the user/LLM.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    category: BudgetCategory = Field(
        description=(
            "Structural category to change. "
            "Map hotel/stay to accommodation, "
            "meal/restaurant to food, "
            "train/flight/car to transportation, "
            "and tickets/tours to activity."
        )
    )

    add_delete_amount: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Absolute signed change in "
            "destination currency. Leave null "
            "when modifier is used."
        ),
    )

    modifier: (
        BudgetAdjustmentModifier | None
    ) = Field(
        default=None,
        description=(
            "Vague-language adjustment. "
            "Use only when no explicit "
            "amount was supplied."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_change_mode(
        self,
    ) -> "BudgetAdjustment":
        has_amount = (
            self.add_delete_amount
            is not None
        )

        has_modifier = (
            self.modifier
            is not None
        )

        if has_amount == has_modifier:
            raise ValueError(
                "exactly one of "
                "add_delete_amount or modifier "
                "must be provided"
            )

        return self


class BudgetAllocationResult(BaseModel):
    """
    Pure result of a budget-allocation
    transformation.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    budget_allocation: dict[
        BudgetCategory,
        float,
    ]

    warnings: tuple[str, ...] = ()


class BudgetTrip(BaseModel):
    """
    The exact trip a budget assessment belongs to.

    Validated once on construction, so every assessment rule can trust it.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    origin: NonEmptyText
    destination: NonEmptyText
    destination_city: NonEmptyText
    start_date: str
    end_date: str
    num_people: StrictInt = Field(ge=1)

    @field_validator("start_date", "end_date")
    @classmethod
    def _iso_calendar_date(cls, value: str) -> str:
        try:
            parsed = date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("dates must use YYYY-MM-DD") from exc

        if parsed.isoformat() != value:
            raise ValueError("dates must use YYYY-MM-DD")

        return value

    @model_validator(mode="after")
    def _end_not_before_start(self) -> "BudgetTrip":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not precede start_date")

        return self


class BudgetEvidence(BaseModel):
    """Cheapest real provider objects and their calculation inputs."""

    outbound_flight: dict[str, Any]
    return_flight: dict[str, Any]
    hotel: dict[str, Any]
    outbound_flight_price: float = Field(gt=0, allow_inf_nan=False)
    return_flight_price: float = Field(gt=0, allow_inf_nan=False)
    hotel_price_per_night: float = Field(ge=0, allow_inf_nan=False)
    hotel_nights: int = Field(ge=0)

    @model_validator(mode="after")
    def _hotel_evidence_matches_overnight_requirement(self) -> "BudgetEvidence":
        if self.hotel_nights == 0:
            if self.hotel_price_per_night != 0 or self.hotel:
                raise ValueError("zero-night trips require canonical empty hotel evidence")
        elif self.hotel_price_per_night <= 0:
            raise ValueError("overnight trips require a positive hotel price")
        return self


class BudgetAssessment(BaseModel):
    """A reusable, provider-grounded minimum budget for one exact trip.

    The stored shape must not change: saved assessments are read back with
    it in a later request.
    """

    assessment_id: str = Field(min_length=1)
    calculation_version: str = Field(min_length=1)
    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    destination_city: str = Field(min_length=1)
    start_date: str = Field(min_length=1)
    end_date: str = Field(min_length=1)
    num_people: int = Field(ge=1)
    base_currency: str = Field(min_length=3, max_length=3)
    destination_currency: str = Field(min_length=3, max_length=3)
    exchange_rate: float = Field(gt=0, allow_inf_nan=False)
    minimum_destination_budget: float = Field(gt=0, allow_inf_nan=False)
    recommended_minimum_budget: float = Field(gt=0, allow_inf_nan=False)
    evidence: BudgetEvidence
    created_at: str
    expires_at: str


class BudgetAssessmentResult(BaseModel):
    """Assessment plus whether it is safe to confirm in a later request."""

    assessment: BudgetAssessment
    persisted: bool


__all__ = [
    "BudgetCategory",
    "BudgetAdjustmentModifier",
    "BudgetAdjustment",
    "BudgetAllocationResult",
    "BudgetTrip",
    "BudgetEvidence",
    "BudgetAssessment",
    "BudgetAssessmentResult",
]
