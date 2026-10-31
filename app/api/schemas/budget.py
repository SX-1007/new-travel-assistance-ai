"""HTTP transport schemas for budget-gate responses."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.shared.types import NonEmptyText


class BudgetEvidenceResponse(BaseModel):
    """Provider price components used in the deterministic minimum."""

    model_config = ConfigDict(
        extra="forbid",
    )

    outbound_flight_price: float = Field(
        gt=0,
        allow_inf_nan=False,
    )

    return_flight_price: float = Field(
        gt=0,
        allow_inf_nan=False,
    )

    hotel_price_per_night: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    hotel_nights: int = Field(
        ge=0,
    )

    @model_validator(mode="after")
    def _hotel_price_matches_nights(
        self,
    ) -> "BudgetEvidenceResponse":
        if (
            (self.hotel_nights == 0)
            != (self.hotel_price_per_night == 0)
        ):
            raise ValueError(
                "hotel nights and price must use "
                "the same zero-night semantics"
            )

        return self


class BudgetConfirmationResponse(BaseModel):
    """Grounded minimum that requires an explicit user decision."""

    model_config = ConfigDict(
        extra="forbid",
    )

    status: Literal[
        "budget_confirmation_required"
    ]

    reason: Literal[
        "insufficient_budget",
        "recommendation_requested",
    ]

    chat_reply: str

    budget_assessment_id: str

    resolved_cities: list[NonEmptyText] = Field(
        min_length=1,
    )

    stated_budget: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )

    recommended_minimum_budget: float = Field(
        gt=0,
        allow_inf_nan=False,
    )

    base_currency: str
    destination_currency: str
    expires_at: str

    evidence: BudgetEvidenceResponse

    # Always null: the frontend expects these keys on every form response.
    itinerary: None = None
    daily_geojson_maps: None = None


class BudgetCheckUnavailableResponse(BaseModel):
    """Fail-closed result when no reusable grounded assessment exists."""

    model_config = ConfigDict(
        extra="forbid",
    )

    status: Literal[
        "budget_check_unavailable"
    ]

    reason: Literal[
        "provider_data_unavailable",
        "assessment_cache_unavailable",
        "assessment_expired_or_invalid",
        "destination_resolution_unavailable",
    ]

    chat_reply: str

    # Always null: the frontend expects these keys on every form response.
    itinerary: None = None
    daily_geojson_maps: None = None


__all__ = [
    "BudgetEvidenceResponse",
    "BudgetConfirmationResponse",
    "BudgetCheckUnavailableResponse",
]
