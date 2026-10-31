"""HTTP transport schemas for initial trip submission."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.api.schemas.budget import (
    BudgetCheckUnavailableResponse,
    BudgetConfirmationResponse,
)
from app.api.schemas.common import (
    PlanningUnavailableResponse,
)
from app.modules.itinerary.models import (
    DailyItinerary,
)
from app.modules.routing.models import (
    GeoJSONFeatureCollection,
    canonicalize_public_daily_maps,
)
from app.shared.types import (
    CountryCode,
    NonEmptyText,
)


class InitialFormRequest(BaseModel):
    """Initial trip-planning form submitted by the user."""

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    country: str = Field(
        ...,
        min_length=1,
        description="Destination country.",
    )

    city: list[str] = Field(
        default_factory=list,
        description=(
            "Optional destination states/cities. "
            "When omitted, the server resolves "
            "and verifies a planning city before "
            "budget assessment."
        ),
    )

    num_people: int = Field(
        ...,
        ge=1,
        description=(
            "Number of travellers "
            "(must be at least 1)."
        ),
    )

    total_budget: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Total trip budget in the "
            "user's base currency."
        ),
    )

    request_budget_recommendation: bool = Field(
        default=False,
        description=(
            "Explicit opt-in used only when "
            "the user does not know a budget."
        ),
    )

    budget_assessment_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description=(
            "Cached assessment identifier echoed "
            "only after the user accepts the "
            "recommended minimum."
        ),
    )

    start_date: str = Field(
        ...,
        description=(
            "Trip start date in YYYY-MM-DD format."
        ),
    )

    end_date: str = Field(
        ...,
        description=(
            "Trip end date in YYYY-MM-DD format."
        ),
    )

    @field_validator("city")
    @classmethod
    def validate_city(
        cls,
        value: list[str],
    ) -> list[str]:
        """Allow omission, but reject supplied blank localities."""

        cleaned = [
            item.strip()
            for item in value
        ]

        if any(
            not item
            for item in cleaned
        ):
            raise ValueError(
                "Every provided destination "
                "state/city must be non-empty."
            )

        return cleaned

    @model_validator(mode="after")
    def validate_request(
        self,
    ) -> "InitialFormRequest":
        """Validate budget mode and ordered ISO calendar dates."""

        if self.request_budget_recommendation:
            if (
                self.total_budget is not None
                or self.budget_assessment_id is not None
            ):
                raise ValueError(
                    "Budget recommendation mode "
                    "cannot include an amount or "
                    "assessment confirmation."
                )

        elif self.total_budget is None:
            raise ValueError(
                "Provide total_budget or explicitly "
                "request a budget recommendation."
            )

        start = _iso_date(self.start_date, "start_date")
        end = _iso_date(self.end_date, "end_date")

        if end < start:
            raise ValueError(
                "end_date must be on or after start_date"
            )

        return self


def _iso_date(value: str, field_name: str) -> date:
    """Exactly ``YYYY-MM-DD``. ``date.fromisoformat`` alone also accepts
    other ISO forms such as "20260105", which the budget and planning models
    reject; the round trip below refuses them."""

    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        parsed = None

    if parsed is None or parsed.isoformat() != value:
        raise ValueError(f"{field_name} must be a calendar date in YYYY-MM-DD format")

    return parsed


class FinalResponse(BaseModel):
    """Successful initial-planning API response."""

    model_config = ConfigDict(
        extra="forbid",
    )

    status: Literal["success"]

    chat_reply: NonEmptyText

    itinerary: list[DailyItinerary]

    daily_geojson_maps: dict[
        int,
        GeoJSONFeatureCollection,
    ]

    destination_country_code: CountryCode

    resolved_cities: list[
        NonEmptyText
    ] = Field(
        min_length=1,
    )

    total_budget: float = Field(
        default=0.0,
        ge=0,
        allow_inf_nan=False,
    )

    currency: str = ""

    budget_allocation: dict[
        str,
        float,
    ] = Field(
        default_factory=dict,
    )

    session_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Identifier of the underlying "
            "LangGraph thread. The frontend must "
            "echo this value as session_id in "
            "subsequent chat requests."
        ),
    )

    @field_validator(
        "daily_geojson_maps",
        mode="before",
    )
    @classmethod
    def _canonical_map_keys(
        cls,
        value,
    ):
        return canonicalize_public_daily_maps(
            value
        )


TripSubmissionResponse = Annotated[
    FinalResponse
    | BudgetConfirmationResponse
    | BudgetCheckUnavailableResponse
    | PlanningUnavailableResponse,
    Field(discriminator="status"),
]


__all__ = [
    "InitialFormRequest",
    "FinalResponse",
    "TripSubmissionResponse",
]