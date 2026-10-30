"""Booking-domain models shared by booking services and itinerary composition."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.shared.types import (
    CountryCode,
    NonEmptyText,
    ensure_finite_number,
)


class HotelLocation(BaseModel):
    """Closed public hotel coordinate shape emitted by booking providers."""

    model_config = ConfigDict(
        extra="forbid",
    )

    lat: float | None = Field(
        default=None,
        ge=-90,
        le=90,
        allow_inf_nan=False,
    )

    lng: float | None = Field(
        default=None,
        ge=-180,
        le=180,
        allow_inf_nan=False,
    )

    country_code: CountryCode | None = None

    @field_validator(
        "lat",
        "lng",
        mode="before",
    )
    @classmethod
    def _coordinates_are_finite(
        cls,
        value: Any,
        info: Any,
    ) -> Any:
        if value is None:
            return value

        return ensure_finite_number(
            value,
            field_name=info.field_name,
        )


class FlightAirport(BaseModel):
    """Closed airport subset consumed by the application."""

    model_config = ConfigDict(
        extra="forbid",
    )

    id: str = ""
    name: str = ""
    time: str = ""


class HotelResult(BaseModel):
    """Normalized hotel search result."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    hotel_name: str = ""

    hotel_class: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        allow_inf_nan=False,
    )

    overall_rating: float | None = Field(
        default=None,
        ge=0.0,
        le=5.0,
        allow_inf_nan=False,
    )

    reviews: int | None = Field(
        default=None,
        ge=0,
    )

    description: str = ""

    price_per_night: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
    )

    amenities: list[str] = Field(
        default_factory=list,
    )

    check_in_time: str = ""
    check_out_time: str = ""

    location: HotelLocation | None = None

    image: str = ""
    booking_url: str = ""

    over_budget: bool = False


class FlightResult(BaseModel):
    """Normalized flight search result."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    airline: str = ""
    flight_number: str = ""
    airline_logo: str = ""
    travel_class: str = ""
    airplane: str = ""

    departure_airport: FlightAirport | None = None
    arrival_airport: FlightAirport | None = None

    departure_time: str = ""
    arrival_time: str = ""

    duration: int = Field(
        default=0,
        ge=0,
    )

    stops: int = Field(
        default=0,
        ge=0,
    )

    layovers: list[str] = Field(
        default_factory=list,
    )

    price: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
    )

    booking_url: str | None = None

    over_budget: bool = False


class BookingTrip(BaseModel):
    """The trip facts booking needs, validated once at the boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    origin_country: NonEmptyText
    destination_country: NonEmptyText
    destination_country_code: CountryCode
    city: NonEmptyText
    start_date: str
    end_date: str
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    adults: int = Field(ge=1)
    transportation_budget: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    accommodation_budget: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    @field_validator("start_date", "end_date")
    @classmethod
    def _iso_date(cls, value: str) -> str:
        try:
            parsed = date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("dates must use YYYY-MM-DD") from exc

        if parsed.isoformat() != value:
            raise ValueError("dates must use YYYY-MM-DD")

        return value

    @model_validator(mode="after")
    def _end_not_before_start(self) -> "BookingTrip":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not precede start_date")

        return self

    @property
    def nights(self) -> int:
        return (
            date.fromisoformat(self.end_date) - date.fromisoformat(self.start_date)
        ).days

    @property
    def hotel_destination(self) -> str:
        return f"{self.city}, {self.destination_country}"


SkeletonSource = Literal["assessment", "search"]


class SkeletonResult(BaseModel):
    """A day skeleton plus which bookings could not be found."""

    model_config = ConfigDict(frozen=True)

    itinerary: list[dict[str, Any]]
    source: SkeletonSource
    missing: tuple[Literal["outbound_flight", "return_flight", "hotel"], ...] = ()


__all__ = [
    "BookingTrip",
    "SkeletonResult",
    "SkeletonSource",
    "HotelLocation",
    "FlightAirport",
    "HotelResult",
    "FlightResult",
]