"""Contracts required by the booking domain."""

from __future__ import annotations

from typing import Protocol

from app.modules.booking.models import FlightResult, HotelResult

class HotelLocationVerifierPort(Protocol):
    """Geographic verification required before accepting hotel results."""

    def resolve_country_code(
        self,
        latitude: float,
        longitude: float,
    ) -> str | None:
        ...

    def resolve_locality_names(
        self,
        latitude: float,
        longitude: float,
        destination_country_code: str,
    ) -> tuple[str, ...] | None:
        ...


class IataResolverPort(Protocol):
    """Resolve a country to its primary international-gateway IATA code."""

    def resolve_iata_code(
        self,
        country_name: str,
    ) -> str | None:
        ...


class FlightSearchPort(Protocol):
    """Provider-neutral flight-search capability."""

    def search_flights(
        self,
        *,
        origin: str,
        destination: str,
        departure_date: str,
        currency: str,
        max_budget: float | None = None,
        adults: int = 1,
    ) -> list[FlightResult]:
        ...


class HotelSearchPort(Protocol):
    """Provider-neutral, geographically verified hotel search."""

    def search_hotels(
        self,
        *,
        destination: str,
        check_in: str,
        check_out: str,
        currency: str,
        max_price_per_night: float | None = None,
        adults: int = 1,
        destination_country_code: str | None = None,
    ) -> list[HotelResult]:
        ...


__all__ = [
    "HotelLocationVerifierPort",
    "IataResolverPort",
    "FlightSearchPort",
    "HotelSearchPort",
]