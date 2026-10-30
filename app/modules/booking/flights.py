"""Pure flight-domain helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.modules.booking.models import (
    FlightAirport,
    FlightResult,
)
from app.modules.booking.validation import parse_budget_cap


def normalise_iata_code(
    value: Any,
) -> str | None:
    """Return one canonical three-letter IATA code or None."""

    if not isinstance(value, str):
        return None

    code = value.strip().upper()

    if (
        len(code) != 3
        or not code.isascii()
        or not code.isalpha()
    ):
        return None

    return code


def airport_iata_identity(
    airport: FlightAirport | None,
) -> str | None:
    """Return the canonical IATA identity of a normalized airport."""

    if airport is None:
        return None

    return normalise_iata_code(
        airport.id
    )


def flight_matches_query_endpoints(
    flight: FlightResult,
    *,
    origin_iata: str,
    destination_iata: str,
) -> bool:
    """Require a normalized flight to exactly match the requested route."""

    expected_origin = normalise_iata_code(
        origin_iata
    )

    expected_destination = normalise_iata_code(
        destination_iata
    )

    if (
        expected_origin is None
        or expected_destination is None
    ):
        return False

    return (
        airport_iata_identity(
            flight.departure_airport
        )
        == expected_origin
        and airport_iata_identity(
            flight.arrival_airport
        )
        == expected_destination
    )


def filter_flights_by_budget(
    flights: Sequence[FlightResult],
    *,
    max_budget: float | None,
) -> list[FlightResult]:
    """Return flights within an optional budget cap (None or 0 = no cap)."""

    cap = parse_budget_cap(
        max_budget,
        field_name="max_budget",
    )

    if cap is None:
        return list(flights)

    return [
        flight
        for flight in flights
        if flight.price <= cap
    ]


__all__ = [
    "normalise_iata_code",
    "airport_iata_identity",
    "flight_matches_query_endpoints",
    "filter_flights_by_budget",
]