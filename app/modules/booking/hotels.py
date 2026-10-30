"""Pure hotel rules and provider-independent geographic verification.

No network, cache, settings, orchestration or provider implementation imports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from app.modules.booking.models import HotelLocation, HotelResult
from app.modules.booking.ports import HotelLocationVerifierPort
from app.modules.booking.validation import parse_budget_cap


def calculate_total_nights(start_date: str, end_date: str) -> int:
    """Return nights between ISO dates; same-day = 0, reversed/invalid = error."""
    try:
        if not isinstance(start_date, str) or not isinstance(end_date, str):
            raise ValueError("dates must be strings")
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        if start.isoformat() != start_date or end.isoformat() != end_date:
            raise ValueError("dates must use YYYY-MM-DD")
    except ValueError as exc:
        raise ValueError("start_date and end_date must use valid YYYY-MM-DD dates") from exc

    nights = (end - start).days
    if nights < 0:
        raise ValueError("end_date must not precede start_date")
    return nights


def normalise_hotel_coordinates(location: Any) -> tuple[float, float] | None:
    """Accept SerpAPI or public coordinate spellings, only if geographically valid."""
    if not isinstance(location, Mapping):
        return None
    raw_lat = location.get("latitude")
    raw_lng = location.get("longitude")
    if raw_lat is None:
        raw_lat = location.get("lat")
    if raw_lng is None:
        raw_lng = location.get("lng")
    if any(isinstance(v, bool) or v is None for v in (raw_lat, raw_lng)):
        return None
    try:
        lat, lng = float(raw_lat), float(raw_lng)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lng)):
        return None
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return None
    return lat, lng


def normalise_hotel_city(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return " ".join(value.split()) or None


def requested_hotel_city(destination: Any) -> str | None:
    """Take the city from the `city, country` hotel destination."""
    normalised = normalise_hotel_city(destination)
    if normalised is None:
        return None
    return normalise_hotel_city(normalised.partition(",")[0])


def verify_hotel_location(
    location: Any,
    *,
    requested_city: str,
    destination_country_code: str,
    verifier: HotelLocationVerifierPort,
) -> HotelLocation | None:
    """Fail closed unless country AND structured locality match the request."""
    coordinates = normalise_hotel_coordinates(location)
    city = normalise_hotel_city(requested_city)
    if (
        coordinates is None
        or city is None
        or not isinstance(destination_country_code, str)
    ):
        return None
    country_code = destination_country_code.strip().upper()
    if not (len(country_code) == 2 and country_code.isascii() and country_code.isalpha()):
        return None

    lat, lng = coordinates
    try:
        observed_country = verifier.resolve_country_code(lat, lng)
        if observed_country != country_code:
            return None
        locality_names = verifier.resolve_locality_names(lat, lng, country_code)
    except Exception:
        return None
    # Only accept explicitly structured locality evidence, not freeform labels.
    if not isinstance(locality_names, tuple):
        return None
    if not any(
        isinstance(name, str)
        and (normalise_hotel_city(name) or "").casefold() == city.casefold()
        for name in locality_names
    ):
        return None

    return HotelLocation(lat=lat, lng=lng, country_code=country_code)


def filter_hotels_by_budget(
    hotels: Sequence[HotelResult], *, max_price_per_night: float | None
) -> list[HotelResult]:
    """Filter results without influencing provider/cache queries.

    None or 0 means "no cap"; negative/NaN are errors.
    """
    limit = parse_budget_cap(max_price_per_night, field_name="max_price_per_night")
    if limit is None:
        return list(hotels)
    return [hotel for hotel in hotels if hotel.price_per_night <= limit]


__all__ = [
    "calculate_total_nights",
    "normalise_hotel_coordinates",
    "normalise_hotel_city",
    "requested_hotel_city",
    "verify_hotel_location",
    "filter_hotels_by_budget",
]