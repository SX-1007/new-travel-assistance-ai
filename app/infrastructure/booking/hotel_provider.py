"""SerpAPI Google Hotels adapter; owns provider parsing, cache and search.

Depends only on hotel-domain contracts and injected generic integrations.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Protocol
from urllib.parse import quote_plus, urlsplit

from pydantic import ValidationError

from app.infrastructure.search.serpapi import SerpApiUnavailableError
from app.modules.booking.hotels import (
    calculate_total_nights,
    filter_hotels_by_budget,
    normalise_hotel_coordinates,
    requested_hotel_city,
    verify_hotel_location,
)
from app.modules.booking.models import HotelResult
from app.modules.booking.ports import HotelLocationVerifierPort
from app.shared.countries import country_alpha2
from app.modules.booking.validation import (
    normalise_adults,
    normalise_currency_code,
)


class _SearchClient(Protocol):
    def search(self, params: Mapping[str, Any]) -> dict[str, Any]: ...


class _SearchCache(Protocol):
    def get(self, *, namespace: str, key: Mapping[str, Any]) -> Any | None: ...

    def set(
        self,
        *,
        namespace: str,
        key: Mapping[str, Any],
        payload: Any,
        ttl_hours: int = 1,
    ) -> bool: ...


# New namespace avoids mixing newly validated entries with the legacy format.
_CACHE_NAMESPACE = "hotel_v2"


class SerpApiHotelProvider:
    """Return verified, normalized hotels without LangGraph or AgentState."""

    def __init__(
        self,
        *,
        client: _SearchClient,
        cache: _SearchCache,
        location_verifier: HotelLocationVerifierPort,
    ) -> None:
        self._client = client
        self._cache = cache
        self._location_verifier = location_verifier

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
        city = requested_hotel_city(destination)
        if city is None:
            raise ValueError("destination must contain a city")
        if not isinstance(destination, str):
            raise ValueError("destination must be text")
        destination = " ".join(destination.split())

        nights = calculate_total_nights(check_in, check_out)
        currency = normalise_currency_code(currency)
        adults = normalise_adults(adults)
        # Validate budget even if no network call will be needed.
        filter_hotels_by_budget([], max_price_per_night=max_price_per_night)
        if nights == 0:
            # Same-day trips need no hotel nights; don't make a provider request.
            return []
        country_code = _resolve_country_code(destination, destination_country_code)
        if country_code is None:
            return []

        cache_key = {
            "destination": destination,
            "check_in": check_in,
            "check_out": check_out,
            "currency": currency,
            "adults": adults,
            "country_code": country_code,
        }
        hotels = self._validate_cached_hotels(
            self._cache.get(namespace=_CACHE_NAMESPACE, key=cache_key),
            requested_city=city,
            destination_country_code=country_code,
        )
        if not hotels:
            params = {
                "engine": "google_hotels",
                "q": destination,
                "check_in_date": check_in,
                "check_out_date": check_out,
                "adults": str(adults),
                "currency": currency,
                "hl": "en",
            }
            try:
                payload = self._client.search(params)
            except SerpApiUnavailableError:
                return []
            if not isinstance(payload, dict) or payload.get("error"):
                return []
            properties = payload.get("properties")
            if not isinstance(properties, list):
                return []

            hotels = []
            for property_payload in properties:
                hotel = _parse_hotel_property(property_payload, destination=destination)
                if hotel is None or hotel.location is None:
                    continue
                location = verify_hotel_location(
                    hotel.location.model_dump(),
                    requested_city=city,
                    destination_country_code=country_code,
                    verifier=self._location_verifier,
                )
                if location is None:
                    continue
                hotels.append(hotel.model_copy(update={"location": location}))
            hotels.sort(key=lambda hotel: hotel.price_per_night)
            if hotels:
                self._cache.set(
                    namespace=_CACHE_NAMESPACE,
                    key=cache_key,
                    payload=[hotel.model_dump(mode="json") for hotel in hotels],
                    ttl_hours=1,
                )

        return filter_hotels_by_budget(
            hotels,
            max_price_per_night=max_price_per_night,
        )

    def _validate_cached_hotels(
        self,
        payload: Any,
        *,
        requested_city: str,
        destination_country_code: str,
    ) -> list[HotelResult]:
        if not isinstance(payload, list):
            return []
        hotels: list[HotelResult] = []
        for item in payload:
            try:
                hotel = HotelResult.model_validate(item)
            except (ValidationError, TypeError, ValueError):
                continue
            if (
                hotel.price_per_night <= 0
                or hotel.location is None
                or not hotel.hotel_name.strip()
            ):
                continue
            # A persisted record is untrusted for links as well as coordinates.
            booking_url = _safe_http_url(hotel.booking_url) or (
                "https://www.google.com/travel/hotels?q="
                + quote_plus(f"{hotel.hotel_name} {requested_city}".strip())
            )
            image = _safe_http_url(hotel.image) or ""
            # Never trust cached country/city claims: independently reverify.
            location = verify_hotel_location(
                hotel.location.model_dump(),
                requested_city=requested_city,
                destination_country_code=destination_country_code,
                verifier=self._location_verifier,
            )
            if location is None:
                continue
            hotels.append(hotel.model_copy(update={
                "location": location,
                "booking_url": booking_url,
                "image": image,
            }))
        hotels.sort(key=lambda hotel: hotel.price_per_night)
        return hotels


def _finite_number(value: Any, *, min_value: float = 0.0) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= min_value else None


def _extract_rate(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None

    primary = _finite_number(
        value.get("extracted_lowest"),
        min_value=0.01,
    )
    if primary is not None:
        return primary

    raw = value.get("lowest")

    if isinstance(raw, (int, float)):
        return _finite_number(raw, min_value=0.01)

    if not isinstance(raw, str) or not raw.strip():
        return None

    # Reject negative values and negative accounting formats.
    if any(symbol in raw for symbol in (
        "-", "–", "—", "−", "(", ")"
    )):
        return None

    tokens = re.findall(
        r"\d[\d,]*(?:\.\d+)?",
        raw,
    )

    if len(tokens) != 1:
        return None

    token = tokens[0]

    if not re.fullmatch(
        r"\d+(?:\.\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?",
        token,
    ):
        return None

    # Ambiguous: could mean 1.234 or 1234.
    # Reject instead of silently returning the wrong rate.
    if re.fullmatch(r"\d{1,3}\.\d{3}", token):
        return None

    return _finite_number(
        token.replace(",", ""),
        min_value=0.01,
    )


def _safe_http_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned = value.strip()

    if not cleaned:
        return None

    if any(
        ord(char) < 32 or ord(char) == 127
        for char in cleaned
    ):
        return None

    try:
        parts = urlsplit(cleaned)
        hostname = parts.hostname
    except ValueError:
        return None

    if (
        parts.scheme.lower() not in {"http", "https"}
        or not hostname
        or parts.username is not None
        or parts.password is not None
    ):
        return None

    return cleaned


def _nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if not value.isascii() or not value.isdigit():
            return None
    elif isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        return None
    try:
        num = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return num if num >= 0 else None


def _parse_hotel_property(value: Any, *, destination: str) -> HotelResult | None:
    if not isinstance(value, Mapping):
        return None
    rate = _extract_rate(value.get("rate_per_night"))
    if rate is None:
        return None
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()
    coordinates = normalise_hotel_coordinates(value.get("gps_coordinates"))
    if coordinates is None:
        return None
    lat, lng = coordinates

    hotel_class = _finite_number(value.get("extracted_hotel_class"))
    if hotel_class is None or hotel_class > 5:
        hotel_class = 0.0
    rating = _finite_number(value.get("overall_rating"))
    if rating is not None and rating > 5:
        rating = None
    reviews = _nonnegative_int(value.get("reviews"))

    images = value.get("images")
    first_image = images[0] if isinstance(images, list) and images else None
    if isinstance(first_image, Mapping):
        first_image = first_image.get("thumbnail") or first_image.get("original_image")
    image = _safe_http_url(first_image) or ""
    booking_url = _safe_http_url(value.get("link")) or (
        "https://www.google.com/travel/hotels?q="
        + quote_plus(f"{name} {destination}".strip())
    )
    amenities = value.get("amenities")
    if not isinstance(amenities, list):
        amenities = []
    amenities = [x.strip() for x in amenities if isinstance(x, str) and x.strip()]

    def _text(key: str) -> str:
        val = value.get(key)
        return val.strip() if isinstance(val, str) else ""

    try:
        return HotelResult(
            hotel_name=name,
            hotel_class=hotel_class,
            overall_rating=rating,
            reviews=reviews,
            description=_text("description"),
            price_per_night=rate,
            amenities=amenities,
            check_in_time=_text("check_in_time"),
            check_out_time=_text("check_out_time"),
            location={"lat": lat, "lng": lng},
            image=image,
            booking_url=booking_url,
        )
    except ValidationError:
        return None


def _resolve_country_code(
    destination: str,
    override: str | None,
) -> str | None:
    if override is None:
        return country_alpha2(destination.rsplit(",", 1)[-1])
    if isinstance(override, str):
        return country_alpha2(override)
    return None


__all__ = ["SerpApiHotelProvider"]