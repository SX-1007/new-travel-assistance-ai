"""Deterministic destination-resolution rules."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final


# Countries where the country name itself is the practical city/locality.
# Keyed by an already-verified ISO country code, so these destinations do
# not need an AI call to choose their primary planning city.
CITY_STATE_PRIMARY_CITY: Final[Mapping[str, str]] = MappingProxyType(
    {
        "SG": "Singapore",
        "MC": "Monaco",
        "VA": "Vatican City",
    }
)

# How many AI candidates may be proposed before giving up.
MAX_CITY_SELECTION_ATTEMPTS: Final = 3


def clean_city(value: Any) -> str:
    """Normalise whitespace in one city/locality value."""

    if not isinstance(value, str):
        return ""

    return " ".join(value.split())


def geocoding_query(city: str, country: str) -> str:
    """Query that geocodes a locality together with its trusted country."""

    return f"{clean_city(city)}, {clean_city(country)}"


def city_state_primary_city(country_code: str) -> str | None:
    """Return the fixed planning city for a city-state, if any."""

    return CITY_STATE_PRIMARY_CITY.get(country_code)


__all__ = [
    "CITY_STATE_PRIMARY_CITY",
    "MAX_CITY_SELECTION_ATTEMPTS",
    "city_state_primary_city",
    "clean_city",
    "geocoding_query",
]
