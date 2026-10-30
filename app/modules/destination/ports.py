"""Contracts required by destination resolution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.shared.geo import GeocodedPlace


class ForwardGeocoderPort(Protocol):
    """Place name → coordinates."""

    def geocode(
        self,
        query: str,
        *,
        proximity: tuple[float, float] | None = None,
    ) -> GeocodedPlace | None:
        """Return the best match, or None when there is no usable answer."""
        ...


class CountryLocatorPort(Protocol):
    """Coordinates → ISO 3166-1 alpha-2 country code."""

    def resolve_country_code(
        self,
        latitude: float,
        longitude: float,
    ) -> str | None:
        """Return the containing country, or None when it is unknown."""
        ...


class PrimaryCityPickerPort(Protocol):
    """Suggest one practical planning city for a country-only trip."""

    def propose_primary_city(
        self,
        *,
        country: str,
        start_date: str,
        end_date: str,
        num_people: int,
        rejected: Sequence[str],
    ) -> str:
        """
        Return one candidate city name (possibly empty).

        Raise on provider failure; the service treats any exception as
        "selection unavailable".
        """
        ...


__all__ = [
    "CountryLocatorPort",
    "ForwardGeocoderPort",
    "PrimaryCityPickerPort",
]
