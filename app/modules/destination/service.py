"""Resolve optional user destination input into verified planning cities.

The public form may omit city information. This service guarantees that
downstream budget, provider, itinerary, and map operations receive at least
one locality whose coordinates are verified to lie inside the destination
country.
"""

from __future__ import annotations

import logging

from app.modules.destination.exceptions import (
    DestinationResolutionInvalid,
    DestinationResolutionUnavailable,
)
from app.modules.destination.models import ResolvedDestination
from app.modules.destination.policies import (
    MAX_CITY_SELECTION_ATTEMPTS,
    city_state_primary_city,
    clean_city,
    geocoding_query,
)
from app.modules.destination.ports import (
    CountryLocatorPort,
    ForwardGeocoderPort,
    PrimaryCityPickerPort,
)
from app.shared.countries import country_alpha2


logger = logging.getLogger(__name__)


class DestinationResolutionService:
    """Country + optional cities → verified planning cities."""

    def __init__(
        self,
        *,
        geocoder: ForwardGeocoderPort,
        country_locator: CountryLocatorPort,
        city_picker: PrimaryCityPickerPort,
    ) -> None:
        self._geocoder = geocoder
        self._country_locator = country_locator
        self._city_picker = city_picker

    def resolve(
        self,
        *,
        country: str,
        requested_cities: list[str],
        start_date: str,
        end_date: str,
        num_people: int,
    ) -> ResolvedDestination:
        """Return verified, server-owned planning cities.

        Raises ``DestinationResolutionInvalid`` for bad user input and
        ``DestinationResolutionUnavailable`` when verification cannot run.
        """

        clean_country = clean_city(country)
        country_code = country_alpha2(clean_country)

        if country_code is None:
            raise DestinationResolutionInvalid(
                "Destination country could not be verified."
            )

        # ── User supplied one or more cities ──────────────────────────
        if requested_cities:
            cities = self._verify_user_cities(
                requested_cities,
                country=clean_country,
                country_code=country_code,
            )

            return ResolvedDestination(
                country=clean_country,
                country_code=country_code,
                cities=cities,
                source="user",
            )

        # ── Country-only city-state request: deterministic ────────────
        city_state = city_state_primary_city(country_code)

        if city_state is not None:
            return ResolvedDestination(
                country=clean_country,
                country_code=country_code,
                cities=(city_state,),
                source="city_state",
            )

        # ── Other country-only requests: AI proposes, Mapbox verifies ──
        city = self._select_primary_city(
            country=clean_country,
            country_code=country_code,
            start_date=start_date,
            end_date=end_date,
            num_people=num_people,
        )

        return ResolvedDestination(
            country=clean_country,
            country_code=country_code,
            cities=(city,),
            source="ai",
        )

    # ── Internals ───────────────────────────────────────────────────

    def _verify_user_cities(
        self,
        requested_cities: list[str],
        *,
        country: str,
        country_code: str,
    ) -> tuple[str, ...]:
        resolved: list[str] = []
        seen: set[str] = set()

        for requested_city in requested_cities:
            verified = self._verified_city(
                city=requested_city,
                country=country,
                country_code=country_code,
                user_supplied=True,
            )

            if not verified:
                continue

            key = verified.casefold()

            if key not in seen:
                seen.add(key)
                resolved.append(verified)

        if not resolved:
            raise DestinationResolutionInvalid(
                "At least one valid destination state/city is required "
                "when city data is supplied."
            )

        return tuple(resolved)

    def _select_primary_city(
        self,
        *,
        country: str,
        country_code: str,
        start_date: str,
        end_date: str,
        num_people: int,
    ) -> str:
        rejected: list[str] = []

        for _ in range(MAX_CITY_SELECTION_ATTEMPTS):
            try:
                proposal = self._city_picker.propose_primary_city(
                    country=country,
                    start_date=start_date,
                    end_date=end_date,
                    num_people=num_people,
                    rejected=tuple(rejected),
                )

            except Exception as exc:
                # Keep the public response generic, but preserve the error
                # class in server logs so failures are diagnosable.
                logger.warning(
                    "AI destination selection failed for country_code=%s: %s",
                    country_code,
                    type(exc).__name__,
                )

                raise DestinationResolutionUnavailable(
                    "AI destination selection is currently unavailable."
                ) from exc

            candidate = clean_city(proposal)

            if not candidate:
                rejected.append("empty candidate")
                continue

            verified = self._verified_city(
                city=candidate,
                country=country,
                country_code=country_code,
                user_supplied=False,
            )

            if verified:
                return verified

            rejected.append(candidate)

        raise DestinationResolutionUnavailable(
            "A verified planning city could not be selected for this country."
        )

    def _verified_city(
        self,
        *,
        city: str,
        country: str,
        country_code: str,
        user_supplied: bool,
    ) -> str | None:
        """Return the verified locality or reject a wrong-country one."""

        candidate = clean_city(city)

        if not candidate:
            if user_supplied:
                raise DestinationResolutionInvalid(
                    "Every provided destination state/city must be non-empty."
                )

            return None

        # Forward geocode the locality together with the trusted country.
        place = self._geocoder.geocode(
            geocoding_query(candidate, country)
        )

        if place is None:
            raise DestinationResolutionUnavailable(
                "Destination locality verification is currently unavailable."
            )

        # Reverse-check the coordinates. This prevents an AI-selected or
        # user-entered locality in another country from entering planning
        # state.
        verified_code = self._country_locator.resolve_country_code(
            place.latitude,
            place.longitude,
        )

        if verified_code is None:
            raise DestinationResolutionUnavailable(
                "Destination country verification is currently unavailable."
            )

        if verified_code != country_code:
            if user_supplied:
                raise DestinationResolutionInvalid(
                    f"'{candidate}' could not be verified inside {country}."
                )

            return None

        # Keep the practical city label supplied by the user or the AI.
        # Mapbox may canonicalise a travel city into a wider administrative
        # region (for example "Tokyo" → "Tokyo Prefecture"), which breaks
        # downstream hotel locality matching.
        return candidate


__all__ = [
    "DestinationResolutionService",
]
