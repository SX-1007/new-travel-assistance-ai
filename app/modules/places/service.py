"""Provider place search with destination verification.

Every result returned here has passed the same checks:

1. inside ``MAX_KM_FROM_CITY`` of the trip city (when the city is anchored);
2. reverse-geocoded to the destination country;
3. reverse-geocoded locality equals the trip city, or the place carries
   distance evidence from step 1 (nearby day-trip place).

A missing city anchor never disables a check: it only removes the nearby
allowance, so exact-locality evidence is then required.
"""

from __future__ import annotations

import logging

from app.modules.places.models import (
    PlaceCandidate,
    SearchArea,
    VerifiedPlace,
)
from app.modules.places.policies import (
    CITY_SEARCH_ZOOM,
    MAX_CANDIDATES_INSPECTED,
    MAX_KM_FROM_CITY,
    choose_locality,
    clean_text,
    distance_from_anchor_km,
    join_query,
)
from app.modules.places.ports import (
    CityGeocoderPort,
    PlaceLocatorPort,
    PlaceSearchPort,
    SearchQuotaPort,
)
from app.shared.countries import country_alpha2
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)


class VerifiedPlaceSearch:
    """Search a provider and keep only destination-verified places."""

    def __init__(
        self,
        *,
        search: PlaceSearchPort,
        geocoder: CityGeocoderPort,
        locator: PlaceLocatorPort,
    ) -> None:
        self._search = search
        self._geocoder = geocoder
        self._locator = locator

    def area(self, *, country: str, city: str) -> SearchArea | None:
        """Build the verification area for one trip city.

        Returns None when the country is unknown or the city is blank.
        """

        clean_country = clean_text(country)
        clean_city = clean_text(city)
        country_code = country_alpha2(clean_country)

        if country_code is None or not clean_city:
            return None

        anchor: tuple[float, float] | None = None

        try:
            place = self._geocoder.geocode(join_query(clean_city, clean_country))

        except Exception:
            place = None
            safe_log(logger, logging.WARNING, "places.anchor_failed")

        if place is not None:
            anchor = (place.latitude, place.longitude)

        return SearchArea(
            country=clean_country,
            country_code=country_code,
            requested_city=clean_city,
            anchor=anchor,
        )

    def search(
        self,
        query: str,
        *,
        area: SearchArea,
        limit: int = 1,
        quota: SearchQuotaPort | None = None,
    ) -> list[VerifiedPlace]:
        """Return up to ``limit`` verified places, best provider rank first.

        Raises ``PlaceSearchUnavailable`` when the provider cannot answer.
        """

        if limit < 1 or not clean_text(query):
            return []

        candidates = self._search.search(
            clean_text(query),
            near=area.anchor,
            zoom=CITY_SEARCH_ZOOM,
            quota=quota,
        )

        verified: list[VerifiedPlace] = []

        for candidate in candidates[:MAX_CANDIDATES_INSPECTED]:
            place = self.verify(candidate, area)

            if place is None:
                continue

            verified.append(place)

            if len(verified) >= limit:
                break

        return verified

    def verify(
        self,
        candidate: PlaceCandidate,
        area: SearchArea,
    ) -> VerifiedPlace | None:
        """Apply the destination checks to one candidate."""

        distance = distance_from_anchor_km(
            candidate.latitude,
            candidate.longitude,
            area.anchor,
        )

        if distance is not None and distance > MAX_KM_FROM_CITY:
            safe_log(logger, logging.INFO, "places.out_of_area")
            return None

        try:
            country_code = self._locator.resolve_country_code(
                candidate.latitude,
                candidate.longitude,
            )
            locality_names = (
                self._locator.resolve_locality_names(
                    candidate.latitude,
                    candidate.longitude,
                    area.country_code,
                )
                if country_code == area.country_code
                else None
            )

        except Exception:
            safe_log(logger, logging.WARNING, "places.locator_failed")
            return None

        if country_code != area.country_code:
            safe_log(logger, logging.INFO, "places.country_unverified")
            return None

        locality = choose_locality(
            locality_names or (),
            area.requested_city,
            nearby_allowed=distance is not None,
        )

        if locality is None:
            safe_log(logger, logging.INFO, "places.locality_unverified")
            return None

        return VerifiedPlace(
            candidate=candidate,
            country_code=area.country_code,
            requested_city=area.requested_city,
            verified_locality=locality,
            distance_from_requested_city_km=distance,
        )


__all__ = [
    "VerifiedPlaceSearch",
]
