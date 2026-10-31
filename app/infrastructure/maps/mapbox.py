"""Mapbox forward geocoding: place name → coordinates.

Reverse geocoding lives in ``reverse_geocoder.py`` with its own HTTP session
and caches, so a failure in one never affects the other.
"""

from __future__ import annotations

import logging
import math
from threading import RLock
from typing import Any, Final

import requests
from cachetools import TTLCache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.shared.geo import GeocodedPlace
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)


_REQUEST_TIMEOUT_SECONDS: Final = 10.0

_CACHE_SIZE: Final = 2048
_CACHE_TTL_SECONDS: Final = 86_400

_FORWARD_GEOCODING_URL: Final = (
    "https://api.mapbox.com/"
    "search/searchbox/v1/forward"
)

# Status codes that mean "try again later". Everything else that is not 200
# is a permanent answer for this query and is cached as a negative result.
_TRANSIENT_STATUS_CODES: Final = frozenset(
    {429, 500, 502, 503, 504}
)


_RETRY_STRATEGY = Retry(
    total=3,
    connect=3,
    read=1,
    status=3,
    backoff_factor=0.5,
    status_forcelist=tuple(
        sorted(_TRANSIENT_STATUS_CODES)
    ),
    allowed_methods=frozenset(
        {"GET"}
    ),
    respect_retry_after_header=True,
    raise_on_status=False,
)


def _build_http_session(
) -> requests.Session:
    session = requests.Session()

    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=_RETRY_STRATEGY,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def _normalise_proximity(
    proximity: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Return a valid (longitude, latitude) bias or None."""

    if proximity is None or len(proximity) != 2:
        return None

    lng, lat = proximity

    if isinstance(lng, bool) or isinstance(lat, bool):
        return None

    try:
        lng = float(lng)
        lat = float(lat)

    except (TypeError, ValueError, OverflowError):
        return None

    if not (math.isfinite(lng) and math.isfinite(lat)):
        return None

    if not (-180.0 <= lng <= 180.0 and -90.0 <= lat <= 90.0):
        return None

    return lng, lat


def _parse_first_feature(
    payload: dict[str, Any],
    fallback_name: str,
) -> GeocodedPlace | None:
    """Extract the first usable match from a Search Box response."""

    features = payload.get("features") or []

    if not features or not isinstance(features[0], dict):
        return None

    feature = features[0]
    properties = feature.get("properties") or {}
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates")

    if (
        not isinstance(coordinates, (list, tuple))
        or len(coordinates) < 2
    ):
        return None

    try:
        lng = float(coordinates[0])
        lat = float(coordinates[1])

    except (TypeError, ValueError, OverflowError):
        return None

    if not (math.isfinite(lng) and math.isfinite(lat)):
        return None

    if not (-180.0 <= lng <= 180.0 and -90.0 <= lat <= 90.0):
        return None

    name = properties.get("name")

    if not isinstance(name, str) or not name.strip():
        name = fallback_name

    context = properties.get("place_formatted")

    place_name = (
        f"{name.strip()}, {context.strip()}"
        if isinstance(context, str) and context.strip()
        else name.strip()
    )

    return GeocodedPlace(
        latitude=lat,
        longitude=lng,
        place_name=place_name,
    )


class MapboxForwardGeocoder:
    """Thread-safe Mapbox forward geocoder with its own cache."""

    def __init__(
        self,
        *,
        access_token: str,
        session: (
            requests.Session | None
        ) = None,
    ) -> None:
        token = access_token.strip()

        if not token:
            raise ValueError(
                "access_token must not be empty"
            )

        self._access_token = token

        self._session = (
            session
            or _build_http_session()
        )

        self._owns_session = (
            session is None
        )

        # ``None`` values are cached negative results: Mapbox answered,
        # but had no usable match for this query.
        self._cache: TTLCache[
            tuple[str, tuple[float, float] | None],
            GeocodedPlace | None,
        ] = TTLCache(
            maxsize=_CACHE_SIZE,
            ttl=_CACHE_TTL_SECONDS,
        )

        self._lock = RLock()

    def close(self) -> None:
        """Close the internally-owned HTTP session."""

        if self._owns_session:
            self._session.close()

    def geocode(
        self,
        query: str,
        *,
        proximity: tuple[float, float] | None = None,
    ) -> GeocodedPlace | None:
        """The best match for a place name, or None.

        ``proximity`` is an optional (longitude, latitude) bias. "No match" answers
        are cached; transport failures are not.
        """

        if not isinstance(query, str):
            return None

        clean_query = " ".join(query.split())

        if not clean_query:
            return None

        bias = _normalise_proximity(proximity)

        cache_key = (
            clean_query.casefold(),
            (
                (round(bias[0], 5), round(bias[1], 5))
                if bias is not None
                else None
            ),
        )

        with self._lock:
            # Membership is required: ``.get()`` cannot distinguish a
            # cached negative result (None) from an absent key.
            if cache_key in self._cache:
                return self._cache[cache_key]

        params: dict[str, Any] = {
            "q": clean_query,
            "limit": 1,
            "access_token": self._access_token,
        }

        if bias is not None:
            params["proximity"] = f"{bias[0]},{bias[1]}"

        try:
            response = self._session.get(
                _FORWARD_GEOCODING_URL,
                params=params,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )

        except requests.RequestException:
            safe_log(
                logger,
                logging.WARNING,
                "provider.mapbox.forward_transport_failed",
            )

            return None

        if response.status_code != 200:
            safe_log(
                logger,
                logging.WARNING,
                "provider.mapbox.forward_http_failed",
                status_code=response.status_code,
            )

            if response.status_code not in _TRANSIENT_STATUS_CODES:
                self._store(cache_key, None)

            return None

        try:
            payload = response.json()

        except ValueError:
            safe_log(
                logger,
                logging.WARNING,
                "provider.mapbox.forward_invalid_json",
            )

            return None

        if not isinstance(payload, dict):
            return None

        place = _parse_first_feature(payload, clean_query)

        if place is None:
            safe_log(
                logger,
                logging.INFO,
                "provider.mapbox.forward_empty",
            )

        self._store(cache_key, place)

        return place

    def _store(
        self,
        key: tuple[str, tuple[float, float] | None],
        value: GeocodedPlace | None,
    ) -> None:
        with self._lock:
            self._cache[key] = value


__all__ = [
    "MapboxForwardGeocoder",
]
