"""Mapbox reverse geocoding: coordinates → country and locality names.

Used to verify that a place, hotel or city really is where a provider says.
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

from app.shared.logging import safe_log


logger = logging.getLogger(__name__)


_REQUEST_TIMEOUT_SECONDS: Final = 10.0

_CACHE_SIZE: Final = 4096
_CACHE_TTL_SECONDS: Final = 86_400

_REVERSE_GEOCODING_URL: Final = (
    "https://api.mapbox.com/"
    "search/geocode/v6/reverse"
)


_RETRY_STRATEGY = Retry(
    total=3,
    connect=3,
    read=1,
    status=3,
    backoff_factor=0.5,
    status_forcelist=(
        429,
        500,
        502,
        503,
        504,
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

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    return session


def _normalise_coordinates(
    latitude: float,
    longitude: float,
) -> tuple[float, float] | None:
    """Finite, in-range coordinates (no booleans), or None."""

    if (
        isinstance(latitude, bool)
        or isinstance(longitude, bool)
    ):
        return None

    try:
        lat = float(latitude)
        lng = float(longitude)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None

    if (
        not math.isfinite(lat)
        or not math.isfinite(lng)
    ):
        return None

    if not (
        -90.0 <= lat <= 90.0
    ):
        return None

    if not (
        -180.0 <= lng <= 180.0
    ):
        return None

    return lat, lng


class MapboxReverseGeocoder:
    """Thread-safe Mapbox reverse geocoder with its own caches."""

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

        self._country_cache: TTLCache[
            tuple[float, float],
            str | None,
        ] = TTLCache(
            maxsize=_CACHE_SIZE,
            ttl=_CACHE_TTL_SECONDS,
        )

        self._locality_cache: TTLCache[
            tuple[
                float,
                float,
                str,
            ],
            tuple[str, ...],
        ] = TTLCache(
            maxsize=_CACHE_SIZE,
            ttl=_CACHE_TTL_SECONDS,
        )

        self._country_lock = RLock()
        self._locality_lock = RLock()

    def close(self) -> None:
        """Close the internally-owned HTTP session."""

        if self._owns_session:
            self._session.close()

    def resolve_country_code(
        self,
        latitude: float,
        longitude: float,
    ) -> str | None:
        """The ISO 3166 alpha-2 country containing the coordinates, or None.

        "No country" answers are cached; transport failures are not.
        """

        coordinates = (
            _normalise_coordinates(
                latitude,
                longitude,
            )
        )

        if coordinates is None:
            return None

        lat, lng = coordinates

        cache_key = (
            round(lat, 5),
            round(lng, 5),
        )

        with self._country_lock:
            if (
                cache_key
                in self._country_cache
            ):
                return (
                    self._country_cache[
                        cache_key
                    ]
                )

        payload = self._reverse_lookup(
            latitude=lat,
            longitude=lng,
            types="country",
        )

        if payload is None:
            return None

        country_code: str | None = None

        features = (
            payload.get("features")
            or []
        )

        if (
            features
            and isinstance(
                features[0],
                dict,
            )
        ):
            properties = (
                features[0].get(
                    "properties"
                )
                or {}
            )

            context = (
                properties.get(
                    "context"
                )
                or {}
            )

            country = (
                context.get(
                    "country"
                )
                or {}
            )

            raw_code = country.get(
                "country_code"
            )

            if (
                raw_code is None
                and properties.get(
                    "feature_type"
                )
                == "country"
            ):
                raw_code = (
                    properties.get(
                        "country_code"
                    )
                )

            if isinstance(
                raw_code,
                str,
            ):
                normalised = (
                    raw_code
                    .strip()
                    .upper()
                )

                if (
                    len(normalised) == 2
                    and normalised.isalpha()
                ):
                    country_code = (
                        normalised
                    )

        with self._country_lock:
            self._country_cache[
                cache_key
            ] = country_code

        return country_code

    def resolve_locality_names(
        self,
        latitude: float,
        longitude: float,
        destination_country_code: str,
    ) -> tuple[str, ...] | None:
        """Locality and admin-area names for the coordinates.

        None: the request failed and may be retried later. An empty tuple: Mapbox
        answered but gave no usable locality.
        """

        coordinates = (
            _normalise_coordinates(
                latitude,
                longitude,
            )
        )

        if coordinates is None:
            return ()

        if not isinstance(
            destination_country_code,
            str,
        ):
            return ()

        country_code = (
            destination_country_code
            .strip()
            .upper()
        )

        if (
            len(country_code) != 2
            or not country_code.isalpha()
        ):
            return ()

        lat, lng = coordinates

        cache_key = (
            round(lat, 5),
            round(lng, 5),
            country_code,
        )

        with self._locality_lock:
            if (
                cache_key
                in self._locality_cache
            ):
                return (
                    self._locality_cache[
                        cache_key
                    ]
                )

        payload = self._reverse_lookup(
            latitude=lat,
            longitude=lng,
            country=(
                country_code.lower()
            ),
        )

        if payload is None:
            return None

        names: list[str] = []

        features = (
            payload.get("features")
            or []
        )

        if (
            features
            and isinstance(
                features[0],
                dict,
            )
        ):
            properties = (
                features[0].get(
                    "properties"
                )
                or {}
            )

            context = (
                properties.get(
                    "context"
                )
                or {}
            )

            country = (
                context.get(
                    "country"
                )
                or {}
            )

            raw_country = country.get(
                "country_code"
            )

            country_matches = (
                isinstance(
                    raw_country,
                    str,
                )
                and raw_country
                .strip()
                .upper()
                == country_code
            )

            if country_matches:
                feature_type = (
                    properties.get(
                        "feature_type"
                    )
                )

                if feature_type in {
                    "region",
                    "district",
                    "place",
                    "locality",
                    "neighborhood",
                }:
                    own_name = (
                        properties.get(
                            "name"
                        )
                    )

                    if (
                        isinstance(
                            own_name,
                            str,
                        )
                        and own_name.strip()
                    ):
                        names.append(
                            own_name.strip()
                        )

                for layer in (
                    "neighborhood",
                    "locality",
                    "place",
                    "district",
                    "region",
                ):
                    component = (
                        context.get(
                            layer
                        )
                        or {}
                    )

                    name = (
                        component.get(
                            "name"
                        )
                        if isinstance(
                            component,
                            dict,
                        )
                        else None
                    )

                    if (
                        isinstance(
                            name,
                            str,
                        )
                        and name.strip()
                    ):
                        names.append(
                            name.strip()
                        )

        unique_names: list[str] = []
        observed: set[str] = set()

        for name in names:
            normalised_name = " ".join(
                name.split()
            )

            key = (
                normalised_name.casefold()
            )

            if key in observed:
                continue

            observed.add(key)

            unique_names.append(
                normalised_name
            )

        result = tuple(
            unique_names
        )

        with self._locality_lock:
            self._locality_cache[
                cache_key
            ] = result

        return result

    def _reverse_lookup(
        self,
        *,
        latitude: float,
        longitude: float,
        types: str | None = None,
        country: str | None = None,
    ) -> dict[str, Any] | None:
        """One reverse-geocoding request; retries are left to the session's HTTP adapter."""

        params: dict[str, Any] = {
            "longitude": longitude,
            "latitude": latitude,
            "limit": 1,
            "language": "en",
            "access_token":
                self._access_token,
        }

        if types is not None:
            params["types"] = types

        if country is not None:
            params["country"] = country

        try:
            response = (
                self._session.get(
                    _REVERSE_GEOCODING_URL,
                    params=params,
                    timeout=(
                        _REQUEST_TIMEOUT_SECONDS
                    ),
                )
            )

        except requests.RequestException:
            safe_log(
                logger,
                logging.WARNING,
                (
                    "provider.mapbox."
                    "reverse_transport_failed"
                ),
            )

            return None

        if response.status_code != 200:
            safe_log(
                logger,
                logging.WARNING,
                (
                    "provider.mapbox."
                    "reverse_http_failed"
                ),
                status_code=(
                    response.status_code
                ),
            )

            return None

        try:
            payload = response.json()

        except ValueError:
            safe_log(
                logger,
                logging.WARNING,
                (
                    "provider.mapbox."
                    "reverse_invalid_json"
                ),
            )

            return None

        if not isinstance(
            payload,
            dict,
        ):
            return None

        return payload


__all__ = [
    "MapboxReverseGeocoder",
]