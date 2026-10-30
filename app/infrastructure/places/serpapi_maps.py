"""SerpAPI Google Maps place-search adapter.

Owns provider request building, response parsing and caching. It returns
UNVERIFIED candidates: destination verification belongs to
``app.modules.places.service`` so every caller applies the same rules.

Raw candidates (not verified results) are cached, so a cache entry can never
smuggle an unverified place past the verification step.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Final, Protocol

from app.infrastructure.search.serpapi import SerpApiUnavailableError
from app.modules.places.exceptions import (
    PlaceSearchQuotaExhausted,
    PlaceSearchUnavailable,
)
from app.modules.places.models import PlaceCandidate
from app.modules.places.ports import SearchQuotaPort
from app.shared.geo import valid_lat_lng
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)


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
        ttl_hours: int,
    ) -> bool: ...


_CACHE_NAMESPACE: Final = "place_search_v2"

# Google Maps listings change slowly; a day of reuse saves most repeat
# quota without serving meaningfully stale venues.
_CACHE_TTL_HOURS: Final = 24

_MIN_ZOOM: Final = 3
_MAX_ZOOM: Final = 21


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""

    return " ".join(value.split())


def _category(value: Any) -> str:
    """SerpAPI ``type`` may be a string or a list of strings."""

    if isinstance(value, str):
        return _text(value)

    if isinstance(value, (list, tuple)):
        return ", ".join(part for part in map(_text, value) if part)

    return ""


def _rating(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    try:
        number = float(value)

    except (TypeError, ValueError, OverflowError):
        return None

    return number if math.isfinite(number) and 0.0 <= number <= 5.0 else None


def _reviews(value: Any) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value >= 0 else None

    return None


def _thumbnail(place: Mapping[str, Any]) -> str:
    """Best photo URL; list results and direct place results differ."""

    for key in ("thumbnail", "serpapi_thumbnail"):
        value = _text(place.get(key))

        if value:
            return value

    images = place.get("images") or []

    if isinstance(images, Mapping):
        images = [images]

    if not isinstance(images, list):
        return ""

    for image in images:
        if isinstance(image, str) and _text(image):
            return _text(image)

        if not isinstance(image, Mapping):
            continue

        for key in ("thumbnail", "image", "original_image"):
            value = _text(image.get(key))

            if value:
                return value

    return ""


def parse_candidate(raw: Any) -> PlaceCandidate | None:
    """Normalise one SerpAPI result, or None when it is unusable."""

    if not isinstance(raw, Mapping):
        return None

    title = _text(raw.get("title"))
    gps = raw.get("gps_coordinates")

    if not title or not isinstance(gps, Mapping):
        return None

    coordinates = valid_lat_lng(gps.get("latitude"), gps.get("longitude"))

    if coordinates is None:
        return None

    phone = _text(raw.get("phone"))

    return PlaceCandidate(
        title=title,
        latitude=coordinates[0],
        longitude=coordinates[1],
        address=_text(raw.get("address")),
        category=_category(raw.get("type")),
        rating=_rating(raw.get("rating")),
        reviews=_reviews(raw.get("reviews")),
        thumbnail=_thumbnail(raw),
        phone=phone or None,
    )


def parse_candidates(payload: Mapping[str, Any]) -> list[PlaceCandidate]:
    """List search → ``local_results``; a direct hit → ``place_results``."""

    raw_results = payload.get("local_results")

    if not isinstance(raw_results, list) or not raw_results:
        direct = payload.get("place_results")
        raw_results = [direct] if isinstance(direct, Mapping) else []

    return [
        candidate
        for candidate in map(parse_candidate, raw_results)
        if candidate is not None
    ]


def _from_cache(payload: Any) -> list[PlaceCandidate] | None:
    if not isinstance(payload, list):
        return None

    candidates: list[PlaceCandidate] = []

    for item in payload:
        if not isinstance(item, Mapping):
            return None

        try:
            candidate = PlaceCandidate(**item)

        except TypeError:
            return None

        if valid_lat_lng(candidate.latitude, candidate.longitude) is None:
            return None

        candidates.append(candidate)

    return candidates


class SerpApiMapsPlaceSearch:
    """Cached Google Maps search through one injected SerpAPI client."""

    def __init__(
        self,
        *,
        client: _SearchClient,
        cache: _SearchCache,
    ) -> None:
        self._client = client
        self._cache = cache

    def search(
        self,
        query: str,
        *,
        near: tuple[float, float] | None = None,
        zoom: int = 11,
        quota: SearchQuotaPort | None = None,
    ) -> list[PlaceCandidate]:
        clean_query = _text(query)

        if not clean_query:
            return []

        position = valid_lat_lng(*near) if near is not None else None
        zoom = min(max(int(zoom), _MIN_ZOOM), _MAX_ZOOM)

        cache_key = {
            "q": clean_query.casefold(),
            "ll": (
                [round(position[0], 4), round(position[1], 4)]
                if position is not None
                else None
            ),
            "zoom": zoom,
        }

        cached = _from_cache(
            self._cache.get(namespace=_CACHE_NAMESPACE, key=cache_key)
        )

        if cached is not None:
            return cached

        if quota is not None and not quota.try_acquire():
            safe_log(logger, logging.WARNING, "provider.places.quota_exhausted")
            raise PlaceSearchQuotaExhausted("Place-search budget is exhausted.")

        params: dict[str, Any] = {
            "engine": "google_maps",
            "type": "search",
            "q": clean_query,
            "hl": "en",
        }

        if position is not None:
            # Without ``ll`` SerpAPI searches from US datacentres and Google
            # biases ambiguous names towards US places.
            params["ll"] = f"@{position[0]},{position[1]},{zoom}z"

        try:
            payload = self._client.search(params)

        except SerpApiUnavailableError as exc:
            raise PlaceSearchUnavailable("Place search failed.") from exc

        candidates = parse_candidates(payload)

        self._cache.set(
            namespace=_CACHE_NAMESPACE,
            key=cache_key,
            payload=[asdict(candidate) for candidate in candidates],
            ttl_hours=_CACHE_TTL_HOURS,
        )

        return candidates


__all__ = [
    "SerpApiMapsPlaceSearch",
    "parse_candidate",
    "parse_candidates",
]
