"""Exact map pins for places named in a chat reply.

The chat UI pins ``[MAP: …]`` places by searching each name + address again
instead of trusting coordinates the model copied into its reply (it garbles
digits). The provider search is already cached across processes; a short
in-process cache on top spares that store a round trip on every re-render.

The old router cached every miss for the life of the process (a provider
hiccup returning nothing stuck until restart) and had no lock around a
cache shared by worker threads.
"""

from __future__ import annotations

import threading
from typing import Final

from cachetools import TTLCache

from app.modules.places.models import PlaceCandidate
from app.modules.places.ports import PlaceSearchPort


# Street-level view; the bias anchor is rounded (~1 km) so nearby repeat
# lookups share cache entries.
_PIN_ZOOM: Final = 13
_ANCHOR_DECIMALS: Final = 2
_CACHE_SIZE: Final = 512
_CACHE_TTL_SECONDS: Final = 3600


class PlacePinLocator:
    """Free-text place → the provider's top match with coordinates."""

    def __init__(
        self,
        *,
        search: PlaceSearchPort,
        cache_size: int = _CACHE_SIZE,
        cache_ttl_seconds: float = _CACHE_TTL_SECONDS,
    ) -> None:
        self._search = search
        self._cache: TTLCache[tuple[str, tuple[float, float] | None], PlaceCandidate | None] = TTLCache(
            maxsize=cache_size,
            ttl=cache_ttl_seconds,
        )
        self._lock = threading.Lock()

    def locate(self, query: str, *, near: tuple[float, float] | None = None) -> PlaceCandidate | None:
        """The best match, or None when nothing was found.

        Raises ``PlaceSearchUnavailable`` (not cached) when the provider
        fails. Blocking: call it from a worker thread.
        """

        clean = " ".join(query.split())

        if not clean:
            return None

        anchor = (
            (round(near[0], _ANCHOR_DECIMALS), round(near[1], _ANCHOR_DECIMALS))
            if near is not None
            else None
        )
        key = (clean.casefold(), anchor)

        with self._lock:
            if key in self._cache:
                return self._cache[key]

        candidates = self._search.search(clean, near=anchor, zoom=_PIN_ZOOM)
        best = candidates[0] if candidates else None

        with self._lock:
            self._cache[key] = best

        return best


__all__ = [
    "PlacePinLocator",
]
