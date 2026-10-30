"""Deterministic nearby-amenity rules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from app.modules.places.policies import clean_text


@dataclass(frozen=True, slots=True)
class AmenityTerms:
    """Search terms for one amenity kind, plus local-language extras."""

    triggers: tuple[str, ...]
    exact: frozenset[str]
    queries: tuple[str, ...]
    local_queries: Mapping[str, tuple[str, ...]]


# Local-language queries matter where English listings are sparse (Japan:
# koban/police boxes are often listed only as 交番). Keyed by ISO country code
# so adding a country is a data change, not a code change.
AMENITY_TERMS: Final[tuple[AmenityTerms, ...]] = (
    AmenityTerms(
        triggers=("police",),
        exact=frozenset({"koban", "交番", "警察署"}),
        queries=("police station",),
        local_queries=MappingProxyType({"JP": ("交番", "警察署")}),
    ),
    AmenityTerms(
        triggers=("hospital", "clinic"),
        exact=frozenset({"病院"}),
        queries=("hospital",),
        local_queries=MappingProxyType({"JP": ("病院",)}),
    ),
    AmenityTerms(
        # Not a bare "fire" trigger: "fireworks" is not a fire station.
        triggers=("fire station", "fire department", "fire brigade"),
        exact=frozenset({"fire", "消防署"}),
        queries=("fire station",),
        local_queries=MappingProxyType({"JP": ("消防署",)}),
    ),
    AmenityTerms(
        triggers=("pharmacy", "drugstore", "chemist"),
        exact=frozenset({"薬局"}),
        queries=("pharmacy",),
        local_queries=MappingProxyType({"JP": ("薬局",)}),
    ),
)

# Close search first; widen only when nothing is close.
SEARCH_ZOOMS: Final = (15, 12)

MAX_RESULTS: Final = 5

# Ignore anything this far from the origin even if the provider returns it.
MAX_RESULT_DISTANCE_KM: Final = 50.0


def amenity_queries(category: str, country_code: str | None) -> tuple[str, ...]:
    """Provider search terms for a free-text amenity category."""

    raw = clean_text(category)
    key = raw.casefold()

    if not key:
        return ()

    for terms in AMENITY_TERMS:
        if key in terms.exact or any(trigger in key for trigger in terms.triggers):
            local = terms.local_queries.get((country_code or "").upper(), ())
            return tuple(dict.fromkeys((*terms.queries, *local)))

    return (raw,)


__all__ = [
    "AMENITY_TERMS",
    "MAX_RESULTS",
    "MAX_RESULT_DISTANCE_KM",
    "SEARCH_ZOOMS",
    "AmenityTerms",
    "amenity_queries",
]
