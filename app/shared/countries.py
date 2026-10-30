"""Strict country-name → ISO 3166-1 alpha-2 resolution.

The single definition used by every feature that must agree on what a
destination country is (hotel verification, itinerary requirements, ...).
Deliberately exact: fuzzy matching would let a typo silently pick a different
country. Currency lookup keeps its own lenient fuzzy search on purpose.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Final, Mapping

import pycountry


# Common names pycountry's exact lookup does not know.
_COUNTRY_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "uk": "GB",
        "u.k.": "GB",
        "macau": "MO",
    }
)


def country_alpha2(value: Any) -> str | None:
    """Return the alpha-2 code for a country name/code, or None if unknown."""

    if not isinstance(value, str):
        return None

    candidate = " ".join(value.split())

    if not candidate:
        return None

    alias = _COUNTRY_ALIASES.get(candidate.casefold())

    if alias is not None:
        return alias

    try:
        return pycountry.countries.lookup(candidate).alpha_2
    except LookupError:
        return None


__all__ = [
    "country_alpha2",
]
