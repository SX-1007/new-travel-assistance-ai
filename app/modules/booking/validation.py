"""Pure input normalisers shared by every booking search provider.

One definition per rule, so flight and hotel searches cannot drift apart.
"""

from __future__ import annotations

import math
from typing import Any


def normalise_currency_code(value: Any) -> str:
    """Return an upper-case three-letter ISO-4217 code or raise ValueError."""

    if not isinstance(value, str):
        raise ValueError("currency must be a 3-letter ISO-4217 code")

    code = value.strip().upper()

    if not (len(code) == 3 and code.isascii() and code.isalpha()):
        raise ValueError("currency must be a 3-letter ISO-4217 code")

    return code


def normalise_adults(value: Any) -> int:
    """Return a positive whole traveller count or raise ValueError."""

    if isinstance(value, bool):
        raise ValueError("adults must be a positive integer")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("adults must be a positive integer") from exc

    if not math.isfinite(number) or not number.is_integer() or number < 1:
        raise ValueError("adults must be a positive integer")

    return int(number)


def parse_budget_cap(value: Any, *, field_name: str) -> float | None:
    """An optional spending cap; None or 0 both mean "no cap".

    Negative, NaN, infinite and boolean values are rejected. 0 means no cap
    because an unallocated category must not filter out every result.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative finite number")

    try:
        cap = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a non-negative finite number") from exc

    if not math.isfinite(cap) or cap < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number")

    return None if cap == 0 else cap


__all__ = [
    "normalise_currency_code",
    "normalise_adults",
    "parse_budget_cap",
]
