"""Small reusable scalar types and validation helpers.

Keep this module limited to primitive cross-cutting types. Domain models belong in
``app.modules.*.models`` and HTTP transport models belong in ``app.api.schemas``.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

from pydantic import StringConstraints


NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

CountryCode = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z]{2}$"),
]


def ensure_finite_number(value: Any, *, field_name: str) -> Any:
    """Reject booleans, non-numeric values, NaN and infinities."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")

    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")

    return value


__all__ = [
    "NonEmptyText",
    "CountryCode",
    "ensure_finite_number",
]