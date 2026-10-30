"""Pure activity identity helpers used by itinerary composition and validation."""

from __future__ import annotations

import math
from collections.abc import (
    Mapping,
    Sequence,
)
from typing import Any


def _normalise_text(
    value: Any,
) -> str:
    if not isinstance(value, str):
        return ""

    folded = value.strip().casefold()

    unicode_safe = "".join(
        character
        if character.isalnum()
        else " "
        for character in folded
    )

    return " ".join(
        unicode_safe.split()
    )


def activity_identity_key(
    activity: Any,
) -> str | None:
    """Return a stable provider-grounded identity for duplicate detection."""

    if not isinstance(
        activity,
        Mapping,
    ):
        return None

    location = activity.get(
        "location"
    )

    location = (
        location
        if isinstance(location, Mapping)
        else {}
    )

    name = _normalise_text(
        location.get("place_name")
        or activity.get("name")
    )

    locality = _normalise_text(
        location.get("verified_locality")
        or location.get("requested_city")
    )

    if not name:
        return None

    latitude = location.get(
        "latitude"
    )

    longitude = location.get(
        "longitude"
    )

    if (
        isinstance(latitude, (int, float))
        and not isinstance(latitude, bool)
        and isinstance(longitude, (int, float))
        and not isinstance(longitude, bool)
        and math.isfinite(float(latitude))
        and math.isfinite(float(longitude))
    ):
        return (
            f"{name}|"
            f"{float(latitude):.5f}|"
            f"{float(longitude):.5f}"
        )

    return f"{name}|{locality}"


def find_duplicate_activity_paths(
    itinerary: Sequence[Any],
) -> tuple[tuple[str, str], ...]:
    """Return (duplicate_path, first_path) for repeated activities."""

    seen: dict[str, str] = {}

    duplicates: list[
        tuple[str, str]
    ] = []

    for day_index, raw_day in enumerate(
        itinerary
    ):
        if not isinstance(
            raw_day,
            Mapping,
        ):
            continue

        activities = raw_day.get(
            "activities"
        )

        if (
            not isinstance(
                activities,
                Sequence,
            )
            or isinstance(
                activities,
                (str, bytes),
            )
        ):
            continue

        for (
            activity_index,
            activity,
        ) in enumerate(activities):
            key = activity_identity_key(
                activity
            )

            if key is None:
                continue

            path = (
                f"itinerary[{day_index}]."
                f"activities[{activity_index}]"
            )

            first_path = seen.get(key)

            if first_path is None:
                seen[key] = path
            else:
                duplicates.append(
                    (
                        path,
                        first_path,
                    )
                )

    return tuple(duplicates)


__all__ = [
    "activity_identity_key",
    "find_duplicate_activity_paths",
]