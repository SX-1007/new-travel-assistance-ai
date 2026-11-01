"""Pure activity-bucket cleanup and rescue helpers for itinerary planning.

This module has no knowledge of AgentState, LLMs, providers, persistence, or
orchestration. Provider-specific activity generation is injected through the
``rescue_round`` callback.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.modules.itinerary.activity_identity import (
    activity_identity_key,
)


ActivityBucket = Mapping[
    int,
    Sequence[Mapping[str, Any]],
]

# Called with {day: places still needed} and the round number.
RescueRound = Callable[
    [Mapping[int, int], int],
    Mapping[
        int,
        Sequence[Mapping[str, Any]],
    ],
]


def deduplicate_activity_buckets(
    by_day: ActivityBucket,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    int,
]:
    """Remove repeated activities across the itinerary, keeping first use."""

    seen: set[str] = set()

    cleaned: dict[
        int,
        list[dict[str, Any]],
    ] = {}

    removed = 0

    for raw_day in sorted(by_day):
        try:
            day = int(raw_day)

        except (TypeError, ValueError):
            continue

        kept: list[
            dict[str, Any]
        ] = []

        raw_activities = (
            by_day.get(raw_day)
            or []
        )

        for raw_activity in raw_activities:
            if not isinstance(
                raw_activity,
                Mapping,
            ):
                continue

            activity = copy.deepcopy(
                dict(raw_activity)
            )

            key = activity_identity_key(
                activity
            )

            if (
                key is not None
                and key in seen
            ):
                removed += 1
                continue

            if key is not None:
                seen.add(key)

            kept.append(activity)

        for (
            order,
            activity,
        ) in enumerate(
            kept,
            start=1,
        ):
            activity["order"] = order

        cleaned[day] = kept

    return cleaned, removed


def missing_activity_days(
    expected_days: Sequence[int],
    by_day: ActivityBucket,
) -> tuple[int, ...]:
    """Return expected day numbers that currently have no usable activities."""

    missing: list[int] = []

    for raw_day in expected_days:
        try:
            day = int(raw_day)

        except (TypeError, ValueError):
            continue

        activities = by_day.get(
            day
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
            or not activities
        ):
            missing.append(day)

    return tuple(missing)


def short_activity_days(
    expected_days: Sequence[int],
    by_day: ActivityBucket,
    targets: Mapping[int, int] | None = None,
) -> dict[int, int]:
    """Day → how many more activities it needs to reach its target.

    A day without a target needs one (only empty days are short).
    """

    short: dict[int, int] = {}

    for raw_day in expected_days:
        try:
            day = int(raw_day)
        except (TypeError, ValueError):
            continue

        activities = by_day.get(day)
        count = len(activities) if isinstance(activities, Sequence) and not isinstance(activities, (str, bytes)) else 0
        needed = max(1, (targets or {}).get(day, 1)) - count

        if needed > 0:
            short[day] = needed

    return short


@dataclass(frozen=True)
class ActivityRescueResult:
    """Result of deterministic cross-day activity rescue."""

    by_day: dict[
        int,
        list[dict[str, Any]],
    ]

    # Days still empty: the plan can't use them.
    missing_days: tuple[
        int,
        ...,
    ]

    rounds_used: int

    removed_duplicates: int

    # Days with some activities but fewer than their target: usable.
    short_days: tuple[int, ...] = ()


def rescue_activity_buckets(
    expected_days: Sequence[int],
    initial_by_day: ActivityBucket,
    *,
    rescue_round: RescueRound,
    max_rounds: int = 3,
    targets: Mapping[int, int] | None = None,
) -> ActivityRescueResult:
    """Top up days below their target (``targets``; default: empty days only)
    while keeping every activity a day already has.

    ``rescue_round`` is dependency injection: this pure domain helper decides
    when rescue is required, while an outer provider/application layer decides
    how replacement activities are obtained.
    """

    if max_rounds < 0:
        raise ValueError(
            "max_rounds must be non-negative"
        )

    current, removed_duplicates = (
        deduplicate_activity_buckets(
            initial_by_day
        )
    )

    for raw_day in expected_days:
        day = int(raw_day)

        current.setdefault(
            day,
            [],
        )

    short = short_activity_days(expected_days, current, targets)

    rounds_used = 0

    for round_number in range(
        1,
        max_rounds + 1,
    ):
        if not short:
            break

        rounds_used = round_number

        additions = rescue_round(
            dict(short),
            round_number,
        )

        if not isinstance(
            additions,
            Mapping,
        ):
            additions = {}

        merged = {
            day: [
                copy.deepcopy(
                    activity
                )
                for activity
                in activities
            ]
            for (
                day,
                activities,
            ) in current.items()
        }

        for day in short:
            raw_additions = (
                additions.get(day)
                or []
            )

            if (
                isinstance(
                    raw_additions,
                    Sequence,
                )
                and not isinstance(
                    raw_additions,
                    (str, bytes),
                )
            ):
                merged[day].extend(
                    copy.deepcopy(
                        dict(activity)
                    )
                    for activity
                    in raw_additions
                    if isinstance(
                        activity,
                        Mapping,
                    )
                )

        current, removed = (
            deduplicate_activity_buckets(
                merged
            )
        )

        removed_duplicates += (
            removed
        )

        for raw_day in expected_days:
            current.setdefault(
                int(raw_day),
                [],
            )

        short = short_activity_days(expected_days, current, targets)

    missing = missing_activity_days(expected_days, current)

    return ActivityRescueResult(
        by_day=current,
        missing_days=missing,
        rounds_used=rounds_used,
        removed_duplicates=(
            removed_duplicates
        ),
        short_days=tuple(day for day in short if day not in missing),
    )


__all__ = [
    "ActivityRescueResult",
    "deduplicate_activity_buckets",
    "missing_activity_days",
    "rescue_activity_buckets",
    "short_activity_days",
]