"""Translate orchestration state into the itinerary domain contract."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
import math
import re
from typing import Any

from app.modules.itinerary.models import (
    ExpectedDay,
    TripRequirements,
)
from app.orchestration.state import AgentState
from app.shared.countries import country_alpha2


_CURRENCY_CODE = re.compile(
    r"^[A-Z]{3}$"
)


def build_trip_requirements(
    state: AgentState,
) -> TripRequirements:
    """Create an immutable itinerary contract from server-owned state."""

    origin_country = _required_text(
        state.origin_country,
        "origin_country",
    )

    destination_country = _required_text(
        state.country,
        "destination country",
    )

    destination_country_code = _country_code(
        destination_country
    )

    base_currency = _currency_code(
        state.base_currency_code,
        "base currency",
    )

    destination_currency = _currency_code(
        state.dest_currency_code,
        "destination currency",
    )

    try:
        start = datetime.strptime(
            _required_text(
                state.start_date,
                "start date",
            ),
            "%Y-%m-%d",
        )

        end = datetime.strptime(
            _required_text(
                state.end_date,
                "end date",
            ),
            "%Y-%m-%d",
        )

    except ValueError as exc:
        raise ValueError(
            "trip dates must use "
            "YYYY-MM-DD calendar dates"
        ) from exc

    if end < start:
        raise ValueError(
            "end date must not be before start date"
        )

    if (
        not isinstance(state.num_people, int)
        or isinstance(state.num_people, bool)
    ):
        raise ValueError(
            "num_people is required"
        )

    if state.num_people < 1:
        raise ValueError(
            "num_people must be at least 1"
        )

    total_budget = _finite_number(
        state.total_convert_budget,
        "total converted budget",
    )

    if total_budget < 0:
        raise ValueError(
            "total converted budget "
            "must be non-negative"
        )

    allocation = state.budget_allocation

    if not isinstance(allocation, Mapping):
        raise ValueError(
            "budget allocation must be a mapping"
        )

    cities = state.city

    if (
        not isinstance(cities, list)
        or not all(
            isinstance(city, str)
            and city.strip()
            for city in cities
        )
    ):
        raise ValueError(
            "cities must be a list "
            "of non-empty strings"
        )

    expected_days: list[ExpectedDay] = []

    current = start
    day_number = 1

    while current <= end:
        expected_days.append(
            ExpectedDay(
                day=day_number,
                date=current.strftime(
                    "%Y-%m-%d"
                ),
            )
        )

        current += timedelta(days=1)
        day_number += 1

    return TripRequirements(
        origin_country=origin_country,
        destination_country=destination_country,
        destination_country_code=(
            destination_country_code
        ),
        cities=tuple(
            city.strip()
            for city in cities
        ),
        expected_days=tuple(
            expected_days
        ),
        num_people=state.num_people,
        base_currency=base_currency,
        destination_currency=(
            destination_currency
        ),
        total_budget=total_budget,
        budget_allocation=dict(
            allocation
        ),
    )


def _required_text(
    value: Any,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{label} is required"
        )

    return value.strip()


def _country_code(
    destination_country: str,
) -> str:
    code = country_alpha2(
        destination_country
    )

    if code is None:
        raise ValueError(
            "destination country must resolve to "
            "ISO 3166-1 alpha-2"
        )

    return code


def _currency_code(
    value: Any,
    label: str,
) -> str:
    currency = _required_text(
        value,
        label,
    )

    if not _CURRENCY_CODE.fullmatch(
        currency
    ):
        raise ValueError(
            f"{label} must be a "
            "three-letter ISO 4217 code"
        )

    return currency


def _finite_number(
    value: Any,
    label: str,
) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{label} must be a finite number"
        )

    try:
        number = float(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a finite number"
        ) from exc

    if not math.isfinite(number):
        raise ValueError(
            f"{label} must be a finite number"
        )

    return number


__all__ = [
    "build_trip_requirements",
]