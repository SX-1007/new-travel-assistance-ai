"""AgentState adapter for currency preprocessing."""

from __future__ import annotations

from collections.abc import Callable
from datetime import (
    datetime,
    timezone,
)
from typing import Any

from app.modules.budget.currency import (
    convert_amount,
)
from app.modules.budget.ports import (
    CurrencyProviderPort,
)
from app.orchestration.state import (
    AgentState,
)


def run_currency_pipeline(
    state: AgentState,
    *,
    currency_provider: (
        CurrencyProviderPort
    ),
) -> dict[str, Any]:
    """
    Resolve state currencies and return only
    the state fields to update.
    """

    if not state.origin_country:
        raise ValueError(
            "origin_country is required "
            "before currency conversion"
        )

    if not state.country:
        raise ValueError(
            "country is required before "
            "currency conversion"
        )

    base_currency = (
        currency_provider
        .get_currency_code(
            state.origin_country
        )
    )

    destination_currency = (
        currency_provider
        .get_currency_code(
            state.country
        )
    )

    rate = (
        currency_provider
        .get_currency_rate(
            base_currency,
            destination_currency,
        )
    )

    converted_budget = (
        convert_amount(
            state.total_base_budget
            or 0.0,
            rate,
        )
    )

    return {
        "base_currency_code":
            base_currency,

        "dest_currency_code":
            destination_currency,

        "exchange_rate": {
            base_currency: 1.0,
            destination_currency: rate,
        },

        "total_convert_budget":
            converted_budget,

        "currency_fetched_at": (
            datetime.now(
                timezone.utc
            )
            .isoformat()
            .replace(
                "+00:00",
                "Z",
            )
        ),
    }


def build_currency_pipeline(
    currency_provider: (
        CurrencyProviderPort
    ),
) -> Callable[
    [AgentState],
    dict[str, Any],
]:
    """
    Bind a concrete provider once at the
    composition boundary.
    """

    def currency_pipeline(
        state: AgentState,
    ) -> dict[str, Any]:
        return run_currency_pipeline(
            state,
            currency_provider=(
                currency_provider
            ),
        )

    return currency_pipeline


__all__ = [
    "run_currency_pipeline",
    "build_currency_pipeline",
]