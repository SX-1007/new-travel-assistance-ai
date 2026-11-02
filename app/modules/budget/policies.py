"""Pure budget adjustment policies."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from app.modules.budget.models import (
    BudgetAdjustment,
    BudgetAdjustmentModifier,
    BudgetCategory,
)


# Share of the trip total given to each category on a fresh plan. The
# assessment minimum is also derived from these ratios: the return flights
# the plan books must fit in "transportation" and the cheapest hotel in
# "accommodation". Changing a ratio changes both, so bump
# assessment.CALCULATION_VERSION when you do.
DEFAULT_BUDGET_RATIOS: Final[
    Mapping[BudgetCategory, float]
] = MappingProxyType(
    {
        "transportation": 0.25,
        "accommodation": 0.35,
        "food": 0.15,
        "activity": 0.15,
        "shopping": 0.05,
        "emergency_fund": 0.05,
    }
)


_MODIFIER_FACTORS: Final[
    dict[
        BudgetAdjustmentModifier,
        float,
    ]
] = {
    "small_increase": 0.10,
    "large_increase": 0.30,
    "small_decrease": -0.10,
    "large_decrease": -0.30,
}


def adjustment_delta(
    adjustment: BudgetAdjustment,
    *,
    current_value: float,
    total_budget: float,
) -> float:
    """
    Resolve one explicit or vague adjustment
    into an absolute delta.
    """

    if (
        adjustment.add_delete_amount
        is not None
    ):
        return float(
            adjustment.add_delete_amount
        )

    modifier = adjustment.modifier

    if modifier is None:
        raise ValueError(
            "budget adjustment is missing "
            "a change value"
        )

    reference_value = (
        current_value
        if current_value > 0
        else 0.05 * total_budget
    )

    return (
        _MODIFIER_FACTORS[modifier]
        * reference_value
    )


__all__ = [
    "DEFAULT_BUDGET_RATIOS",
    "adjustment_delta",
]