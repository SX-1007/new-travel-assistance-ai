"""Pure budget-allocation transformations.

No orchestration state, LangGraph tools or provider
implementations are allowed in this module.
"""

from __future__ import annotations

import math
from collections.abc import (
    Mapping,
    Sequence,
)
from typing import Final, get_args

from app.modules.budget.models import (
    BudgetAdjustment,
    BudgetAllocationResult,
    BudgetCategory,
)
from app.modules.budget.policies import (
    DEFAULT_BUDGET_RATIOS,
    adjustment_delta,
)


# Derived from the BudgetCategory Literal so the category list has exactly one
# definition across budget, itinerary and API code.
BUDGET_CATEGORIES: Final[
    tuple[BudgetCategory, ...]
] = get_args(BudgetCategory)

_BUDGET_CATEGORY_SET = frozenset(
    BUDGET_CATEGORIES
)


def empty_budget_allocation(
) -> dict[str, float]:
    """Return a fresh canonical zero allocation."""

    return {
        category: 0.0
        for category
        in BUDGET_CATEGORIES
    }


def initial_budget_allocation(
    total_budget: float,
) -> dict[str, float]:
    """
    Split a new trip total across the canonical
    categories using DEFAULT_BUDGET_RATIOS.

    Cent rounding is reconciled so the parts
    always add up to the rounded total.
    """

    _require_non_negative_finite(
        total_budget,
        field_name="total_budget",
    )

    allocation = {
        category: round(
            float(total_budget)
            * DEFAULT_BUDGET_RATIOS[category],
            2,
        )
        for category in BUDGET_CATEGORIES
    }

    return _reconcile_rounding(
        allocation,
        target_total=round(
            float(total_budget),
            2,
        ),
    )


def normalise_budget_allocation(
    allocation: Mapping[
        str,
        float,
    ] | None,
) -> dict[str, float]:
    """
    Return a validated allocation containing
    exactly canonical categories.
    """

    result = (
        empty_budget_allocation()
    )

    if not allocation:
        return result

    unexpected = (
        set(allocation)
        - _BUDGET_CATEGORY_SET
    )

    if unexpected:
        raise ValueError(
            "unsupported budget categories: "
            + ", ".join(
                sorted(unexpected)
            )
        )

    for (
        category,
        raw_amount,
    ) in allocation.items():

        if (
            isinstance(
                raw_amount,
                bool,
            )
            or not isinstance(
                raw_amount,
                (int, float),
            )
        ):
            raise ValueError(
                "budget allocation for "
                f"{category} must be finite"
            )

        amount = float(
            raw_amount
        )

        if (
            not math.isfinite(amount)
            or amount < 0
        ):
            raise ValueError(
                "budget allocation for "
                f"{category} must be "
                "non-negative and finite"
            )

        result[category] = round(
            amount,
            2,
        )

    return result


def rescale_budget_allocation(
    allocation: Mapping[
        str,
        float,
    ] | None,
    *,
    new_total_budget: float,
) -> dict[str, float]:
    """
    Rescale an existing allocation
    proportionally to a new positive total.
    """

    _require_positive_finite(
        new_total_budget,
        field_name="new_total_budget",
    )

    current = (
        normalise_budget_allocation(
            allocation
        )
    )

    allocated_total = sum(
        current.values()
    )

    if allocated_total <= 0:
        return current

    scale = (
        float(new_total_budget)
        / allocated_total
    )

    scaled = {
        category: round(
            amount * scale,
            2,
        )
        for category, amount
        in current.items()
    }

    return _reconcile_rounding(
        scaled,
        target_total=round(
            float(new_total_budget),
            2,
        ),
    )


def adjust_budget_allocation(
    allocation: Mapping[
        str,
        float,
    ] | None,
    *,
    total_budget: float,
    adjustments: Sequence[
        BudgetAdjustment
    ],
) -> BudgetAllocationResult:
    """
    Apply category edits and proportionally
    rebalance non-edited categories.
    """

    _require_non_negative_finite(
        total_budget,
        field_name="total_budget",
    )

    current = (
        normalise_budget_allocation(
            allocation
        )
    )

    if not adjustments:
        raise ValueError(
            "at least one budget adjustment "
            "is required"
        )

    explicit_categories: set[str] = (
        set()
    )

    net_added = 0.0

    for adjustment in adjustments:
        category = adjustment.category

        explicit_categories.add(
            category
        )

        old_value = current[
            category
        ]

        delta = adjustment_delta(
            adjustment,
            current_value=old_value,
            total_budget=float(
                total_budget
            ),
        )

        new_value = max(
            0.0,
            round(
                old_value + delta,
                2,
            ),
        )

        current[
            category
        ] = new_value

        net_added += (
            new_value
            - old_value
        )

    donor_categories = [
        category
        for category
        in BUDGET_CATEGORIES
        if category
        not in explicit_categories
    ]

    donor_total = sum(
        current[category]
        for category
        in donor_categories
    )

    if (
        abs(net_added) > 1e-9
        and donor_total > 0
    ):
        original_donors = {
            category: current[
                category
            ]
            for category
            in donor_categories
        }

        target_donor_total = max(
            0.0,
            round(
                donor_total
                - net_added,
                2,
            ),
        )

        if target_donor_total == 0:
            for category in (
                donor_categories
            ):
                current[
                    category
                ] = 0.0

        else:
            donor_scale = (
                target_donor_total
                / donor_total
            )

            for category in (
                donor_categories
            ):
                current[
                    category
                ] = round(
                    original_donors[
                        category
                    ]
                    * donor_scale,
                    2,
                )

            _reconcile_subset_rounding(
                current,
                donor_categories,
                target_total=(
                    target_donor_total
                ),
            )

    warnings: list[str] = []

    allowed_total = round(
        float(total_budget),
        2,
    )

    actual_total = round(
        sum(
            current.values()
        ),
        2,
    )

    if actual_total > allowed_total:
        warnings.append(
            f"Adjusted total "
            f"{actual_total:.2f} "
            f"exceeded budget "
            f"{allowed_total:.2f}; "
            "categories were "
            "proportionally trimmed "
            "to fit."
        )

        scale = (
            allowed_total
            / actual_total
        )

        current = {
            category: round(
                amount * scale,
                2,
            )
            for category, amount
            in current.items()
        }

        current = (
            _reconcile_rounding(
                current,
                target_total=(
                    allowed_total
                ),
            )
        )

    return BudgetAllocationResult(
        budget_allocation=current,
        warnings=tuple(
            warnings
        ),
    )


def _reconcile_subset_rounding(
    allocation: dict[
        str,
        float,
    ],
    categories: Sequence[str],
    *,
    target_total: float,
) -> None:
    """
    Correct cent-level rounding drift
    within selected categories.
    """

    difference = round(
        target_total
        - sum(
            allocation[category]
            for category
            in categories
        ),
        2,
    )

    if abs(difference) < 0.01:
        return

    ordered = sorted(
        categories,
        key=lambda category: (
            allocation[category]
        ),
        reverse=True,
    )

    for category in ordered:
        candidate = round(
            allocation[category]
            + difference,
            2,
        )

        if candidate >= 0:
            allocation[
                category
            ] = candidate

            return


def _reconcile_rounding(
    allocation: dict[
        str,
        float,
    ],
    *,
    target_total: float,
) -> dict[str, float]:
    """
    Correct cent-level rounding drift without
    materially changing proportions.
    """

    difference = round(
        target_total
        - sum(
            allocation.values()
        ),
        2,
    )

    if abs(difference) < 0.01:
        return allocation

    ordered = sorted(
        BUDGET_CATEGORIES,
        key=lambda category: (
            allocation[category]
        ),
        reverse=True,
    )

    for category in ordered:
        candidate = round(
            allocation[category]
            + difference,
            2,
        )

        if candidate >= 0:
            allocation[
                category
            ] = candidate

            break

    return allocation


def _require_positive_finite(
    value: float,
    *,
    field_name: str,
) -> None:
    _require_non_negative_finite(
        value,
        field_name=field_name,
    )

    if float(value) <= 0:
        raise ValueError(
            f"{field_name} must be "
            "greater than 0"
        )


def _require_non_negative_finite(
    value: float,
    *,
    field_name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise ValueError(
            f"{field_name} must be "
            "a finite number"
        )

    if (
        not math.isfinite(
            float(value)
        )
        or float(value) < 0
    ):
        raise ValueError(
            f"{field_name} must be "
            "non-negative and finite"
        )


__all__ = [
    "BUDGET_CATEGORIES",
    "empty_budget_allocation",
    "initial_budget_allocation",
    "normalise_budget_allocation",
    "rescale_budget_allocation",
    "adjust_budget_allocation",
]