"""Thin LangGraph wrappers around pure budget operations."""

from __future__ import annotations

from typing import (
    Annotated,
    Any,
    Optional,
)

from langchain_core.tools import tool
from langgraph.prebuilt import (
    InjectedState,
)

from app.modules.budget.allocation import (
    adjust_budget_allocation,
    rescale_budget_allocation,
)
from app.modules.budget.chat_gate import (
    amount_is_stated,
    budget_change_proposal,
    confirm_recommendation_proposal,
    stated_amounts,
)
from app.modules.budget.currency import (
    home_currency_amount,
)
from app.modules.budget.models import (
    BudgetAdjustment,
)
from app.orchestration.messages import (
    latest_user_request,
)
from app.orchestration.state import (
    AgentState,
)


def _state_budget_allocation(
    state: AgentState,
) -> dict[str, float]:
    allocation = dict(
        state.budget_allocation
        or {}
    )

    # Compatibility with old checkpoints only.
    # "flight" duplicated transportation.
    allocation.pop(
        "flight",
        None,
    )

    return allocation


@tool
def update_total_budget(
    new_total_budget: float,
    state: Annotated[
        AgentState,
        InjectedState,
    ],
) -> dict[str, Any]:
    """
    Update the trip total and rescale existing
    categories proportionally.
    """

    try:
        allocation = (
            rescale_budget_allocation(
                _state_budget_allocation(
                    state
                ),
                new_total_budget=(
                    new_total_budget
                ),
            )
        )

    except ValueError as exc:
        return {
            "error": str(exc)
        }

    return {
        "total_convert_budget": round(
            float(
                new_total_budget
            ),
            2,
        ),
        "budget_allocation":
            allocation,
    }


@tool
def update_budget_category(
    adjustment: list[
        BudgetAdjustment
    ],
    state: Annotated[
        AgentState,
        InjectedState,
    ],
) -> dict[str, Any]:
    """
    Adjust categories and rebalance the
    remaining categories.
    """

    try:
        result = (
            adjust_budget_allocation(
                _state_budget_allocation(
                    state
                ),
                total_budget=(
                    state.total_convert_budget
                    or 0.0
                ),
                adjustments=(
                    adjustment
                ),
            )
        )

    except ValueError as exc:
        return {
            "error": str(exc)
        }

    payload: dict[str, Any] = {
        "budget_allocation":
            result.budget_allocation,
    }

    if result.warnings:
        payload["warnings"] = list(
            result.warnings
        )

    return payload


@tool
def propose_budget_change(
    state: Annotated[
        AgentState,
        InjectedState,
    ],
    amount: Optional[float] = None,
    currency: Optional[str] = None,
    request_recommendation: bool = False,
) -> dict[str, Any]:
    """
    Propose a new trip total exactly as the user stated it: the amount and
    its currency code (home or destination currency). Never convert it
    yourself; the server does. With no amount and
    request_recommendation=true, ask for the provider-grounded minimum.
    Changes nothing until the budget check accepts it.
    """

    # The old tool took a bare home-currency number. Found live: the model
    # converted "5000 MYR" to JPY 193,219 and passed that, so the budget
    # became MYR 193,219. Units are now explicit and converted here, and
    # the amount must be one the user actually wrote.
    home = state.base_currency_code or ""
    destination = state.dest_currency_code or ""

    try:
        if request_recommendation:
            proposal = budget_change_proposal(
                amount,
                request_recommendation=True,
            )

        else:
            if amount is None:
                raise ValueError(
                    "Provide the amount and its currency, or request a recommendation."
                )

            rates = state.exchange_rate or {}
            rate = (
                rates.get(destination, 0.0)
                / (rates.get(home) or 1.0)
            )
            home_amount = home_currency_amount(
                amount,
                currency or "",
                home=home,
                destination=destination,
                rate=rate,
            )

            # Found live, after the fix above: the model reused a number
            # from its own earlier (wrong) call instead of the user's.
            # A total must come from the user's latest message.
            if not amount_is_stated(
                home_amount,
                stated_amounts(
                    latest_user_request(state.messages)
                ),
                rate=rate,
                current_home_total=state.total_base_budget,
            ):
                raise ValueError(
                    "This total does not match any amount in the user's "
                    "latest message. Ask the user for the exact new total "
                    "instead of proposing one."
                )

            proposal = budget_change_proposal(home_amount)

    except ValueError as exc:
        return {
            "error": str(exc)
        }

    result: dict[str, Any] = {
        "status": "budget_proposal",
        "proposal": proposal,
    }

    if proposal["total_base_budget"] is not None:
        # Lets the reply state the amount the check will judge.
        result["home_currency_amount"] = (
            f"{home} {proposal['total_base_budget']:,.2f}"
        )

    return result


@tool
def confirm_recommended_budget() -> dict[str, Any]:
    """Explicitly accept the exact server-owned pending recommendation."""

    return {
        "status": "budget_proposal",
        "proposal":
            confirm_recommendation_proposal(),
    }


__all__ = [
    "update_total_budget",
    "update_budget_category",
    "propose_budget_change",
    "confirm_recommended_budget",
]