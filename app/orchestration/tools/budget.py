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
from app.orchestration.pending_trip import (
    with_pending_trip,
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

    # Sessions saved by earlier versions may also hold "flight", a duplicate
    # of transportation.
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

    # The model passes the amount and currency exactly as the user said them,
    # and the server converts. Left to convert itself, the model once turned
    # "5000 MYR" into JPY 193,219 and passed that as the home-currency total.
    # A trip change waiting on a recommendation brings its own currencies.
    state = with_pending_trip(state)
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

            # The total must come from the user's latest message: the model
            # has been seen reusing a wrong number from its own earlier call.
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


@tool
def keep_current_plan() -> dict[str, Any]:
    """Decline the pending recommendation: keep the current trip and budget
    exactly as they are, and drop any trip change waiting on it."""

    return {
        "status": "budget_proposal",
        "proposal": {
            "mode": "decline",
            "total_base_budget": None,
        },
    }


__all__ = [
    "update_total_budget",
    "update_budget_category",
    "propose_budget_change",
    "confirm_recommended_budget",
    "keep_current_plan",
]