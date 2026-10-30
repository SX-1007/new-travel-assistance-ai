"""Chat budget proposal tools and the proposal builders behind them."""

import math

import pytest

from app.modules.budget.chat_gate import (
    budget_change_proposal,
    confirm_recommendation_proposal,
)
from langchain_core.messages import HumanMessage

from app.modules.budget.chat_gate import amount_is_stated, stated_amounts
from app.modules.budget.currency import home_currency_amount
from app.orchestration.state import AgentState
from app.orchestration.tools.budget import (
    confirm_recommended_budget,
    propose_budget_change,
)

TRIP = AgentState(
    base_currency_code="MYR",
    dest_currency_code="JPY",
    exchange_rate={"MYR": 1.0, "JPY": 38.6438890572},
    total_base_budget=9000.0,
)


def propose(said="Please change my total budget to 5000 MYR.", **args):
    state = TRIP.model_copy(update={"messages": [HumanMessage(content=said)]})
    return propose_budget_change.invoke({**args, "state": state})


def test_amount_proposal():
    assert budget_change_proposal(1500) == {"mode": "amount", "total_base_budget": 1500.0}


def test_recommendation_proposal():
    assert budget_change_proposal(request_recommendation=True) == {
        "mode": "recommendation",
        "total_base_budget": None,
    }


@pytest.mark.parametrize("amount", [0, -5, math.inf, math.nan, True, "abc"])
def test_invalid_amounts_are_rejected(amount):
    with pytest.raises(ValueError, match="finite and greater than 0"):
        budget_change_proposal(amount)


def test_missing_amount_is_rejected():
    with pytest.raises(ValueError, match="Provide a positive budget"):
        budget_change_proposal(None)


def test_recommendation_with_amount_is_rejected():
    with pytest.raises(ValueError, match="cannot include an amount"):
        budget_change_proposal(100, request_recommendation=True)


def test_confirm_proposal():
    assert confirm_recommendation_proposal() == {"mode": "confirm", "total_base_budget": None}


def test_propose_tool_wraps_proposal():
    assert propose("make it RM2,000", amount=2000, currency="myr") == {
        "status": "budget_proposal",
        "proposal": {"mode": "amount", "total_base_budget": 2000.0},
        "home_currency_amount": "MYR 2,000.00",
    }


def test_a_destination_currency_amount_is_converted_by_the_server():
    # Found live: the model converted "5000 MYR" to JPY 193,219.45 and passed
    # that as a home-currency total (the budget became MYR 193,219).
    result = propose(amount=193219.45, currency="JPY")

    assert result["proposal"] == {"mode": "amount", "total_base_budget": 5000.0}


def test_the_currency_is_required_and_must_be_the_trips():
    assert "error" in propose(amount=5000)
    assert "chat_currency_conversion" in propose(amount=5000, currency="USD")["error"]
    assert "error" in propose(currency="MYR")


def test_recommendations_take_no_amount():
    assert propose(request_recommendation=True)["proposal"] == {"mode": "recommendation", "total_base_budget": None}
    assert "error" in propose(amount=10, currency="MYR", request_recommendation=True)


def test_a_total_the_user_never_stated_is_refused():
    # Found live: the model reused 193,219.45 from its own earlier call
    # although the user wrote 5000.
    assert "Ask the user" in propose(amount=193219.45, currency="MYR")["error"]
    # A change relative to the current total is the user's number too.
    assert propose("add 1000 to my budget", amount=10000, currency="MYR")["proposal"]["total_base_budget"] == 10000.0
    assert "error" in propose("double my budget", amount=18000, currency="MYR")


def test_stated_amounts_and_matching():
    assert stated_amounts("RM5,000.50 or maybe 5.5k, not 1.2m") == (5000.5, 5500.0, 1_200_000.0)
    assert amount_is_stated(5000.0, [5000], rate=38.6)
    assert amount_is_stated(5000.0, [193_000], rate=38.6)  # stated in JPY
    assert amount_is_stated(10_000.0, [1000], rate=38.6, current_home_total=9000.0)
    assert not amount_is_stated(193_219.45, [5000], rate=38.6438890572)


def test_propose_tool_returns_error_instead_of_raising():
    result = propose("make it -1", amount=-1, currency="MYR")
    assert "error" in result


def test_home_currency_amount():
    assert home_currency_amount(500, " myr ", home="MYR", destination="JPY", rate=40.0) == 500.0
    assert home_currency_amount(20000, "JPY", home="MYR", destination="JPY", rate=40.0) == 500.0
    with pytest.raises(ValueError):
        home_currency_amount(20000, "JPY", home="MYR", destination="JPY", rate=0.0)


def test_confirm_tool():
    assert confirm_recommended_budget.invoke({}) == {
        "status": "budget_proposal",
        "proposal": {"mode": "confirm", "total_base_budget": None},
    }
