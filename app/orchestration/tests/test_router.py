"""Graph routing: pure reads of the state."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from app.orchestration import router
from app.orchestration.nodes.factual import ROUTER_UNAVAILABLE
from app.orchestration.tests.node_fakes import candidate, committed_state, tool_call, trip_state


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"budget_gate_outcome": "accepted"}, "resume_accepted_budget"),
        ({"accepted_budget_decision": {"stage": "currency"}}, "resume_accepted_budget"),
        ({"chat_budget_action": "accept_recommended"}, "assess_chat_budget"),
        ({"pending_budget_proposal": {"mode": "recommendation"}}, "assess_chat_budget"),
        ({"pending_budget_confirmation": {"budget_assessment_id": "a"}}, "budget_decision_agent"),
        # The old graph locked the chat into budget-only mode here.
        ({"budget_gate_outcome": "budget_check_unavailable"}, "begin_chat_turn"),
        ({}, "begin_chat_turn"),
    ],
)
def test_start_route_for_a_committed_trip(overrides, expected):
    assert router.route_start(committed_state(**overrides)) == expected


def test_a_trip_with_neither_plan_nor_currency_is_new():
    assert router.route_start(trip_state(base_currency_code=None)) == "process_initial_form"


def test_a_failed_classification_goes_to_the_fail_closed_answer():
    state = committed_state(factual_issue_codes=[ROUTER_UNAVAILABLE])

    assert router.route_after_factual_router(state) == "factual_answer"
    assert router.route_after_factual_router(committed_state()) == "agent"


def test_factual_routes_map_to_node_names():
    route = {"request_type": "factual", "domain": "weather", "factual_question": "rain?", "requires_external_evidence": True}
    state = committed_state(factual_route=route)

    assert router.route_after_factual_router(state) == "factual_retrieval"
    # Retrieval never validated: never the general agent.
    assert router.route_after_factual_retrieval(state) == "factual_answer"


@pytest.mark.parametrize(("outcome", "node"), [*router.POST_TOOL_ROUTES.items(), (None, "agent")])
def test_post_tool_outcomes_map_to_nodes(outcome, node):
    assert router.route_after_post_tool(committed_state(post_tool_outcome=outcome)) == node


def test_only_an_acceptance_with_financials_may_plan():
    assert router.route_after_budget(committed_state(budget_gate_outcome="accepted")) == "budget_gate_response"
    assert (
        router.route_after_budget(committed_state(budget_gate_outcome="accepted", plan_overrides={"total_base_budget": 1}))
        == "execute_validated_transaction"
    )


def test_review_and_agent_routes():
    assert router.route_after_review(committed_state(candidate_plan=candidate())) == "promote_candidate"
    assert router.route_after_review(committed_state()) == END
    assert router.route_after_agent(committed_state(HumanMessage(content="x"), tool_call("search_places"))) == "tools"
    assert router.route_after_agent(committed_state(AIMessage(content="hi"))) == END
    assert router.route_after_budget_decision(committed_state(tool_call("confirm_recommended_budget"))) == "budget_decision_tools"
