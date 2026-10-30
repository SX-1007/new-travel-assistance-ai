"""Where the graph goes next: pure functions of the state, no I/O.

The nodes record their decisions in state (``post_tool_outcome``,
``budget_gate_outcome``, ``candidate_plan``), so routing only reads them.
The old routers re-derived decisions instead: ``route_after_post_tool``
re-parsed the tool messages and re-validated trip changes against the
already-updated state, and two routers looped back into planning for retries
the planning workflow now runs itself.
"""

from __future__ import annotations

from typing import Final, Literal

from langgraph.graph import END

from app.orchestration.nodes.budget import budget_decision_pending
from app.orchestration.nodes.factual import ROUTER_UNAVAILABLE
from app.orchestration.state import AgentState
from app.orchestration.workflows import factual_qa


StartRoute = Literal[
    "resume_accepted_budget",
    "assess_chat_budget",
    "budget_decision_agent",
    "process_initial_form",
    "begin_chat_turn",
]

POST_TOOL_ROUTES: Final[dict[str, str]] = {
    "replan_full": "process_initial_form",
    "replan_itinerary": "prepare_planning_transaction",
    "review_candidate": "generate_reviewed_response",
    "assess_budget": "assess_chat_budget",
    "budget_gate": "budget_gate_response",
    "reply_sent": END,
    "agent": "agent",
}


def route_start(state: AgentState) -> StartRoute:
    """Entry: budget work first, then a new trip, else an ordinary chat turn."""

    # A stored acceptance is never authority on its own; the resume node
    # re-verifies it (or turns a malformed one into a retryable gate reply).
    if state.budget_gate_outcome == "accepted" or state.accepted_budget_decision:
        return "resume_accepted_budget"

    if state.chat_budget_action == "accept_recommended" or state.pending_budget_proposal is not None:
        return "assess_chat_budget"

    if budget_decision_pending(state):
        return "budget_decision_agent"

    if not state.draft_itinerary and not state.base_currency_code:
        return "process_initial_form"

    return "begin_chat_turn"


def route_after_factual_router(
    state: AgentState,
) -> Literal["factual_retrieval", "factual_answer", "agent"]:
    # A failed classification must not hand a possibly live, high-risk
    # question to the unrestricted agent (the workflow router alone would).
    if state.factual_route is None and ROUTER_UNAVAILABLE in state.factual_issue_codes:
        return "factual_answer"

    return {
        "retrieval": "factual_retrieval",
        "answer": "factual_answer",
        "agent": "agent",
    }[factual_qa.route_after_factual_router(state)]


def route_after_factual_retrieval(state: AgentState) -> Literal["factual_answer", "agent"]:
    return "agent" if factual_qa.route_after_factual_retrieval(state) == "agent" else "factual_answer"


def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    return "tools" if _wants_tools(state) else END


def route_after_budget_decision(state: AgentState) -> Literal["budget_decision_tools", "__end__"]:
    return "budget_decision_tools" if _wants_tools(state) else END


def route_after_post_tool(state: AgentState) -> str:
    return POST_TOOL_ROUTES.get(state.post_tool_outcome or "agent", "agent")


def route_after_budget(
    state: AgentState,
) -> Literal["execute_validated_transaction", "budget_gate_response"]:
    """Only an acceptance whose financials were built may enter planning."""

    if state.budget_gate_outcome == "accepted" and state.plan_overrides:
        return "execute_validated_transaction"

    return "budget_gate_response"


def route_after_review(state: AgentState) -> Literal["promote_candidate", "__end__"]:
    return "promote_candidate" if state.candidate_plan is not None else END


def _wants_tools(state: AgentState) -> bool:
    return bool(state.messages) and bool(getattr(state.messages[-1], "tool_calls", None))


__all__ = [
    "POST_TOOL_ROUTES",
    "route_after_agent",
    "route_after_budget",
    "route_after_budget_decision",
    "route_after_factual_retrieval",
    "route_after_factual_router",
    "route_after_post_tool",
    "route_after_review",
    "route_start",
]
