"""LangGraph node functions, grouped by concern.

Each node is ``async (state, config, *, <services>) -> dict`` and returns
only the fields it updates. ``build_nodes`` binds the services once (7c
passes what ``bootstrap/container.py`` builds) and returns the nodes by
graph name, each with the plain ``(state, config)`` signature LangGraph
expects. Nothing here constructs a client or reads settings.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.modules.factual_qa.service import FactualQAService
from app.orchestration.nodes.agent import (
    AgentModels,
    AgentToolset,
    agent_node,
    budget_decision_agent_node,
    build_agent_toolset,
)
from app.orchestration.nodes.budget import (
    assess_chat_budget_node,
    budget_gate_response_node,
    resume_accepted_budget_node,
)
from app.orchestration.nodes.context import ProfileLoader
from app.orchestration.nodes.factual import (
    factual_answer_node,
    factual_retrieval_node,
    factual_router_node,
)
from app.orchestration.nodes.planning import (
    PlanningServices,
    execute_validated_transaction_node,
    generate_reviewed_response_node,
    prepare_planning_transaction_node,
    process_initial_form_node,
    promote_candidate_node,
)
from app.orchestration.nodes.post_tool import EditServices, post_tool_processing_node
from app.orchestration.nodes.replies import ReplyServices
from app.orchestration.nodes.turn import MemoryExtractionQueue, begin_chat_turn_node
from app.orchestration.state import AgentState


Node = Callable[[AgentState, RunnableConfig], Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class GraphServices:
    """Everything the nodes call out to, built by the composition root."""

    planning: PlanningServices
    replies: ReplyServices
    models: AgentModels
    edits: EditServices
    profiles: ProfileLoader
    factual: FactualQAService
    memory: MemoryExtractionQueue | None = None


def build_nodes(services: GraphServices) -> dict[str, Callable[..., Any]]:
    """Graph name → node with its services bound."""

    reviewer = services.replies.reviewer

    return {
        "process_initial_form": _bind(process_initial_form_node, planning=services.planning),
        "prepare_planning_transaction": prepare_planning_transaction_node,
        "execute_validated_transaction": _bind(
            execute_validated_transaction_node,
            planning=services.planning,
            profiles=services.profiles,
        ),
        "generate_reviewed_response": _bind(
            generate_reviewed_response_node,
            replies=services.replies,
            profiles=services.profiles,
        ),
        "promote_candidate": _bind(promote_candidate_node, reviewer=reviewer),
        "begin_chat_turn": _bind(begin_chat_turn_node, memory=services.memory),
        "factual_router": _bind(factual_router_node, factual=services.factual),
        "factual_retrieval": _bind(factual_retrieval_node, factual=services.factual),
        "factual_answer": _bind(factual_answer_node, factual=services.factual),
        "agent": _bind(
            agent_node,
            models=services.models,
            replies=services.replies,
            profiles=services.profiles,
            factual=services.factual,
        ),
        "budget_decision_agent": _bind(
            budget_decision_agent_node,
            models=services.models,
            replies=services.replies,
        ),
        "assess_chat_budget": _bind(assess_chat_budget_node, budget=services.planning.budget),
        "resume_accepted_budget": _bind(resume_accepted_budget_node, budget=services.planning.budget),
        "budget_gate_response": _bind(budget_gate_response_node, reviewer=reviewer),
        "post_tool_processing": _bind(post_tool_processing_node, edits=services.edits),
    }


def _bind(node: Callable[..., Awaitable[dict[str, Any]]], **services: Any) -> Node:
    async def bound(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        return await node(state, config, **services)

    bound.__name__ = node.__name__
    bound.__qualname__ = node.__qualname__
    bound.__doc__ = node.__doc__
    return bound


__all__ = [
    "AgentModels",
    "AgentToolset",
    "EditServices",
    "GraphServices",
    "MemoryExtractionQueue",
    "PlanningServices",
    "ProfileLoader",
    "ReplyServices",
    "build_agent_toolset",
    "build_nodes",
]
