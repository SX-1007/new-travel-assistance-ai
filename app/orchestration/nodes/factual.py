"""Factual Q&A nodes: thin graph boundaries over ``workflows/factual_qa``.

The service already falls back on classifier, search and writer failures;
these nodes only keep an unexpected error from failing the turn. A routing
failure is marked ``fact.router_unavailable`` so the router sends the turn
to the fail-closed answer instead of the unrestricted agent.
"""

from __future__ import annotations

import logging
from typing import Any, Final

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from app.modules.factual_qa.service import FactualQAService
from app.orchestration.nodes.context import node_log_context
from app.orchestration.nodes.replies import SERVER_OWNED_KEY
from app.orchestration.state import AgentState
from app.orchestration.workflows.factual_qa import (
    empty_factual_state,
    run_factual_answer,
    run_factual_retrieval,
    run_factual_routing,
)
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)

ROUTER_UNAVAILABLE: Final = "fact.router_unavailable"
_RETRIEVAL_UNAVAILABLE: Final = "fact.retrieval_unavailable"
_ANSWER_UNAVAILABLE: Final = "fact.answer_unavailable"
_UNAVAILABLE_REPLY: Final = "I couldn't verify that right now, so I won't guess. Please try again shortly."


async def factual_router_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    factual: FactualQAService,
) -> dict[str, Any]:
    log_context = node_log_context(config)

    try:
        return await run_factual_routing(state, service=factual, log_context=log_context)
    except Exception:
        _log_failure(log_context, "factual_router", ROUTER_UNAVAILABLE)
        return {**empty_factual_state(), "factual_issue_codes": [ROUTER_UNAVAILABLE]}


async def factual_retrieval_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    factual: FactualQAService,
) -> dict[str, Any]:
    log_context = node_log_context(config)

    try:
        return await run_factual_retrieval(state, service=factual, log_context=log_context)
    except Exception:
        _log_failure(log_context, "factual_retrieval", _RETRIEVAL_UNAVAILABLE)
        return {
            "factual_evidence": [],
            "factual_validation": {"approved": False, "issue_codes": [_RETRIEVAL_UNAVAILABLE]},
            "factual_issue_codes": [_RETRIEVAL_UNAVAILABLE],
        }


async def factual_answer_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    factual: FactualQAService,
) -> dict[str, Any]:
    log_context = node_log_context(config)

    try:
        return await run_factual_answer(state, service=factual, log_context=log_context)
    except Exception:
        _log_failure(log_context, "factual_answer", _ANSWER_UNAVAILABLE)
        return {
            "messages": [
                AIMessage(
                    content=_UNAVAILABLE_REPLY,
                    additional_kwargs={SERVER_OWNED_KEY: "factual_qa_unavailable"},
                )
            ],
            "factual_issue_codes": [_ANSWER_UNAVAILABLE],
        }


def _log_failure(log_context: LogContext, stage: str, code: str) -> None:
    log_event(logger, "factual.node", context=log_context, stage=stage, outcome="unavailable", issue_codes=[code])


__all__ = [
    "ROUTER_UNAVAILABLE",
    "factual_answer_node",
    "factual_retrieval_node",
    "factual_router_node",
]
