"""The tool-calling agent, its bound tools, and the pending-budget interpreter.

Tool choice (same set as the old graph, now explicit):

* bound: alternatives search, place search, ``edit_itinerary``, nearby
  amenities, currency conversion, ``propose_budget_change``,
  ``update_budget_category``, ``update_trip_details``;
* not bound: ``update_total_budget`` (it would change the total without the
  provider-grounded budget gate; totals go through ``propose_budget_change``)
  and ``modify_existing_booking`` (``edit_itinerary`` covers flights and
  hotels through the same apply path);
* the edit-continuation model gets only ``edit_itinerary``; the budget
  interpreter only ``propose_budget_change`` and ``confirm_recommended_budget``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from app.modules.activities.service import ActivitySearchService
from app.modules.amenities.service import NearbyAmenityService
from app.modules.booking.service import BookingPlanningService
from app.modules.budget.ports import CurrencyProviderPort
from app.modules.factual_qa.service import FactualQAService
from app.orchestration.edit_intent import (
    DAY_CLARIFICATION_MARKER,
    is_explicit_edit_request,
    pending_mutation_instruction,
    pending_mutation_record,
    place_edit_day_clarification,
)
from app.orchestration.messages import (
    current_turn,
    current_turn_used_tool,
    latest_user_request,
    message_text,
)
from app.orchestration.nodes.context import (
    ProfileLoader,
    node_log_context,
    user_id_from_config,
)
from app.orchestration.nodes.replies import (
    AGENT_UNAVAILABLE,
    AGENT_UNAVAILABLE_TEXT,
    BUDGET_DECISION_UNAVAILABLE,
    BUDGET_DECISION_UNAVAILABLE_TEXT,
    MUTATION_MODEL_UNAVAILABLE,
    MUTATION_MODEL_UNAVAILABLE_TEXT,
    MUTATION_TARGET_REQUIRED,
    MUTATION_TARGET_REQUIRED_TEXT,
    REPLY_UNVERIFIED,
    ChatModel,
    ReplyServices,
    server_reply,
    unverified_reply_text,
    write_reply,
)
from app.orchestration.prompts import build_budget_decision_messages, build_prompt_messages
from app.orchestration.state import AgentState
from app.orchestration.tools.booking import build_search_alternative_opt_tool, edit_itinerary
from app.orchestration.tools.budget import (
    confirm_recommended_budget,
    propose_budget_change,
    update_budget_category,
)
from app.orchestration.tools.currency import build_chat_currency_conversion_tool
from app.orchestration.tools.places import (
    build_search_nearby_amenities_tool,
    build_search_places_tool,
)
from app.orchestration.tools.trip import update_trip_details
from app.orchestration.workflows.factual_qa import mixed_intent_message
from app.orchestration.workflows.output_review import (
    generate_reviewed_reply,
    review_context_from_state,
)
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)

# Tool rounds per user turn. After that the agent must answer without tools;
# the old graph ran until LangGraph's recursion limit failed the whole turn.
MAX_TOOL_ROUNDS: Final = 6

_EDIT_CONTINUATION: Final = (
    "Internal mutation continuation: search_places has already completed "
    "successfully in this same user turn. Inspect the most recent search_places "
    "result and the current itinerary. For activity/restaurant remove or replace, "
    "use the server's authorized target-day rules: an explicit day in the current "
    "message, a server-owned day-clarification reply, or an immediate "
    "yes-confirmation where the same day was already explicitly supplied by the "
    "user before the proposal. Never use an assistant-only guessed day. If no "
    "authorized day exists, ask exactly one concise clarification question and do "
    "not call edit_itinerary. Otherwise, call edit_itinerary using the exact "
    "provider-grounded result or results. If the current request contains "
    "multiple edits, include EVERY requested edit in ONE edit_itinerary call; do "
    "not apply only the first change. Do not output JSON, raw tool data, or a "
    "prose success message."
)
_EXECUTE_EDIT: Final = (
    "Internal execution requirement: the current user message asks to continue or "
    "perform a change to the existing trip. Do not claim that anything was changed "
    "in prose unless you execute the appropriate mutation tool. Use edit_itinerary "
    "for itinerary items; for an activity/restaurant add or replacement, call "
    "search_places first in this turn, then edit_itinerary. Honor the "
    "server-authorized continuation rules for an immediate yes-confirmation or a "
    "server-owned day clarification reply; do not re-ask for a day that the server "
    "has already authorized. If essential target information is genuinely missing, "
    "ask one concise clarification question instead."
)


# ── Tools and models ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AgentToolset:
    """Tools per model; 7c builds one ToolNode for ``agent`` (it also serves
    the edit-continuation model's calls) and one for ``budget_decision``."""

    agent: tuple[BaseTool, ...]
    edit: tuple[BaseTool, ...]
    budget_decision: tuple[BaseTool, ...]


def build_agent_toolset(
    *,
    booking: BookingPlanningService,
    place_search: ActivitySearchService,
    amenities: NearbyAmenityService,
    currency: CurrencyProviderPort,
) -> AgentToolset:
    return AgentToolset(
        agent=(
            build_search_alternative_opt_tool(booking),
            build_search_places_tool(place_search),
            edit_itinerary,
            build_search_nearby_amenities_tool(amenities),
            build_chat_currency_conversion_tool(currency),
            propose_budget_change,
            update_budget_category,
            update_trip_details,
        ),
        edit=(edit_itinerary,),
        budget_decision=(propose_budget_change, confirm_recommended_budget),
    )


@dataclass(frozen=True, slots=True)
class AgentModels:
    """The chat model bound to each tool set."""

    agent: ChatModel
    edit: ChatModel
    budget: ChatModel

    @classmethod
    def bind(cls, chat_model: Any, toolset: AgentToolset) -> AgentModels:
        return cls(
            agent=chat_model.bind_tools(list(toolset.agent)),
            edit=chat_model.bind_tools(list(toolset.edit)),
            budget=chat_model.bind_tools(list(toolset.budget_decision)),
        )


# ── Nodes ───────────────────────────────────────────────────────────


async def agent_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    models: AgentModels,
    replies: ReplyServices,
    profiles: ProfileLoader,
    factual: FactualQAService,
) -> dict[str, Any]:
    """One reasoning step: a tool call, or a reviewed public reply."""

    log_context = node_log_context(config)

    # A numbered place edit without a day is answered before any model call:
    # the model must never pick the day of a destructive edit itself.
    clarification = place_edit_day_clarification(state)

    if clarification is not None:
        return {
            "messages": [
                server_reply(
                    state,
                    clarification,
                    marker=DAY_CLARIFICATION_MARKER,
                    reviewer=replies.reviewer,
                    log_context=log_context,
                )
            ],
            "pending_itinerary_mutation": pending_mutation_record(
                (),
                source_request=latest_user_request(state.messages),
            ),
            "output_review_attempts": 0,
            "output_review_issue_codes": [],
        }

    profile = await profiles.load(user_id_from_config(config), log_context=log_context)
    messages = _agent_messages(state, profile, factual, log_context)

    if tool_rounds(state) >= MAX_TOOL_ROUNDS:
        _log(log_context, "agent", "tool_limit", "agent.tool_rounds_exhausted")
        return await _reviewed_reply(state, config, messages, None, replies, profile, log_context)

    try:
        response = await models.agent.ainvoke(messages, config=config)
    except Exception:
        _log(log_context, "agent", "unavailable", "model.unavailable")
        return _server_only(state, AGENT_UNAVAILABLE_TEXT, AGENT_UNAVAILABLE, replies, log_context, "model.unavailable")

    if getattr(response, "tool_calls", None):
        return {"messages": [response]}

    # A direct edit request must not end as prose-only "success": one bounded
    # retry that pushes the model to the mutation tools.
    if is_explicit_edit_request(state):
        searched = current_turn_used_tool(state.messages, "search_places")
        model, instruction = (models.edit, _EDIT_CONTINUATION) if searched else (models.agent, _EXECUTE_EDIT)

        try:
            corrected = await model.ainvoke([*messages, HumanMessage(content=instruction)], config=config)
        except Exception:
            _log(log_context, "agent_mutation_retry", "unavailable", "model.unavailable")
            corrected = None

            if searched:
                return _server_only(
                    state,
                    MUTATION_MODEL_UNAVAILABLE_TEXT,
                    MUTATION_MODEL_UNAVAILABLE,
                    replies,
                    log_context,
                    "model.unavailable",
                )

        if corrected is not None:
            if getattr(corrected, "tool_calls", None):
                _log(log_context, "agent_mutation_retry", "tool_call_recovered")
                return {"messages": [corrected]}

            if searched:
                # The edit-only model still made no call; a tool-less rewrite
                # loop cannot produce the edit, so ask for the target.
                _log(log_context, "agent_mutation_retry", "clarification_required", "mutation.tool_call_missing")
                return _server_only(
                    state,
                    MUTATION_TARGET_REQUIRED_TEXT,
                    MUTATION_TARGET_REQUIRED,
                    replies,
                    log_context,
                    "mutation.tool_call_missing",
                )

            # Without a search a prose reply may be a fair clarification.
            response = corrected

    return await _reviewed_reply(state, config, messages, response, replies, profile, log_context)


async def budget_decision_agent_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    models: AgentModels,
    replies: ReplyServices,
) -> dict[str, Any]:
    """Interpret a reply to a pending budget recommendation (two tools only)."""

    log_context = node_log_context(config)
    messages = build_budget_decision_messages(state)

    try:
        response = await models.budget.ainvoke(messages, config=config)
    except Exception:
        _log(log_context, "budget_decision", "unavailable", "model.unavailable")
        return _server_only(
            state,
            BUDGET_DECISION_UNAVAILABLE_TEXT,
            BUDGET_DECISION_UNAVAILABLE,
            replies,
            log_context,
            "model.unavailable",
        )

    if getattr(response, "tool_calls", None):
        return {"messages": [response]}

    return await _reviewed_reply(state, config, messages, response, replies, None, log_context)


# ── Internals ───────────────────────────────────────────────────────


def tool_rounds(state: AgentState) -> int:
    """Assistant messages with tool calls in the current turn."""

    return sum(
        1
        for message in current_turn(state.messages)
        if message.type == "ai" and getattr(message, "tool_calls", None)
    )


def _agent_messages(
    state: AgentState,
    profile: dict[str, Any] | None,
    factual: FactualQAService,
    log_context: LogContext,
) -> list[BaseMessage]:
    messages = build_prompt_messages(state, profile)

    # Mixed requests ("is it open tomorrow? if so add it") carry only this
    # turn's validated evidence, never model memory.
    try:
        evidence = mixed_intent_message(state, service=factual)
    except Exception:
        _log(log_context, "mixed_intent", "unavailable", "fact.mixed_intent_unavailable")
        evidence = None

    if evidence is not None:
        messages.append(evidence)

    # The user just answered our "which day?" question: restate the pending
    # edit intent (never its old provider payloads).
    instruction = pending_mutation_instruction(state)

    if instruction is not None:
        messages.append(HumanMessage(content=instruction))

    return messages


async def _reviewed_reply(
    state: AgentState,
    config: RunnableConfig,
    messages: list[BaseMessage],
    response: Any,
    replies: ReplyServices,
    profile: dict[str, Any] | None,
    log_context: LogContext,
) -> dict[str, Any]:
    result = await generate_reviewed_reply(
        generate=lambda previous: write_reply(replies.writer, messages, previous, config),
        build_context=lambda text, attempt: review_context_from_state(
            state,
            text,
            user_profile=profile,
            attempt=attempt,
        ),
        reviewer=replies.reviewer,
        initial_reply=None if response is None else message_text(getattr(response, "content", "")),
        log_context=log_context,
    )

    message = (
        AIMessage(content=result.reply)
        if result.reply is not None
        else server_reply(
            state,
            unverified_reply_text(state),
            marker=REPLY_UNVERIFIED,
            reviewer=replies.reviewer,
            log_context=log_context,
        )
    )

    return {
        "messages": [message],
        "output_review_attempts": result.attempts,
        "output_review_issue_codes": list(result.issue_codes),
    }


def _server_only(
    state: AgentState,
    text: str,
    marker: str,
    replies: ReplyServices,
    log_context: LogContext,
    issue_code: str,
) -> dict[str, Any]:
    return {
        "messages": [server_reply(state, text, marker=marker, reviewer=replies.reviewer, log_context=log_context)],
        "output_review_attempts": 0,
        "output_review_issue_codes": [issue_code],
    }


def _log(log_context: LogContext, stage: str, outcome: str, code: str | None = None) -> None:
    log_event(
        logger,
        f"planning.{stage}",
        context=log_context,
        stage=stage,
        outcome=outcome,
        issue_codes=[code] if code else [],
    )


__all__ = [
    "MAX_TOOL_ROUNDS",
    "AgentModels",
    "AgentToolset",
    "agent_node",
    "budget_decision_agent_node",
    "build_agent_toolset",
]
