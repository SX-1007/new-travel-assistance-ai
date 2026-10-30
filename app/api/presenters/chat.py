"""Chat responses: one finished turn, or one restored session.

Both read only server-owned state. Precedence for a turn: an active budget
gate, then a failed plan, then the revalidated accepted plan.

CPU-bound (the itinerary validator runs per plan); callers run these off
the event loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Final

from app.api.presenters.budget import (
    BUDGET_CHECK_UNAVAILABLE_MESSAGE,
    EXPIRED_BUDGET_CONFIRMATION_MESSAGE,
    chat_gate,
)
from app.api.presenters.plan import (
    PLANNING_UNAVAILABLE_MESSAGE,
    AcceptedPlan,
    accepted_plan,
    planning_unavailable_reason,
    turn_reply,
)
from app.api.schemas.chat import (
    ChatBudgetGateResponse,
    ChatHistoryBudget,
    ChatHistoryMessage,
    ChatHistorySession,
    ChatPlanningUnavailableResponse,
    ChatSuccessResponse,
)
from app.modules.output_review.models import OutputReviewContext
from app.modules.output_review.service import OutputReviewService
from app.orchestration.graph import TurnResult
from app.shared.text import content_text


UNSAFE_HISTORY_MESSAGE: Final = (
    "A saved assistant response was hidden because it could not be safely displayed."
)
_TITLE_MAX_CHARS: Final = 48

ChatTurnResponse = ChatSuccessResponse | ChatBudgetGateResponse | ChatPlanningUnavailableResponse


# ── One turn ────────────────────────────────────────────────────────


def chat_turn_response(result: TurnResult, *, now: datetime | None = None) -> ChatTurnResponse:
    state = result.state
    gate, card = chat_gate(state, now=now)

    if gate == "confirmation":
        assert card is not None
        return ChatBudgetGateResponse(
            status="budget_confirmation_required",
            chat_reply=card.chat_reply,
            budget_confirmation=card,
        )

    if gate == "unavailable":
        return ChatBudgetGateResponse(status="budget_check_unavailable", chat_reply=BUDGET_CHECK_UNAVAILABLE_MESSAGE)

    messages = state.get("messages") or ()
    reply = turn_reply(messages)

    if state.get("planning_outcome") == "unavailable":
        # The graph's own failure reply (already checked) says what was kept;
        # the old router replaced it with a generic retry message.
        return ChatPlanningUnavailableResponse(
            reason=planning_unavailable_reason(state),
            chat_reply=reply or PLANNING_UNAVAILABLE_MESSAGE,
        )

    plan = accepted_plan(state)

    if plan is None or reply is None:
        return ChatPlanningUnavailableResponse(reason="validation_failed", chat_reply=PLANNING_UNAVAILABLE_MESSAGE)

    return ChatSuccessResponse(
        chat_reply=reply,
        draft_itinerary=plan.draft_itinerary,
        daily_map_info=plan.daily_map_info,
        destination_country_code=plan.destination_country_code,
        itinerary_modified=result.plan_changed,
        total_budget=plan.total_convert_budget,
        currency=plan.dest_currency_code,
        budget_allocation=plan.budget_allocation,
    )


# ── Restored history ────────────────────────────────────────────────


def history_session(
    record: Mapping[str, Any],
    reviewer: OutputReviewService,
    *,
    now: datetime | None = None,
) -> ChatHistorySession | None:
    """One saved conversation for display, or None when it can't be listed
    (no id, or a timestamp the client could not parse)."""

    session_id = str(record.get("session_id") or "").strip()
    updated_at = str(record.get("updated_at") or "")

    if not session_id or not _is_timestamp(updated_at):
        return None

    raw_state = record.get("state")
    state: Mapping[str, Any] = raw_state if isinstance(raw_state, Mapping) else {}
    plan = accepted_plan(state)
    messages, plan_messages = _visible_messages(state.get("messages") or (), plan, reviewer)

    _apply_budget_gate(messages, state, now=now)
    _attach_plan(messages, plan, plan_messages)

    destination = _destination_label(state)
    first_user = next((message.content for message in messages if message.role == "user"), "")

    return ChatHistorySession(
        id=session_id,
        title=destination or " ".join(first_user.split())[:_TITLE_MAX_CHARS] or "New trip",
        destination=destination,
        updated_at=updated_at,
        messages=messages,
    )


def _visible_messages(
    raw: Sequence[Any],
    plan: AcceptedPlan | None,
    reviewer: OutputReviewService,
) -> tuple[list[ChatHistoryMessage], dict[int, int]]:
    """User and final assistant messages; each stored reply re-checked by the
    deterministic review rules. Also returns, per plan revision, the index
    of the reply that published it."""

    messages: list[ChatHistoryMessage] = []
    plan_messages: dict[int, int] = {}
    latest_request = ""

    for message in raw:
        kind = getattr(message, "type", "")

        # Tool-calling AI messages are intermediate steps, never shown live.
        if kind not in {"human", "ai"} or getattr(message, "tool_calls", None):
            continue

        content = content_text(getattr(message, "content", "")).strip()

        if not content:
            continue

        if kind == "human":
            latest_request = content
            messages.append(ChatHistoryMessage(role="user", content=content))
            continue

        decision = reviewer.check(
            OutputReviewContext(
                latest_user_request=latest_request,
                proposed_reply=content,
                # A saved plan stands behind the replies that described it.
                candidate_summary={"plan_revision": plan.plan_revision} if plan is not None else None,
                stage="history_restore",
            )
        )
        messages.append(ChatHistoryMessage(role="ai", content=content if decision.approved else UNSAFE_HISTORY_MESSAGE))

        revision = (getattr(message, "additional_kwargs", None) or {}).get("plan_revision")

        if isinstance(revision, int) and not isinstance(revision, bool) and revision > 0:
            plan_messages[revision] = len(messages) - 1

    return messages, plan_messages


def _apply_budget_gate(
    messages: list[ChatHistoryMessage],
    state: Mapping[str, Any],
    *,
    now: datetime | None,
) -> None:
    """The newest reply shows the gate as it stands now, not as stored."""

    latest_ai = _last_index(messages, "ai")

    if latest_ai is None:
        return

    gate, card = chat_gate(state, now=now)
    target = messages[latest_ai]

    if gate == "confirmation":
        assert card is not None
        target.content = card.chat_reply
        target.budget_confirmation = card

    elif gate == "unavailable":
        expired = (
            state.get("budget_gate_outcome") == "budget_confirmation_required"
            and state.get("pending_budget_confirmation") is not None
        )
        target.content = EXPIRED_BUDGET_CONFIRMATION_MESSAGE if expired else BUDGET_CHECK_UNAVAILABLE_MESSAGE


def _attach_plan(
    messages: list[ChatHistoryMessage],
    plan: AcceptedPlan | None,
    plan_messages: Mapping[int, int],
) -> None:
    """Checkpoints keep only the latest plan: attach it to the reply that
    published it (sessions from before revision tags: the first reply)."""

    if plan is None or not (plan.draft_itinerary or plan.daily_map_info):
        return

    index = plan_messages.get(plan.plan_revision)

    if index is None and not plan_messages:
        index = _first_index(messages, "ai")

    if index is None:
        return

    target = messages[index]
    target.itinerary = plan.draft_itinerary
    target.maps = plan.daily_map_info
    target.destination_country_code = plan.destination_country_code
    target.budget = ChatHistoryBudget(
        total=plan.total_convert_budget,
        currency=plan.dest_currency_code,
        allocation=plan.budget_allocation,
    )


def _destination_label(state: Mapping[str, Any]) -> str:
    country = str(state.get("country") or "").strip()
    raw_cities = state.get("city") or []
    cities = [raw_cities] if isinstance(raw_cities, str) else list(raw_cities)
    parts = [str(city).strip() for city in cities if str(city).strip()]

    return ", ".join([*parts, country] if country else parts)


def _is_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False

    return True


def _last_index(messages: Sequence[ChatHistoryMessage], role: str) -> int | None:
    return next((index for index in range(len(messages) - 1, -1, -1) if messages[index].role == role), None)


def _first_index(messages: Sequence[ChatHistoryMessage], role: str) -> int | None:
    return next((index for index, message in enumerate(messages) if message.role == role), None)


__all__ = [
    "UNSAFE_HISTORY_MESSAGE",
    "ChatTurnResponse",
    "chat_turn_response",
    "history_session",
]
