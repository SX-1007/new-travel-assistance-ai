"""Public replies the nodes send: reviewed model text or checked server text.

Two rules cover every public message:

* model-written text goes through ``generate_reviewed_reply`` (deterministic
  rules, then the semantic reviewer, with bounded rewrites);
* fixed server text goes through ``OutputReviewService.check`` only. The old
  graph sent its own fixed apology and budget texts to the paid semantic
  reviewer (and could ask the model to "rewrite" them); it also raised
  ``RuntimeError`` and failed the whole turn when a fallback text failed the
  rules. Now a rejected server text is logged and replaced with the generic
  fallback.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.modules.output_review.models import OutputReviewDecision
from app.modules.output_review.policies import is_read_only_itinerary_history_request
from app.modules.output_review.service import OutputReviewService
from app.orchestration.edit_intent import is_explicit_edit_request
from app.orchestration.messages import latest_user_request, message_text
from app.orchestration.nodes.transaction import PlanKind
from app.orchestration.state import AgentState
from app.orchestration.workflows.output_review import (
    effective_plan_revision,
    review_context_from_state,
    rewrite_instruction,
)
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)

# ``AIMessage.additional_kwargs`` key marking fixed server text.
SERVER_OWNED_KEY: Final = "server_owned"

# Markers (the router and the edit-intent rules read some of them).
PLANNING_UNAVAILABLE: Final = "planning_unavailable"
EDIT_REJECTED: Final = "incremental_edit_rejected"
REPLY_UNVERIFIED: Final = "reply_unverified"
AGENT_UNAVAILABLE: Final = "agent_unavailable"
MUTATION_MODEL_UNAVAILABLE: Final = "mutation_model_unavailable"
MUTATION_TARGET_REQUIRED: Final = "mutation_target_required"
BUDGET_DECISION_UNAVAILABLE: Final = "budget_decision_unavailable"
BUDGET_GATE: Final = "budget_gate"


# ── Fixed texts ─────────────────────────────────────────────────────

GENERIC_UNVERIFIED: Final = (
    "I couldn't put together a verified answer to that just now, and your trip "
    "plan has not been changed. Please try again or rephrase your question."
)
EDIT_NOT_SAVED: Final = (
    "I couldn't verify a complete response for that request, so no unverified "
    "changes were saved to your trip. Please try again or provide the specific "
    "trip detail you would like me to update."
)
HISTORY_UNVERIFIED: Final = (
    "I couldn't verify that historical trip detail from the trusted itinerary "
    "information available in this conversation. Your current itinerary has "
    "not been changed."
)
NEW_PLAN_UNAVAILABLE: Final = (
    "I couldn't build a verified itinerary for this trip right now. Please try "
    "again in a moment; nothing has been saved yet."
)
CHANGE_NOT_SAVED: Final = (
    "I couldn't build a verified plan for that change, so nothing was saved and "
    "your existing itinerary is unchanged. Please try again shortly."
)
EDIT_NOT_VERIFIED: Final = (
    "I found the requested itinerary change, but I couldn't verify the final "
    "updated plan, so I did not save the change. Your existing itinerary has "
    "been preserved exactly as it was."
)
EDIT_FAILED_VALIDATION: Final = (
    # The old text said "the updated day did not pass validation", which the
    # deterministic rules read as a success claim; it was never checked.
    "I found the requested change, but the result did not pass itinerary "
    "validation, so I did not save it. Your existing itinerary is unchanged."
)
EDIT_NOT_APPLICABLE: Final = (
    "I found the requested change, but I couldn't apply it safely to the current "
    "itinerary. Your existing itinerary has been kept exactly as it was."
)
AGENT_UNAVAILABLE_TEXT: Final = (
    "Sorry, I ran into a temporary problem while handling that. Your trip plan "
    "has not been changed; please try again."
)
MUTATION_MODEL_UNAVAILABLE_TEXT: Final = (
    "I found matching place options, but I could not complete the itinerary "
    "update right now. Please try the modification again."
)
# The old text asked for "the item number" even when the user had given the
# day and position, so it read as if the request had not been understood.
MUTATION_TARGET_REQUIRED_TEXT: Final = (
    "I looked for places but couldn't choose one safely, so your itinerary has "
    "not been changed. Tell me the day, which item to change (for example "
    "\"activity 1\") and the place you'd like instead."
)
_MAX_CHOICES: Final = 3
BUDGET_DECISION_UNAVAILABLE_TEXT: Final = (
    "I could not interpret that budget reply. Please confirm the recommended "
    "amount, provide another positive amount, or request a refreshed "
    "recommendation."
)
BUDGET_CONFIRMATION_TEXT: Final = (
    "Before I re-plan the trip, please confirm the recommended minimum budget "
    "or give another amount."
)
BUDGET_UNAVAILABLE_TEXT: Final = (
    "Unable to verify the budget right now. Your existing trip plan has not "
    "been changed."
)

# Instruction appended when the model writes the reply for a qualified plan.
_NO_INTERNALS = (
    "Do not mention internal validation, candidate plans, output review, or "
    "internal state."
)
PLAN_INSTRUCTIONS: Final[dict[PlanKind, str]] = {
    PlanKind.NEW: (
        "Write a clear user-facing summary of the qualified itinerary. Mention the "
        "destination and the most important trip information that is useful to the "
        "traveller. Do not claim that an existing itinerary was changed unless the "
        "user actually requested a modification. Do not merely say that the "
        f"itinerary is ready. {_NO_INTERNALS} Keep the response concise but informative."
    ),
    PlanKind.EDIT: (
        "Write a clear user-facing confirmation of the itinerary change. Explicitly "
        "state: (1) the day and activity/booking position changed, (2) the previous "
        "item and the new item when available, (3) that the affected day's route, "
        "map, and totals were recalculated when applicable, and (4) that unrelated "
        "itinerary items were preserved. Do not merely say that the itinerary is "
        f"ready. {_NO_INTERNALS} Keep the response around 2 to 4 useful sentences."
    ),
    PlanKind.BUDGET: (
        "The user's budget change has already been accepted and the qualified "
        "candidate is the completed itinerary after that budget change. Write a "
        "concise confirmation that the accepted budget has been applied and that "
        "the itinerary was recalculated. Mention the destination and useful current "
        "trip information. Do not ask the user to confirm the budget again. Do not "
        f"claim that a booking was purchased or finalized. {_NO_INTERNALS}"
    ),
    PlanKind.REQUIREMENTS: (
        "The user's change to the trip details (travellers, dates, destination or "
        "origin) has been applied and the qualified candidate is the re-planned "
        "itinerary. Confirm which details changed and summarise the updated trip "
        f"concisely. {_NO_INTERNALS}"
    ),
    PlanKind.ALLOCATION: (
        "The user's budget-category change has been applied. Confirm the new "
        "category amounts from the trusted budget allocation and mention any "
        f"category that had to be rebalanced. {_NO_INTERNALS} Keep it short."
    ),
}


def plan_instruction(kind: PlanKind, view: AgentState) -> str:
    """The writing instruction for a qualified plan, with the accepted
    budget's exact figures for a budget re-plan.

    Found live: after "change my budget to 5000 MYR" was refused and the
    recommended MYR 8,068.14 accepted, the writer still wrote "MYR 5,000"
    from the conversation, and the reviewer rejected every draft.
    """

    instruction = PLAN_INSTRUCTIONS[kind]

    if kind is not PlanKind.BUDGET:
        return instruction

    home, destination = view.base_currency_code, view.dest_currency_code
    home_total, destination_total = view.total_base_budget, view.total_convert_budget

    if not (home and destination and home_total and destination_total):
        return instruction

    return (
        f"{instruction} The accepted total budget is exactly {home} {home_total:,.2f} "
        f"({destination} {destination_total:,.2f}); state these figures. Amounts "
        "mentioned earlier in the conversation are not the budget."
    )

# Used when the reply reviewer is unavailable or rejects every draft, so a
# deterministically validated plan is not thrown away over its wording.
QUALIFIED_FALLBACKS: Final[dict[PlanKind, str]] = {
    PlanKind.NEW: (
        "Your itinerary has been generated successfully. The latest trip details, "
        "activities, accommodation, routes, and estimated costs are available in "
        "the itinerary."
    ),
    PlanKind.EDIT: (
        "Your requested itinerary change has been applied successfully. The "
        "affected trip details, route, map, and totals have been updated where "
        "necessary, while unrelated itinerary items have been preserved."
    ),
    PlanKind.BUDGET: (
        "Your new budget has been applied and the itinerary was recalculated for "
        "it. The updated activities, accommodation, routes, and estimated costs "
        "are available in the itinerary."
    ),
    PlanKind.REQUIREMENTS: (
        "Your trip details have been updated and the itinerary was re-planned to "
        "match. The latest activities, accommodation, routes, and estimated costs "
        "are available in the itinerary."
    ),
    PlanKind.ALLOCATION: (
        "Your budget allocation has been updated. The new category amounts are "
        "shown with your itinerary."
    ),
}


# ── Services ────────────────────────────────────────────────────────


class ChatModel(Protocol):
    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class ReplyServices:
    """What the reply-writing nodes need: the reviewer and a tool-less model."""

    reviewer: OutputReviewService
    writer: ChatModel


# ── Helpers ─────────────────────────────────────────────────────────


def server_reply(
    state: AgentState,
    text: str,
    *,
    marker: str,
    reviewer: OutputReviewService,
    log_context: LogContext | None = None,
    fallback: str = GENERIC_UNVERIFIED,
) -> AIMessage:
    """Fixed server text, after the deterministic rules (no model call)."""

    decision = reviewer.check(review_context_from_state(state, text, stage="server_reply"))

    if not decision.approved:
        log_event(
            logger,
            "reply.server_text",
            context=log_context,
            stage="server_reply",
            outcome="rejected",
            issue_codes=list(decision.issue_codes),
            reason=marker,
        )
        text = fallback

    return AIMessage(content=text, additional_kwargs={SERVER_OWNED_KEY: marker})


def mutation_choice_text(option_names: Sequence[str], *, day: int | None = None) -> str:
    """The "which place?" question when a search found options but no edit
    was made: the user picks from what was found instead of starting over."""

    names = list(dict.fromkeys(name.strip() for name in option_names if name and name.strip()))[:_MAX_CHOICES]

    if not names:
        return MUTATION_TARGET_REQUIRED_TEXT

    listed = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}"
    # The reply must name the day again: only a day the user states can
    # authorize a remove or replace.
    example = f"On day {day or 1}, replace activity 1 with {names[0]}."

    return (
        f"I found {listed}, and your itinerary has not been changed yet. Which one "
        f"would you like? Reply with the day and the item, for example: \"{example}\""
    )


def unverified_reply_text(state: AgentState) -> str:
    """Fallback when no model reply passed review (no plan was built)."""

    if is_explicit_edit_request(state):
        return EDIT_NOT_SAVED

    if is_read_only_itinerary_history_request(latest_user_request(state.messages)):
        return HISTORY_UNVERIFIED

    # The old graph used the "historical trip detail" text for every
    # non-edit question.
    return GENERIC_UNVERIFIED


def planning_failure_text(state: AgentState, kind: PlanKind) -> str:
    """Fallback when a plan could not be built, reviewed or promoted."""

    if kind is PlanKind.EDIT:
        return EDIT_NOT_VERIFIED

    return CHANGE_NOT_SAVED if effective_plan_revision(state) > 0 else NEW_PLAN_UNAVAILABLE


async def write_reply(
    model: ChatModel,
    messages: Sequence[BaseMessage],
    previous: OutputReviewDecision | None,
    config: RunnableConfig | None,
) -> str:
    """One draft from a tool-less model; a rejected draft adds the review feedback."""

    prompt = [*messages, HumanMessage(content=rewrite_instruction(previous))] if previous else list(messages)
    response = await model.ainvoke(prompt, config=config)
    return message_text(getattr(response, "content", ""))


__all__ = [
    "AGENT_UNAVAILABLE",
    "BUDGET_DECISION_UNAVAILABLE",
    "BUDGET_GATE",
    "EDIT_REJECTED",
    "MUTATION_MODEL_UNAVAILABLE",
    "MUTATION_TARGET_REQUIRED",
    "PLANNING_UNAVAILABLE",
    "PLAN_INSTRUCTIONS",
    "plan_instruction",
    "QUALIFIED_FALLBACKS",
    "REPLY_UNVERIFIED",
    "SERVER_OWNED_KEY",
    "ChatModel",
    "ReplyServices",
    "mutation_choice_text",
    "planning_failure_text",
    "server_reply",
    "unverified_reply_text",
    "write_reply",
]
