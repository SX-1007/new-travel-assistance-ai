"""Who asked for which itinerary change, read from the conversation.

Provides the inputs ``modules/itinerary/editing.apply_edits`` trusts:

* ``trusted_place_options`` / ``trusted_booking_options``: provider results
  from searches the assistant made in THIS turn, bound to their tool calls.
* ``day_authorizer``: whether the user (never the assistant alone) named the
  day of a destructive place edit.

It also owns the edit-intent rules the graph uses for clarifications and
fallback wording, and the pending-edit record kept while waiting for a day.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from langchain_core.messages import BaseMessage

from app.modules.activities.service import ActivitySearchService
from app.modules.itinerary.editing import TrustedBookingOption
from app.modules.output_review.policies import is_read_only_itinerary_history_request
from app.orchestration.messages import (
    current_turn_tool_exchanges,
    latest_user_index,
    latest_user_request,
    message_text,
)
from app.orchestration.state import AgentState
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

# ``additional_kwargs["server_owned"]`` of the fixed "which day?" question.
DAY_CLARIFICATION_MARKER: Final = "mutation_target_day_required"

_MAX_SOURCE_REQUEST: Final = 300

_NUMBER_WORDS: Final = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_ORDINAL_WORDS: Final = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_NUMBER = r"0*[1-9]\d*|" + "|".join(_NUMBER_WORDS)
_ORDINAL = r"[1-9]\d*(?:st|nd|rd|th)|" + "|".join(_ORDINAL_WORDS)

# "day 2", "Day #2", "day two", "2nd day", "second day".
_DAY_REFERENCE: Final = re.compile(
    rf"\bday\s*(?:#\s*)?(?P<number>{_NUMBER})\b|\b(?P<ordinal>{_ORDINAL})\s+day\b",
    re.IGNORECASE,
)

_EDIT_VERBS = r"(?:add|change|edit|modify|move|remove|replace|reschedule|swap|delete)"

_EXPLICIT_EDIT_REQUEST: Final = re.compile(
    r"^\s*(?:(?:(?:on|for)\s+)?(?:the\s+)?"
    rf"(?:day\s*(?:#\s*)?(?:{_NUMBER})|(?:{_ORDINAL})\s+day)\s*[,;:\-]?\s*)?"
    rf"(?:please\s+)?{_EDIT_VERBS}\b"
    rf"|\b(?:can|could|would|will)\s+you\s+(?:please\s+)?{_EDIT_VERBS}\b"
    rf"|\b(?:i\s+want|i(?:'d|\s+would)\s+like)\s+(?:you\s+to\s+)?{_EDIT_VERBS}\b",
    re.IGNORECASE,
)

_DISLIKE_EDIT_REQUEST: Final = re.compile(
    r"\bi\s+(?:also\s+)?(?:(?:do\s+not|don't|dont)\s+(?:like|want)|dislike|hate)\b"
    r".{0,96}\b(?:activit(?:y|ies)|attraction|restaurant|hotel|flight|booking|stop)\b",
    re.IGNORECASE | re.DOTALL,
)

_PLACE_ITEM_REFERENCE: Final = re.compile(
    r"\b(?:activity|restaurant)\s*(?:#\s*)?\d+\b"
    rf"|\b(?:{_ORDINAL})\s+(?:activity|restaurant)\b",
    re.IGNORECASE,
)

_PLACE_EDIT_INTENT: Final = re.compile(
    r"\b(?:remove|delete|replace|change|swap)\b"
    r"|\bi\s+(?:also\s+)?(?:(?:do\s+not|don't|dont)\s+(?:like|want)|dislike|hate)\b",
    re.IGNORECASE,
)

_PROPOSAL_EDIT_VERB: Final = re.compile(
    r"\b(?:add|change|edit|modify|move|remove|replace|reschedule|swap|update)\b",
    re.IGNORECASE,
)

_AFFIRMATIVE: Final = re.compile(
    r"^\s*(?:yes|yeah|yep|yes\s+please|sure|okay|ok|confirm(?:ed)?|go\s+ahead|"
    r"please\s+do|do\s+it|sounds\s+good|let'?s\s+do\s+it)\b",
    re.IGNORECASE,
)
# "ok, no, wait" or "yes but not that one" is not a confirmation.
_HESITATION: Final = re.compile(
    r"\b(?:no|not|don'?t|cancel|wait|stop|instead|hold\s+on|never\s*mind)\b",
    re.IGNORECASE,
)

_DAY_ONLY_REPLY: Final = re.compile(
    r"^\s*(?:(?:i\s+mean|it(?:'s|\s+is))\s+)?(?:(?:on\s+)?day\s*(?:#\s*)?)?"
    rf"(?P<number>{_NUMBER})\s*[.!]?\s*$"
    rf"|^\s*(?:(?:i\s+mean|it(?:'s|\s+is))\s+)?(?:the\s+)?(?P<ordinal>{_ORDINAL})(?:\s+day)?\s*[.!]?\s*$",
    re.IGNORECASE,
)


# ── Day references ──────────────────────────────────────────────────


def day_numbers(text: str) -> frozenset[int]:
    """Every day number the text names explicitly."""

    days: set[int] = set()

    for match in _DAY_REFERENCE.finditer(text or ""):
        day = _to_day(match.group("number"), match.group("ordinal"))

        if day is not None:
            days.add(day)

    return frozenset(days)


def _to_day(number: str | None, ordinal: str | None) -> int | None:
    if number:
        key = number.lower()
        return _NUMBER_WORDS.get(key) or (int(key) if key.isdigit() else None)

    if ordinal:
        key = ordinal.lower()
        return _ORDINAL_WORDS.get(key) or int(re.sub(r"\D", "", key))

    return None


# ── Trusted provider evidence ───────────────────────────────────────


def trusted_place_options(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Place results from this turn's ``search_places`` calls."""

    options: list[dict[str, Any]] = []

    for _, result in current_turn_tool_exchanges(messages, "search_places"):
        raw = result.get("results") if isinstance(result, Mapping) else None

        if isinstance(raw, list):
            options.extend(dict(option) for option in raw if isinstance(option, Mapping))

    return options


def trusted_booking_options(messages: Sequence[BaseMessage]) -> list[TrustedBookingOption]:
    """Flight/hotel options from this turn's ``search_alternative_opt`` calls,
    bound to the category and day the assistant asked for."""

    options: list[TrustedBookingOption] = []

    for args, result in current_turn_tool_exchanges(messages, "search_alternative_opt"):
        category = "hotel" if args.get("category") == "accommodation" else args.get("category")
        day = args.get("day_num")
        raw = result.get("options") if isinstance(result, Mapping) else None

        if (
            category not in {"flight", "hotel"}
            or not isinstance(day, int)
            or isinstance(day, bool)
            or not isinstance(raw, list)
        ):
            continue

        options.extend(
            TrustedBookingOption(category=category, day=day, option=dict(option))
            for option in raw
            if isinstance(option, Mapping)
        )

    return options


def recover_place_options(
    edit: Mapping[str, Any],
    state: AgentState,
    *,
    search: ActivitySearchService,
) -> list[dict[str, Any]]:
    """Search the place an add/replace names when the assistant skipped
    ``search_places`` this turn.

    Tool order is not a trust boundary: rather than trusting the assistant's
    remembered details, the server runs the lookup itself and still refuses
    the edit if the provider can't verify the place.
    """

    if edit.get("category") not in {"activity", "restaurant"} or edit.get("action") not in {"add", "replace"}:
        return []

    details = edit.get("new_details")
    name = details.get("name") if isinstance(details, Mapping) else None

    if not isinstance(name, str) or not name.strip():
        return []

    location = details.get("location")
    city_hint = location.get("requested_city") if isinstance(location, Mapping) else None

    try:
        response = search.search(
            place_type="restaurant" if edit.get("category") == "restaurant" else "attraction",
            query=name.strip(),
            country=state.country or "",
            cities=state.city,
            # Only a hint: the search service keeps results inside the
            # configured trip cities whatever the model wrote here.
            city_hint=city_hint if isinstance(city_hint, str) else "",
        )
    except Exception:
        safe_log(logger, logging.WARNING, "edit_intent.place_recovery_failed")
        return []

    raw = response.get("results") if isinstance(response, Mapping) else None

    if not isinstance(raw, list):
        return []

    return [dict(option) for option in raw if isinstance(option, Mapping)]


# ── Day authorization ───────────────────────────────────────────────


def pending_day_reply(state: AgentState) -> int | None:
    """The day the user gave in direct reply to our "which day?" question."""

    pending = state.pending_itinerary_mutation

    if not isinstance(pending, Mapping) or pending.get("status") != "awaiting_day":
        return None

    previous_ai = _previous_ai_message(state.messages)

    if previous_ai is None or (previous_ai.additional_kwargs or {}).get("server_owned") != DAY_CLARIFICATION_MARKER:
        return None

    match = _DAY_ONLY_REPLY.fullmatch(latest_user_request(state.messages))

    return _to_day(match.group("number"), match.group("ordinal")) if match else None


def confirmed_proposal_days(state: AgentState) -> frozenset[int]:
    """Days the user just confirmed, if they had named them first.

    The current message must be a plain "yes"; the assistant message just
    before it must propose an edit; and only days that the user's own message
    before that proposal named count. An assistant-only guess never does.
    """

    request = latest_user_request(state.messages)

    if not _AFFIRMATIVE.search(request) or _HESITATION.search(request):
        return frozenset()

    current = latest_user_index(state.messages)

    if current is None:
        return frozenset()

    proposal_index = _previous_index(state.messages, current, "ai")

    if proposal_index is None:
        return frozenset()

    proposal = message_text(state.messages[proposal_index].content)

    if not _PROPOSAL_EDIT_VERB.search(proposal):
        return frozenset()

    user_index = _previous_index(state.messages, proposal_index, "human")

    if user_index is None:
        return frozenset()

    earlier_request = message_text(state.messages[user_index].content)

    return day_numbers(proposal) & day_numbers(earlier_request)


def is_target_day_authorized(state: AgentState, day: int) -> bool:
    """Whether the user named ``day`` for a destructive place edit."""

    return (
        day in day_numbers(latest_user_request(state.messages))
        or pending_day_reply(state) == day
        or day in confirmed_proposal_days(state)
    )


def day_authorizer(state: AgentState) -> Callable[[int], bool]:
    """``is_day_authorized`` for ``apply_edits``, computed once per call."""

    authorized = (
        day_numbers(latest_user_request(state.messages))
        | confirmed_proposal_days(state)
        | ({reply} if (reply := pending_day_reply(state)) is not None else set())
    )

    return lambda day: day in authorized


# ── Edit intent ─────────────────────────────────────────────────────


def is_explicit_edit_request(state: AgentState) -> bool:
    """Whether the user is asking to change the existing itinerary this turn."""

    if not state.draft_itinerary:
        return False

    if pending_day_reply(state) is not None or confirmed_proposal_days(state):
        return True

    request = latest_user_request(state.messages)

    if is_read_only_itinerary_history_request(request):
        return False

    return bool(_EXPLICIT_EDIT_REQUEST.search(request) or _DISLIKE_EDIT_REQUEST.search(request))


def place_edit_day_clarification(state: AgentState) -> str | None:
    """Fixed "which day?" question when a numbered place edit omits the day."""

    if not state.draft_itinerary:
        return None

    request = latest_user_request(state.messages)

    if is_read_only_itinerary_history_request(request):
        return None

    item = _PLACE_ITEM_REFERENCE.search(request)

    if item is None or not _PLACE_EDIT_INTENT.search(request) or day_numbers(request):
        return None

    return f"Sure — which day is {item.group(0).strip()} on? Your itinerary has not been changed."


# ── Pending edit while waiting for a day ────────────────────────────


def pending_mutation_record(
    edits: Sequence[Any],
    *,
    source_request: str = "",
) -> dict[str, Any]:
    """What to remember while asking for the day: intent only.

    Provider payloads (coordinates, prices, ratings) are never kept across
    turns; the place is searched again once the day is known.
    """

    pending_edits: list[dict[str, Any]] = []

    for edit in edits:
        if not isinstance(edit, Mapping):
            continue

        item: dict[str, Any] = {"action": edit.get("action"), "category": edit.get("category")}
        index = edit.get("index")

        if isinstance(index, int) and not isinstance(index, bool) and index > 0:
            item["index"] = index

        details = edit.get("new_details")
        name = details.get("name") if isinstance(details, Mapping) else None

        if isinstance(name, str) and name.strip():
            item["place_name"] = name.strip()[:120]

        pending_edits.append(item)

    return {
        "status": "awaiting_day",
        # The old record always stored None here, so the continuation lost
        # the user's wording.
        "source_request": source_request.strip()[:_MAX_SOURCE_REQUEST] or None,
        "edits": pending_edits,
    }


def pending_mutation_instruction(state: AgentState) -> str | None:
    """Private continuation instruction once the user has supplied the day."""

    day = pending_day_reply(state)
    pending = state.pending_itinerary_mutation

    if day is None or not isinstance(pending, Mapping):
        return None

    edit_lines: list[str] = []

    for edit in pending.get("edits") or []:
        if not isinstance(edit, Mapping):
            continue

        parts = [str(edit.get("action") or "change"), str(edit.get("category") or "activity")]

        if edit.get("index") is not None:
            parts.append(f"index {edit['index']}")

        if edit.get("place_name"):
            parts.append(f"place {edit['place_name']}")

        edit_lines.append("; ".join(parts))

    intent: list[str] = []
    source = pending.get("source_request")

    if isinstance(source, str) and source.strip():
        intent.append(f"original user request: {source.strip()}")

    if edit_lines:
        intent.append("pending edits: " + " | ".join(edit_lines))

    if not intent:
        return None

    return (
        "Internal server-owned clarification continuation: the user has now "
        f"supplied Day {day} for the pending itinerary edit. Pending intent: "
        f"{' ; '.join(intent)}. Do not reuse provider payloads from an earlier "
        "turn. For every activity/restaurant add or replacement, perform a fresh "
        f"search_places lookup in THIS turn, then call edit_itinerary with Day {day}. "
        "Complete all pending edits atomically when possible."
    )


# ── Internals ───────────────────────────────────────────────────────


def _previous_index(messages: Sequence[BaseMessage], before: int, message_type: str) -> int | None:
    for index in range(before - 1, -1, -1):
        if messages[index].type == message_type:
            return index

    return None


def _previous_ai_message(messages: Sequence[BaseMessage]) -> BaseMessage | None:
    current = latest_user_index(messages)

    if current is None:
        return None

    index = _previous_index(messages, current, "ai")
    return None if index is None else messages[index]


__all__ = [
    "DAY_CLARIFICATION_MARKER",
    "confirmed_proposal_days",
    "day_authorizer",
    "day_numbers",
    "is_explicit_edit_request",
    "is_target_day_authorized",
    "pending_day_reply",
    "pending_mutation_instruction",
    "pending_mutation_record",
    "place_edit_day_clarification",
    "recover_place_options",
    "trusted_booking_options",
    "trusted_place_options",
]
