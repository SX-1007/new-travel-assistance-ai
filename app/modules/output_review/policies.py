"""Deterministic output-review rules.

These checks need no model: a reply that fails them is provably unsafe, so
it is rejected before any reviewer call is spent on it.
"""

from __future__ import annotations

import json
import re
from typing import Final

from app.modules.output_review.models import (
    OutputReviewContext,
    OutputReviewDecision,
)


# Internal field names and phrases that must never reach a user.
_INTERNAL_KEYS: Final = (
    "accepted_plan_snapshot",
    "candidate_plan",
    "daily_map_info",
    "draft_itinerary",
    "planning_outcome",
    "tool_calls",
)
_INTERNAL_PHRASES: Final = (
    "internal state",
    "not injected into the working itinerary summary",
    "state update dict",
    "working itinerary summary",
)

_FENCED_JSON: Final = re.compile(r"```\s*(?:json)?\s*[\[{]", re.IGNORECASE)

# At most this many "[" / "{" positions are tried as JSON starts, so a reply
# full of brackets can't make the check slow.
_MAX_JSON_PROBES: Final = 64

_PLANNING_SUCCESS: Final = re.compile(
    r"\b(?:completed|created|drafted|finalized|finished|generated|planned|revised|updated)\b"
    r".{0,48}\b(?:itinerary|plan|trip)\b"
    r"|\b(?:itinerary|plan|trip)\b.{0,48}"
    r"\b(?:complete|completed|created|drafted|finalized|finished|ready|revised|updated)\b",
    re.IGNORECASE | re.DOTALL,
)

_MUTATION_TARGET = r"(?:activit(?:y|ies)|attraction|restaurant|hotel|flight|booking|stop|day)"
_MUTATION_DONE = (
    r"(?:added|changed|deleted|edited|modified|moved|removed|replaced|"
    r"rescheduled|swapped|updated)"
)
_MUTATION_SUCCESS: Final = re.compile(
    rf"\b{_MUTATION_DONE}\b.{{0,96}}\b{_MUTATION_TARGET}\b"
    r"|\bsuccessfully\s+(?:add|change|delete|edit|modify|move|remove|replace|"
    rf"reschedule|swap|update)\b.{{0,96}}\b{_MUTATION_TARGET}\b"
    rf"|\b{_MUTATION_TARGET}\b.{{0,96}}\b(?:is|are|was|were|has been|have been)\s+"
    rf"(?:successfully\s+)?{_MUTATION_DONE}\b",
    re.IGNORECASE | re.DOTALL,
)

# A clause is a statement of failure when a negation is followed by one of
# the success verbs ("I couldn't update…", "nothing was changed"). A negation
# elsewhere ("Don't worry, I've updated your hotel") does not count.
_NEGATED_SUCCESS: Final = re.compile(
    r"\b(?:cannot|can['’]t|could\s+not|couldn['’]t|did\s+not|didn['’]t|"
    r"not\s+able\s+to|unable\s+to|was\s+not|wasn['’]t|were\s+not|weren['’]t|"
    r"has\s+not|hasn['’]t|have\s+not|haven['’]t|will\s+not|won['’]t|nothing)\b"
    r".{0,64}?\b(?:add|added|change|changed|complete|completed|create|created|"
    r"delete|deleted|draft|drafted|edit|edited|finalize|finalized|finish|finished|"
    r"generate|generated|modify|modified|move|moved|plan|planned|remove|removed|"
    r"replace|replaced|reschedule|rescheduled|revise|revised|swap|swapped|"
    r"update|updated)\b",
    re.IGNORECASE | re.DOTALL,
)

# Clause boundaries: sentence ends, line breaks, semicolons and contrast
# words. Judging each clause on its own stops one negation elsewhere in the
# reply from excusing a separate false claim ("I couldn't change the hotel,
# but I updated your itinerary").
_CLAUSE_SPLIT: Final = re.compile(
    r"[.!?;\n]+|\b(?:but|however|although|though|whereas)\b",
    re.IGNORECASE,
)

# Read-only questions about earlier revisions legitimately use "changed",
# "replaced" or "updated" when describing history.
_HISTORY_REQUEST: Final = re.compile(
    r"\b(?:original|previous|prior|earlier|before|history|historical|used to|"
    r"difference|differences|compare|comparison)\b"
    r"|\b(?:what|which|how)\b.{0,96}\bdid\s+you\s+"
    r"(?:add|change|delete|edit|modify|move|remove|replace|reschedule|swap|update)\b"
    r"|\bwhat\b.{0,64}\b(?:changed|modified|replaced|removed|updated)\b",
    re.IGNORECASE | re.DOTALL,
)

_EDIT_VERB = r"(?:add|change|delete|edit|modify|move|remove|replace|reschedule|swap|update|adjust)"

# Explicit mutation language wins over historical words: "replace Day 2 with
# the previous activity" is still an edit request.
_EXPLICIT_MUTATION: Final = re.compile(
    r"^\s*(?:"
    r"(?:(?:on|for)\s+)?"
    r"(?:day\s*(?:#\s*)?0*[1-9]\d*|[1-9]\d*(?:st|nd|rd|th)\s+day)"
    r"\s*[,;:\-]?\s*"
    r")?"
    rf"(?:please\s+)?{_EDIT_VERB}\b"
    rf"|\b(?:can|could|would|will)\s+you\s+(?:please\s+)?{_EDIT_VERB}\b"
    rf"|\b(?:i\s+want|i(?:'d|\s+would)\s+like)\s+(?:you\s+to\s+)?{_EDIT_VERB}\b"
    rf"|\b(?:go\s+ahead|proceed)\b.{{0,48}}\b{_EDIT_VERB}\b",
    re.IGNORECASE | re.DOTALL,
)


def is_read_only_itinerary_history_request(request: str) -> bool:
    """True for questions that inspect or compare older plan revisions."""

    text = (request or "").strip()

    if not text or _EXPLICIT_MUTATION.search(text):
        return False

    return bool(_HISTORY_REQUEST.search(text))


def deterministic_output_review(context: OutputReviewContext) -> OutputReviewDecision:
    """Reject output that is locally provable to be unsafe or unsupported."""

    reply = context.proposed_reply.strip()

    if not reply:
        return OutputReviewDecision.reject(
            "reply.requirements_missing",
            feedback="Write a non-empty user-facing reply.",
        )

    if _FENCED_JSON.search(reply) or contains_json_container(reply):
        return _raw_internal_rejection()

    folded = reply.casefold()

    if any(key in folded for key in _INTERNAL_KEYS) or any(
        phrase in folded for phrase in _INTERNAL_PHRASES
    ):
        return _raw_internal_rejection()

    if (
        not context.has_qualified_candidate
        and not is_read_only_itinerary_history_request(context.latest_user_request)
        and claims_unconfirmed_success(reply)
    ):
        return OutputReviewDecision.reject(
            "reply.unsupported_claim",
            feedback=(
                "Do not claim that the plan or itinerary was created or changed: "
                "no change was saved this turn."
            ),
        )

    return OutputReviewDecision.approve()


def claims_unconfirmed_success(reply: str) -> bool:
    """True when any clause states a plan or edit succeeded, without negation."""

    for clause in _CLAUSE_SPLIT.split(reply):
        if not clause.strip() or _NEGATED_SUCCESS.search(clause):
            continue

        if _PLANNING_SUCCESS.search(clause) or _MUTATION_SUCCESS.search(clause):
            return True

    return False


def contains_json_container(reply: str) -> bool:
    """Detect a valid JSON object/array anywhere in assistant text.

    A single-integer array such as ``[1]`` inside prose is a citation and is
    allowed; any other valid array or object is raw data.
    """

    decoder = json.JSONDecoder()
    probes = 0

    for index, character in enumerate(reply):
        if character not in "[{":
            continue

        probes += 1

        if probes > _MAX_JSON_PROBES:
            break

        try:
            parsed, end = decoder.raw_decode(reply, index)
        except (ValueError, RecursionError):
            continue

        if isinstance(parsed, dict):
            return True

        if isinstance(parsed, list):
            if not reply[:index].strip() and not reply[end:].strip():
                return True

            is_citation = (
                len(parsed) == 1
                and isinstance(parsed[0], int)
                and not isinstance(parsed[0], bool)
            )

            if not is_citation:
                return True

    return False


def _raw_internal_rejection() -> OutputReviewDecision:
    return OutputReviewDecision.reject(
        "reply.raw_internal_data",
        feedback="Rewrite as concise user-facing prose without JSON or internal state.",
    )


__all__ = [
    "claims_unconfirmed_success",
    "contains_json_container",
    "deterministic_output_review",
    "is_read_only_itinerary_history_request",
]
