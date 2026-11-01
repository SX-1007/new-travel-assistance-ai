"""Server-owned routing policy for factual questions.

A model may classify a message, but these rules decide whether external
evidence is mandatory and how strong it must be. Obvious live or high-risk
topics (visa, health, weather, flight status...) are detected here and can
never be downgraded by the model.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from app.modules.factual_qa.models import (
    EvidenceValidation,
    FactualDomain,
    FactualRoute,
    RequestClassification,
    RequestType,
    RequiredAuthority,
    RiskLevel,
    TripContext,
)


_LIVE_WORDS: Final = re.compile(
    r"\b(?:today|tomorrow|tonight|now|currently|current|latest|live|real[- ]?time|"
    r"delayed?|cancelled?|canceled?|disruption|open now|this morning|"
    r"this afternoon|this evening)\b",
    re.IGNORECASE,
)
_OPENING_WORDS: Final = re.compile(
    r"\b(?:opening hours?|closing hours?|opening time|closing time|"
    r"what time .{0,40}(?:open|close)|when .{0,40}(?:open|close)|open now)\b",
    re.IGNORECASE,
)
_WEATHER_WORDS: Final = re.compile(
    r"\b(?:weather|rain|raining|forecast|temperature|storm|typhoon|snow|humid|humidity)\b",
    re.IGNORECASE,
)
_TRANSIT_WORDS: Final = re.compile(
    r"\b(?:train|rail|mrt|lrt|metro|subway|bus|tram|ferry|transit)\b",
    re.IGNORECASE,
)
_TRANSIT_STATUS_WORDS: Final = re.compile(
    r"\b(?:delay|delayed|cancelled|canceled|disruption|suspended|service status|"
    r"running normally|on time|operating normally)\b",
    re.IGNORECASE,
)
_VISA_WORDS: Final = re.compile(
    r"\b(?:visas?|passport|entry requirements?|immigration|visa[- ]?free|"
    r"entry permit|arrival card|border requirements?)\b",
    re.IGNORECASE,
)
_SAFETY_WORDS: Final = re.compile(
    r"\b(?:travel advisory|travel warning|safe to travel|security advisory|"
    r"civil unrest|evacuation|natural disaster warning)\b",
    re.IGNORECASE,
)
_HEALTH_WORDS: Final = re.compile(
    r"\b(?:vaccin(?:e|es|ation|ations)|health requirements?|yellow fever|quarantine|"
    r"medical requirements?|disease outbreak|health advisory)\b",
    re.IGNORECASE,
)
_FLIGHT_STATUS_WORDS: Final = re.compile(
    r"\b(?:flight status|flight delayed|flight cancelled|flight canceled|"
    r"departure status|arrival status)\b",
    re.IGNORECASE,
)
_RECOMMENDATION_WORDS: Final = re.compile(
    r"\b(?:recommend|best|better|worth|should i|which .{0,60} should|prefer)\b",
    re.IGNORECASE,
)
_ACTION_WORDS: Final = re.compile(
    r"\b(?:add|remove|replace|change|edit|modify|book|reserve|cancel|update|"
    r"move|reschedule|convert|increase|decrease|swap|delete)\b",
    re.IGNORECASE,
)
_QUESTION: Final = re.compile(
    r"\?|^\s*(?:what|when|where|who|why|how|is|are|do|does|did|can|will|"
    r"tell me|check|find out)\b",
    re.IGNORECASE,
)
# Questions about the user's own trip are answered from trip state, never
# by sending the user's plan to a web search.
_TRIP_REFERENCE: Final = re.compile(
    r"\b(?:my|our)\s+(?:trip|itinerary|plan|schedule|hotel|flights?|booking|budget)\b"
    r"|\bday\s*#?\s*[1-9]\d*\b|\b[1-9]\d*(?:st|nd|rd|th)\s+day\b",
    re.IGNORECASE,
)
# Commands with no factual question in them: no model is needed to route.
_EDIT_COMMAND: Final = re.compile(
    r"^\s*(?:please\s+)?(?:(?:on|for)\s+day\s*\d+\s*[,:\-]?\s*)?"
    r"(?:add|remove|replace|change|edit|modify|swap|move|reschedule|delete|"
    r"update|book|find me|show me|search for)\b",
    re.IGNORECASE,
)
_SMALL_TALK: Final = re.compile(
    r"^\s*(?:thanks?(?: you)?(?: so much)?|thank you|ok(?:ay)?|cool|great|nice|"
    r"perfect|awesome|yes|yeah|yep|no|nope|sure|hi|hello|hey|bye|goodbye|"
    r"good (?:morning|afternoon|evening|night))[\s!.]*$",
    re.IGNORECASE,
)

_LOCATION_SENSITIVE: Final = frozenset(
    {
        FactualDomain.WEATHER,
        FactualDomain.TRANSIT_STATUS,
        FactualDomain.OPENING_HOURS,
        FactualDomain.FLIGHT_STATUS,
        FactualDomain.PLACE_INFORMATION,
        FactualDomain.EVENT_INFORMATION,
    }
)
_TRIP_CONTEXT_DOMAINS: Final = frozenset(
    {
        FactualDomain.TRIP_STATE,
        FactualDomain.WEATHER,
        FactualDomain.TRANSIT_STATUS,
        FactualDomain.VISA_ENTRY,
        FactualDomain.FLIGHT_STATUS,
    }
)
# Entry and health rules are published as reference pages that stay in
# force until the authority changes them. A page's date says when it was
# written, not whether it still applies, so a page-age limit would reject
# the current rule (a 2016 embassy page on visa exemption). Their safeguards
# are official sources only, two separate sites and the grounding review.
_STANDING_RULE_DOMAINS: Final = frozenset({FactualDomain.VISA_ENTRY, FactualDomain.HEALTH})
_FACTUAL_TYPES: Final = frozenset(
    {RequestType.FACTUAL, RequestType.PREDICTION, RequestType.MIXED}
)


@dataclass(frozen=True, slots=True)
class DomainPolicy:
    risk: RiskLevel
    external: bool
    authority: RequiredAuthority
    freshness_minutes: int | None
    minimum_sources: int


_DAY: Final = 24 * 60

DOMAIN_POLICIES: Final[Mapping[FactualDomain, DomainPolicy]] = {
    FactualDomain.STATIC_GENERAL: DomainPolicy(RiskLevel.LOW, False, RequiredAuthority.NONE, None, 0),
    FactualDomain.TRIP_STATE: DomainPolicy(RiskLevel.LOW, False, RequiredAuthority.TRIP_STATE, None, 0),
    FactualDomain.PLACE_INFORMATION: DomainPolicy(
        RiskLevel.LOW, True, RequiredAuthority.PRIMARY_OR_TRUSTED, _DAY, 1
    ),
    FactualDomain.OPENING_HOURS: DomainPolicy(
        RiskLevel.MEDIUM, True, RequiredAuthority.PRIMARY_OR_TRUSTED, 180, 1
    ),
    FactualDomain.WEATHER: DomainPolicy(
        RiskLevel.MEDIUM, True, RequiredAuthority.OFFICIAL_OR_TRUSTED, 30, 1
    ),
    # Operators and airlines usually publish on commercial first-party
    # domains, so PRIMARY is accepted for live transport status.
    FactualDomain.TRANSIT_STATUS: DomainPolicy(
        RiskLevel.MEDIUM, True, RequiredAuthority.PRIMARY_OR_TRUSTED, 5, 1
    ),
    # Standing rules (see _STANDING_RULE_DOMAINS): no page-age limit.
    FactualDomain.VISA_ENTRY: DomainPolicy(RiskLevel.HIGH, True, RequiredAuthority.OFFICIAL, None, 2),
    FactualDomain.TRAVEL_SAFETY: DomainPolicy(RiskLevel.HIGH, True, RequiredAuthority.OFFICIAL, 60, 2),
    FactualDomain.HEALTH: DomainPolicy(RiskLevel.HIGH, True, RequiredAuthority.OFFICIAL, None, 2),
    FactualDomain.CURRENCY: DomainPolicy(
        RiskLevel.LOW, True, RequiredAuthority.OFFICIAL_OR_TRUSTED, 60, 1
    ),
    FactualDomain.FLIGHT_STATUS: DomainPolicy(
        RiskLevel.MEDIUM, True, RequiredAuthority.PRIMARY_OR_TRUSTED, 5, 1
    ),
    FactualDomain.HOTEL_INFORMATION: DomainPolicy(
        RiskLevel.LOW, True, RequiredAuthority.PRIMARY_OR_TRUSTED, _DAY, 1
    ),
    FactualDomain.EVENT_INFORMATION: DomainPolicy(
        RiskLevel.MEDIUM, True, RequiredAuthority.PRIMARY_OR_TRUSTED, 180, 1
    ),
    FactualDomain.OTHER: DomainPolicy(RiskLevel.LOW, False, RequiredAuthority.NONE, None, 0),
}


def detect_domain(text: str) -> FactualDomain | None:
    """Live or high-risk domain named in the text, highest risk first."""

    if _VISA_WORDS.search(text):
        return FactualDomain.VISA_ENTRY
    if _HEALTH_WORDS.search(text):
        return FactualDomain.HEALTH
    if _SAFETY_WORDS.search(text):
        return FactualDomain.TRAVEL_SAFETY
    if _FLIGHT_STATUS_WORDS.search(text):
        return FactualDomain.FLIGHT_STATUS
    if _WEATHER_WORDS.search(text):
        return FactualDomain.WEATHER
    if _TRANSIT_WORDS.search(text) and _TRANSIT_STATUS_WORDS.search(text):
        return FactualDomain.TRANSIT_STATUS
    if _OPENING_WORDS.search(text):
        return FactualDomain.OPENING_HOURS
    return None


def rule_classification(message: str) -> RequestClassification | None:
    """Classify without a model when the message is unambiguous.

    Small talk and plain edit commands ("add Tokyo Tower to day 2") carry no
    factual question; routing them needs no model call.
    """

    text = message.strip()

    if _SMALL_TALK.match(text):
        return RequestClassification(request_type=RequestType.CONVERSATION)

    if (
        _EDIT_COMMAND.match(text)
        and "?" not in text
        and detect_domain(text) is None
        and not _LIVE_WORDS.search(text)
    ):
        return RequestClassification(request_type=RequestType.ACTION)

    return None


def fallback_classification(message: str) -> RequestClassification:
    """Conservative classification used only when the model is unavailable."""

    text = message.strip()
    domain = detect_domain(text)
    is_question = bool(_QUESTION.search(text))
    has_action = bool(_ACTION_WORDS.search(text))
    has_recommendation = bool(_RECOMMENDATION_WORDS.search(text))

    if domain is None and _TRIP_REFERENCE.search(text) and is_question and not has_action:
        domain = FactualDomain.TRIP_STATE

    factual = domain is not None or (is_question and not has_action)

    if factual and domain is not None and (has_action or has_recommendation):
        request_type = RequestType.MIXED
    elif domain is FactualDomain.WEATHER:
        request_type = RequestType.PREDICTION
    elif factual:
        request_type = RequestType.FACTUAL
    elif has_action:
        request_type = RequestType.ACTION
    else:
        request_type = RequestType.CONVERSATION

    return RequestClassification(
        request_type=request_type,
        domain=domain or FactualDomain.OTHER,
        factual_question=text if factual else "",
    )


def apply_factual_policy(
    raw: RequestClassification,
    message: str,
    trip: TripContext,
    *,
    classified_by: str = "model",
) -> FactualRoute:
    """Turn a classification into the server-owned retrieval policy."""

    text = message.strip()
    detected = detect_domain(text)
    domain = detected or raw.domain
    request_type = raw.request_type

    # Questions about the user's own trip never go to a web search.
    if (
        detected is None
        and domain in {FactualDomain.OTHER, FactualDomain.STATIC_GENERAL}
        and _TRIP_REFERENCE.search(text)
        and request_type in {RequestType.FACTUAL, RequestType.MIXED}
    ):
        domain = FactualDomain.TRIP_STATE

    if domain is FactualDomain.WEATHER and request_type is RequestType.FACTUAL:
        request_type = RequestType.PREDICTION

    # A detected live/high-risk topic next to an action or recommendation is
    # verified first. Generic question words alone ("can you add...") are not
    # a factual signal: they would make edit commands look like fact questions.
    if detected is not None and request_type in {
        RequestType.ACTION,
        RequestType.RECOMMENDATION,
        RequestType.FACTUAL,
        RequestType.PREDICTION,
    } and (_ACTION_WORDS.search(text) or _RECOMMENDATION_WORDS.search(text)):
        request_type = RequestType.MIXED

    policy = DOMAIN_POLICIES[domain]
    external = policy.external
    authority = policy.authority
    freshness = policy.freshness_minutes
    minimum_sources = policy.minimum_sources

    time_sensitive = raw.time_sensitive or bool(_LIVE_WORDS.search(text))

    # A fact asked "now"/"today" may not come from model memory, even when
    # no specific domain was recognised. Trip-state questions are exempt,
    # and so are mixed requests without a recognised factual topic.
    if (
        time_sensitive
        and domain is not FactualDomain.TRIP_STATE
        and request_type in _FACTUAL_TYPES
        and (request_type is not RequestType.MIXED or domain is not FactualDomain.OTHER)
    ):
        if not external:
            authority = RequiredAuthority.ANY_CITED
        external = True
        if domain not in _STANDING_RULE_DOMAINS:
            freshness = min(freshness or 60, 60)
        minimum_sources = max(minimum_sources, 1)

    location_sensitive = raw.location_sensitive or domain in _LOCATION_SENSITIVE
    target_location = (raw.target_location or "").strip() or None

    if location_sensitive and not target_location:
        target_location = trip.destination

    return FactualRoute(
        request_type=request_type,
        domain=domain,
        factual_question=raw.factual_question.strip() or text,
        target_location=target_location,
        target_date=(raw.target_date or "").strip() or None,
        traveller_origin=(trip.origin_country or "").strip() or None,
        time_sensitive=time_sensitive,
        location_sensitive=location_sensitive,
        requires_trip_context=raw.requires_trip_context or domain in _TRIP_CONTEXT_DOMAINS,
        risk_level=policy.risk,
        requires_external_evidence=external,
        required_authority=authority,
        freshness_minutes=freshness,
        minimum_sources=minimum_sources,
        classified_by=classified_by,
    )


def next_after_routing(route: FactualRoute) -> Literal["retrieval", "answer", "agent"]:
    """Where a routed turn goes: evidence first, a direct answer, or the agent."""

    if route.request_type not in _FACTUAL_TYPES:
        return "agent"

    if route.requires_external_evidence:
        return "retrieval"

    return "agent" if route.request_type is RequestType.MIXED else "answer"


def next_after_retrieval(
    route: FactualRoute,
    validation: EvidenceValidation,
) -> Literal["answer", "agent"]:
    """Unverified live or high-risk facts never reach the general agent."""

    if validation.approved and route.request_type is RequestType.MIXED:
        return "agent"

    return "answer"


def safe_unverified_message(route: FactualRoute, references: Sequence[str] = ()) -> str:
    """The fixed "couldn't verify" reply.

    ``references`` are official pages to check; they are listed only with
    the high-risk message, the one that tells the user to check an authority.
    """

    if route.risk_level is RiskLevel.HIGH:
        message = (
            "I couldn't verify this from enough authoritative current sources, "
            "so I don't want to give you an unsafe definitive answer. Please "
            "check the relevant official government, immigration, embassy, or "
            "health authority before relying on it."
        )

        if not references:
            return message

        listed = "\n".join(f"- {link}" for link in references)
        return f"{message}\n\nOfficial pages I found that may have the details:\n{listed}"

    if route.time_sensitive:
        return (
            "I couldn't verify the current information from a sufficiently "
            "reliable source right now, so I won't guess."
        )

    return "I couldn't verify that fact reliably enough to answer it confidently."


__all__ = [
    "DOMAIN_POLICIES",
    "DomainPolicy",
    "apply_factual_policy",
    "detect_domain",
    "fallback_classification",
    "next_after_retrieval",
    "next_after_routing",
    "rule_classification",
    "safe_unverified_message",
]
