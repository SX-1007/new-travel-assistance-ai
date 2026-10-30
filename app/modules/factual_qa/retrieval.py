"""Evidence rules: search query, source authority, freshness and sufficiency."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Final
from urllib.parse import urlparse

from app.modules.factual_qa.models import (
    EvidenceItem,
    EvidenceValidation,
    FactualDomain,
    FactualRoute,
    RequiredAuthority,
    SearchHit,
    SourceAuthority,
    TripContext,
)


MAX_EVIDENCE_ITEMS: Final = 8
_MAX_TITLE: Final = 200
_MAX_SNIPPET: Final = 600
_MAX_URL: Final = 500

# Government and intergovernmental registries. Matched only at the END of
# the host, so "visa.gov.example.com" is not official.
_OFFICIAL_SUFFIX: Final = re.compile(
    r"(?:^|\.)(?:gov|mil|int)$"
    r"|(?:^|\.)(?:gov|go|gob|gouv|govt|gv|gub)\.[a-z]{2}$"
    r"|(?:^|\.)(?:gc\.ca|europa\.eu|admin\.ch|bund\.de|government\.nl)$"
)

_TRUSTED_PROVIDER_HOSTS: Final = frozenset(
    {
        "weather.com",
        "accuweather.com",
        "timeanddate.com",
        "flightaware.com",
        "flightradar24.com",
        "iata.org",
    }
)
_COMMUNITY_HOSTS: Final = frozenset(
    {
        "reddit.com",
        "tripadvisor.com",
        "quora.com",
        "facebook.com",
        "instagram.com",
        "tiktok.com",
        "x.com",
        "twitter.com",
    }
)
_SECONDARY_HOSTS: Final = frozenset(
    {
        "wikipedia.org",
        "wikivoyage.org",
        "booking.com",
        "expedia.com",
        "agoda.com",
        "kayak.com",
        "skyscanner.com",
        "moovitapp.com",
        "rome2rio.com",
    }
)

_TRUSTED: Final = frozenset(
    {SourceAuthority.OFFICIAL, SourceAuthority.PRIMARY, SourceAuthority.TRUSTED_PROVIDER}
)

_DOMAIN_QUERY_SUFFIX: Final = {
    FactualDomain.TRAVEL_SAFETY: "official government travel advisory",
    FactualDomain.HEALTH: "official health authority government",
    FactualDomain.TRANSIT_STATUS: "official service status",
    FactualDomain.FLIGHT_STATUS: "official flight status",
    FactualDomain.OPENING_HOURS: "official opening hours",
}

_RELATIVE_AGE: Final = re.compile(
    r"(\d+)\s*(minute|min|hour|hr|day|week|month|year)s?\s+ago",
    re.IGNORECASE,
)
_ABSOLUTE_FORMATS: Final = ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y")
_UNIT_MINUTES: Final = {
    "minute": 1,
    "min": 1,
    "hour": 60,
    "hr": 60,
    "day": 24 * 60,
    "week": 7 * 24 * 60,
    "month": 30 * 24 * 60,
    "year": 365 * 24 * 60,
}


def source_host(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower().strip(".")
    except ValueError:
        return ""

    return host[4:] if host.startswith("www.") else host


def _host_in(host: str, candidates: Iterable[str]) -> bool:
    return any(host == item or host.endswith(f".{item}") for item in candidates)


def classify_source_authority(url: str, title: str = "") -> SourceAuthority:
    host = source_host(url)

    if not host or urlparse(url).scheme not in {"http", "https"}:
        return SourceAuthority.UNKNOWN

    if _OFFICIAL_SUFFIX.search(host):
        return SourceAuthority.OFFICIAL

    if _host_in(host, _TRUSTED_PROVIDER_HOSTS):
        return SourceAuthority.TRUSTED_PROVIDER

    if _host_in(host, _COMMUNITY_HOSTS):
        return SourceAuthority.COMMUNITY

    if _host_in(host, _SECONDARY_HOSTS):
        return SourceAuthority.SECONDARY

    # First-party commercial sites (attractions, operators). Weaker than
    # OFFICIAL, so it can never satisfy a visa or health question.
    folded = title.casefold()

    if "official site" in folded or "official website" in folded:
        return SourceAuthority.PRIMARY

    return SourceAuthority.SECONDARY


def build_search_query(route: FactualRoute, trip: TripContext) -> str:
    pieces = [route.factual_question]

    if route.target_location and route.target_location.casefold() not in route.factual_question.casefold():
        pieces.append(route.target_location)

    if route.target_date:
        pieces.append(route.target_date)

    if route.domain is FactualDomain.VISA_ENTRY:
        if route.traveller_origin:
            pieces.append(f"citizen of {route.traveller_origin}")
        if trip.destination_country:
            pieces.append(f"travel to {trip.destination_country}")
        pieces.append("official immigration embassy government")
    elif route.domain in _DOMAIN_QUERY_SUFFIX:
        pieces.append(_DOMAIN_QUERY_SUFFIX[route.domain])

    return " ".join(piece.strip() for piece in pieces if piece and piece.strip())


def evidence_from_hits(hits: Sequence[SearchHit], *, retrieved_at: datetime) -> list[EvidenceItem]:
    """Bounded, de-duplicated evidence with authority attached."""

    stamp = retrieved_at.astimezone(timezone.utc).isoformat()
    seen: set[str] = set()
    evidence: list[EvidenceItem] = []

    for hit in hits:
        url = (hit.url or "").strip()[:_MAX_URL]
        snippet = " ".join((hit.snippet or "").split())[:_MAX_SNIPPET]
        title = " ".join((hit.title or "").split())[:_MAX_TITLE] or source_host(url)

        if not url or not snippet or url in seen:
            continue

        seen.add(url)
        evidence.append(
            EvidenceItem(
                title=title,
                url=url,
                snippet=snippet,
                source_name=(hit.source_name or source_host(url))[:_MAX_TITLE],
                authority=classify_source_authority(url, title),
                retrieved_at=stamp,
                source_date=(hit.source_date or None),
            )
        )

        if len(evidence) >= MAX_EVIDENCE_ITEMS:
            break

    return evidence


def source_age(source_date: str | None, now: datetime) -> timedelta | None:
    """Age of a source from its displayed date; None when it can't be read."""

    if not source_date:
        return None

    text = source_date.strip()

    if match := _RELATIVE_AGE.search(text):
        unit = match.group(2).lower()
        return timedelta(minutes=int(match.group(1)) * _UNIT_MINUTES[unit])

    for fmt in _ABSOLUTE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        return max(now - parsed, timedelta(0))

    return None


def is_stale(item: EvidenceItem, route: FactualRoute, now: datetime) -> bool:
    """A dated source older than the route allows.

    A calendar date (no time of day) can't prove freshness below one day, so
    those are judged against at least one day. Undated sources are kept.
    """

    if route.freshness_minutes is None:
        return False

    age = source_age(item.source_date, now)

    if age is None:
        return False

    allowed = timedelta(minutes=route.freshness_minutes)

    if not _RELATIVE_AGE.search(item.source_date or ""):
        allowed = max(allowed, timedelta(days=1))

    return age > allowed


def authority_satisfies(required: RequiredAuthority, actual: SourceAuthority) -> bool:
    if required is RequiredAuthority.NONE:
        return True
    if required is RequiredAuthority.TRIP_STATE:
        return actual is SourceAuthority.TRIP_STATE
    if required is RequiredAuthority.ANY_CITED:
        return actual not in {SourceAuthority.UNKNOWN, SourceAuthority.COMMUNITY}
    if required is RequiredAuthority.PRIMARY_OR_TRUSTED:
        return actual in _TRUSTED
    if required is RequiredAuthority.OFFICIAL_OR_TRUSTED:
        return actual in {SourceAuthority.OFFICIAL, SourceAuthority.TRUSTED_PROVIDER}
    if required is RequiredAuthority.OFFICIAL:
        return actual is SourceAuthority.OFFICIAL
    return False


def usable_evidence(
    route: FactualRoute,
    evidence: Sequence[EvidenceItem],
    now: datetime,
) -> list[EvidenceItem]:
    """Evidence strong and fresh enough for this route; the only evidence a
    model is allowed to answer from."""

    return [
        item
        for item in evidence
        if authority_satisfies(route.required_authority, item.authority)
        and not is_stale(item, route, now)
    ]


def validate_factual_evidence(
    route: FactualRoute,
    evidence: Sequence[EvidenceItem],
    now: datetime,
) -> EvidenceValidation:
    if not route.requires_external_evidence:
        return EvidenceValidation(approved=True)

    if not evidence:
        return EvidenceValidation(approved=False, issue_codes=["fact.evidence_missing"])

    usable = usable_evidence(route, evidence, now)
    # Several pages from one site are one source.
    usable_hosts = {source_host(item.url) for item in usable}
    official = [item for item in usable if item.authority is SourceAuthority.OFFICIAL]
    trusted = [item for item in usable if item.authority in _TRUSTED]

    issues: list[str] = []

    if len(usable_hosts) < route.minimum_sources:
        stale_only = any(
            authority_satisfies(route.required_authority, item.authority) and is_stale(item, route, now)
            for item in evidence
        )
        issues.append("fact.evidence_stale" if stale_only else "fact.source_count_insufficient")

    # Authority is judged on every result, so a stale official page is
    # reported as stale rather than as a weak source.
    if route.required_authority is RequiredAuthority.OFFICIAL and not any(
        item.authority is SourceAuthority.OFFICIAL for item in evidence
    ):
        issues.append("fact.authority_insufficient")
    elif route.required_authority in {
        RequiredAuthority.PRIMARY_OR_TRUSTED,
        RequiredAuthority.OFFICIAL_OR_TRUSTED,
    } and not any(item.authority in _TRUSTED for item in evidence):
        issues.append("fact.authority_insufficient")

    return EvidenceValidation(
        approved=not issues,
        issue_codes=issues,
        usable_sources=len(usable_hosts),
        official_sources=len({source_host(item.url) for item in official}),
        trusted_sources=len({source_host(item.url) for item in trusted}),
    )


__all__ = [
    "MAX_EVIDENCE_ITEMS",
    "authority_satisfies",
    "build_search_query",
    "classify_source_authority",
    "evidence_from_hits",
    "is_stale",
    "source_age",
    "source_host",
    "usable_evidence",
    "validate_factual_evidence",
]
