"""Factual Q&A contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RequestType(str, Enum):
    FACTUAL = "factual"
    PREDICTION = "prediction"
    OPINION = "opinion"
    RECOMMENDATION = "recommendation"
    ACTION = "action"
    CONVERSATION = "conversation"
    MIXED = "mixed"


class FactualDomain(str, Enum):
    STATIC_GENERAL = "static_general"
    TRIP_STATE = "trip_state"
    PLACE_INFORMATION = "place_information"
    OPENING_HOURS = "opening_hours"
    WEATHER = "weather"
    TRANSIT_STATUS = "transit_status"
    VISA_ENTRY = "visa_entry"
    TRAVEL_SAFETY = "travel_safety"
    HEALTH = "health"
    CURRENCY = "currency"
    FLIGHT_STATUS = "flight_status"
    HOTEL_INFORMATION = "hotel_information"
    EVENT_INFORMATION = "event_information"
    OTHER = "other"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RequiredAuthority(str, Enum):
    NONE = "none"
    TRIP_STATE = "trip_state"
    ANY_CITED = "any_cited"
    PRIMARY_OR_TRUSTED = "primary_or_trusted"
    OFFICIAL_OR_TRUSTED = "official_or_trusted"
    OFFICIAL = "official"


class SourceAuthority(str, Enum):
    TRIP_STATE = "trip_state"
    OFFICIAL = "official"
    PRIMARY = "primary"
    TRUSTED_PROVIDER = "trusted_provider"
    SECONDARY = "secondary"
    COMMUNITY = "community"
    UNKNOWN = "unknown"


class RequestClassification(BaseModel):
    """Semantic classification only; server policy is applied afterwards."""

    model_config = ConfigDict(extra="forbid")

    request_type: RequestType
    domain: FactualDomain = FactualDomain.OTHER
    factual_question: str = ""
    target_location: str | None = None
    target_date: str | None = None
    time_sensitive: bool = False
    location_sensitive: bool = False
    requires_trip_context: bool = False


class FactualRoute(BaseModel):
    """Final server-owned routing policy for the current turn."""

    model_config = ConfigDict(extra="forbid")

    request_type: RequestType
    domain: FactualDomain
    factual_question: str
    target_location: str | None = None
    target_date: str | None = None
    traveller_origin: str | None = None
    time_sensitive: bool = False
    location_sensitive: bool = False
    requires_trip_context: bool = False

    risk_level: RiskLevel = RiskLevel.LOW
    requires_external_evidence: bool = False
    required_authority: RequiredAuthority = RequiredAuthority.NONE
    freshness_minutes: int | None = None
    minimum_sources: int = Field(default=0, ge=0, le=5)

    # How the request was classified: "model", "rules" (no model call
    # needed) or "fallback" (model unavailable).
    classified_by: str = "model"


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    snippet: str
    source_name: str = ""
    authority: SourceAuthority = SourceAuthority.UNKNOWN
    retrieved_at: str
    source_date: str | None = None


class EvidenceValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    issue_codes: list[str] = Field(default_factory=list)
    usable_sources: int = 0
    official_sources: int = 0
    trusted_sources: int = 0


class GroundingReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    issue_codes: list[str] = Field(default_factory=list)
    feedback: str = ""

    @model_validator(mode="after")
    def _coherent(self) -> GroundingReview:
        if self.approved and self.issue_codes:
            raise ValueError("approved review cannot contain issue codes")
        if not self.approved and not self.issue_codes:
            raise ValueError("rejected review must contain an issue code")
        return self


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One raw web result, before authority and freshness are judged."""

    title: str
    url: str
    snippet: str
    source_name: str = ""
    source_date: str | None = None


@dataclass(frozen=True, slots=True)
class TripContext:
    """The trip facts factual Q&A may use; built by the orchestration layer."""

    origin_country: str | None = None
    destination_country: str | None = None
    cities: tuple[str, ...] = ()
    start_date: str | None = None
    end_date: str | None = None
    num_people: int | None = None
    plan_revision: int = 0
    # Compact itinerary; only sent to models for trip-state questions.
    itinerary: Sequence[Mapping[str, Any]] = ()

    @property
    def destination(self) -> str | None:
        city = self.cities[0].strip() if self.cities else ""
        country = (self.destination_country or "").strip()

        if city and country:
            return f"{city}, {country}"

        return city or country or None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    evidence: tuple[EvidenceItem, ...]
    validation: EvidenceValidation


@dataclass(frozen=True, slots=True)
class FactualAnswer:
    """Text to publish; ``issue_codes`` empty means a grounded answer."""

    text: str
    issue_codes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def grounded(self) -> bool:
        return not self.issue_codes


__all__ = [
    "EvidenceItem",
    "EvidenceValidation",
    "FactualAnswer",
    "FactualDomain",
    "FactualRoute",
    "GroundingReview",
    "RequestClassification",
    "RequestType",
    "RequiredAuthority",
    "RetrievalResult",
    "RiskLevel",
    "SearchHit",
    "SourceAuthority",
    "TripContext",
]
