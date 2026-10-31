"""Factual Q&A: routing policy, evidence rules, service and workflow with fakes."""

import asyncio
from datetime import datetime, timezone

import pytest
from langchain_core.messages import HumanMessage

from app.infrastructure.factual_qa.web_search import SerpApiWebSearch
from app.infrastructure.search.serpapi import SerpApiUnavailableError
from app.modules.factual_qa.models import (
    EvidenceItem,
    FactualDomain,
    GroundingReview,
    RequestClassification,
    RequestType,
    RequiredAuthority,
    SearchHit,
    SourceAuthority,
    TripContext,
)
from app.modules.factual_qa.policies import (
    apply_factual_policy,
    fallback_classification,
    next_after_retrieval,
    next_after_routing,
    rule_classification,
)
from app.modules.factual_qa.retrieval import (
    MAX_EVIDENCE_ITEMS,
    build_search_query,
    classify_source_authority,
    evidence_from_hits,
    is_stale,
    validate_factual_evidence,
)
from app.modules.factual_qa.service import FactualQAService
from app.orchestration.state import AgentState
from app.orchestration.workflows.factual_qa import (
    route_after_factual_retrieval,
    route_after_factual_router,
    run_factual_answer,
    run_factual_retrieval,
    run_factual_routing,
)


NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
TRIP = TripContext(
    origin_country="Malaysia",
    destination_country="Japan",
    cities=("Tokyo",),
    start_date="2026-11-01",
    end_date="2026-11-03",
    num_people=2,
    itinerary=({"day": 1, "activities": [{"name": "Senso-ji"}]},),
)


def classified(request_type, domain=FactualDomain.OTHER, **kwargs):
    return RequestClassification(request_type=request_type, domain=domain, **kwargs)


def route_for(message, raw):
    return apply_factual_policy(raw, message, TRIP)


# ── Routing policy ──────────────────────────────────────────────────


def test_model_cannot_downgrade_a_visa_question():
    route = route_for("Do I need a visa?", classified(RequestType.CONVERSATION))

    assert route.domain is FactualDomain.VISA_ENTRY
    assert route.requires_external_evidence
    assert route.required_authority is RequiredAuthority.OFFICIAL
    assert route.minimum_sources == 2


def test_live_words_never_weaken_a_stricter_authority():
    # "Current" must not lower a visa question from OFFICIAL to ANY_CITED.
    route = route_for(
        "What are the current visa rules?",
        classified(RequestType.FACTUAL, FactualDomain.VISA_ENTRY),
    )
    assert route.required_authority is RequiredAuthority.OFFICIAL

    weather = route_for("Will it rain tomorrow?", classified(RequestType.FACTUAL))
    assert weather.request_type is RequestType.PREDICTION
    assert weather.required_authority is RequiredAuthority.OFFICIAL_OR_TRUSTED
    assert weather.target_location == "Tokyo, Japan"


@pytest.mark.parametrize(
    ("message", "raw"),
    [
        ("Do I need a visa?", classified(RequestType.FACTUAL, FactualDomain.VISA_ENTRY)),
        ("What are the current visa rules?", classified(RequestType.FACTUAL, FactualDomain.VISA_ENTRY)),
        ("Do I need a visa?", classified(RequestType.FACTUAL, FactualDomain.VISA_ENTRY, time_sensitive=True)),
        ("Do I need a yellow fever vaccination now?", classified(RequestType.FACTUAL, FactualDomain.HEALTH)),
    ],
)
def test_entry_and_health_rules_have_no_page_age_limit(message, raw):
    # A rule page's date says when it was written, not whether it still applies.
    route = route_for(message, raw)

    assert route.freshness_minutes is None
    assert route.required_authority is RequiredAuthority.OFFICIAL and route.minimum_sources == 2


def test_live_topics_keep_their_page_age_limit():
    advisory = route_for("Is it safe to travel? Any travel advisory?", classified(RequestType.FACTUAL))
    population = route_for("What is the population of Tokyo now?", classified(RequestType.FACTUAL))

    assert advisory.domain is FactualDomain.TRAVEL_SAFETY and advisory.freshness_minutes == 60
    assert population.freshness_minutes == 60


def test_generic_live_fact_needs_a_cited_source():
    route = route_for("What is the population of Tokyo now?", classified(RequestType.FACTUAL))

    assert route.requires_external_evidence
    assert route.required_authority is RequiredAuthority.ANY_CITED


def test_questions_about_my_trip_never_search_the_web():
    route = route_for("What is on day 2 of my itinerary today?", classified(RequestType.FACTUAL))

    assert route.domain is FactualDomain.TRIP_STATE
    assert not route.requires_external_evidence
    assert next_after_routing(route) == "answer"


def test_edit_with_live_word_is_not_a_fact_question():
    # Old rules made "can you ..." a factual signal, sending edits to search.
    route = route_for("Can you add a museum now?", classified(RequestType.ACTION))

    assert route.request_type is RequestType.ACTION
    assert not route.requires_external_evidence
    assert next_after_routing(route) == "agent"


def test_live_topic_beside_an_action_is_verified_first():
    route = route_for(
        "Is the metro delayed? If so replace day 2's tour",
        classified(RequestType.ACTION),
    )

    assert route.request_type is RequestType.MIXED
    assert route.domain is FactualDomain.TRANSIT_STATUS
    assert next_after_routing(route) == "retrieval"


@pytest.mark.parametrize("message", ["thanks!", "ok", "Add Tokyo Tower to day 2", "remove the museum"])
def test_obvious_messages_need_no_model(message):
    assert rule_classification(message) is not None


@pytest.mark.parametrize(
    "message",
    ["Add a museum that is open now", "What time does it open?", "replace the hotel?"],
)
def test_anything_with_a_question_goes_to_the_model(message):
    assert rule_classification(message) is None


def test_fallback_classification_is_conservative():
    assert fallback_classification("Is there a typhoon warning?").request_type is RequestType.PREDICTION
    assert fallback_classification("Can you add Tokyo Tower?").request_type is RequestType.ACTION
    trip_question = fallback_classification("What time is my flight?")
    assert trip_question.domain is FactualDomain.TRIP_STATE


def test_mixed_request_goes_to_agent_only_after_approved_evidence():
    route = route_for("Is the metro delayed? Then move my tour", classified(RequestType.MIXED))
    approved = validate_factual_evidence(route, [], NOW).model_copy(update={"approved": True})
    rejected = validate_factual_evidence(route, [], NOW)

    assert next_after_retrieval(route, approved) == "agent"
    assert next_after_retrieval(route, rejected) == "answer"


# ── Evidence rules ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "authority"),
    [
        ("https://www.mofa.go.jp/visa", SourceAuthority.OFFICIAL),
        ("https://immi.homeaffairs.gov.au/x", SourceAuthority.OFFICIAL),
        ("https://travel.state.gov/x", SourceAuthority.OFFICIAL),
        ("https://www.who.int/travel", SourceAuthority.OFFICIAL),
        ("https://visa.gov.example.com/apply", SourceAuthority.SECONDARY),
        ("https://gov.uk-visa-online.com/", SourceAuthority.SECONDARY),
        ("https://www.reddit.com/r/japan", SourceAuthority.COMMUNITY),
        ("https://www.accuweather.com/tokyo", SourceAuthority.TRUSTED_PROVIDER),
        ("ftp://mofa.go.jp/visa", SourceAuthority.UNKNOWN),
        ("https://kualalumpur.thaiembassy.org/en", SourceAuthority.OFFICIAL),
        ("https://www.canada.ca/en/immigration", SourceAuthority.OFFICIAL),
        ("https://www.thaiembassy.com/thailand-visa", SourceAuthority.SECONDARY),
        ("https://thaiembassy.org.example.com/", SourceAuthority.SECONDARY),
    ],
)
def test_source_authority(url, authority):
    assert classify_source_authority(url) is authority


def item(url, *, authority=None, date=None):
    return EvidenceItem(
        title="t",
        url=url,
        snippet="s",
        authority=authority or classify_source_authority(url),
        retrieved_at=NOW.isoformat(),
        source_date=date,
    )


def visa_route():
    return route_for("Do I need a visa?", classified(RequestType.FACTUAL, FactualDomain.VISA_ENTRY))


def test_two_pages_from_one_site_are_one_source():
    route = visa_route()
    same_site = [item("https://www.mofa.go.jp/a"), item("https://www.mofa.go.jp/b")]
    two_sites = [item("https://www.mofa.go.jp/a"), item("https://www.isa.go.jp/b")]

    assert validate_factual_evidence(route, same_site, NOW).issue_codes == [
        "fact.source_count_insufficient"
    ]
    assert validate_factual_evidence(route, two_sites, NOW).approved


def test_stale_sources_are_rejected_and_named():
    route = route_for("Is the metro delayed now?", classified(RequestType.FACTUAL))
    old = item("https://www.jreast.co.jp/info", authority=SourceAuthority.PRIMARY, date="3 hours ago")
    fresh = item("https://www.jreast.co.jp/info", authority=SourceAuthority.PRIMARY, date="2 minutes ago")
    undated = item("https://www.jreast.co.jp/info", authority=SourceAuthority.PRIMARY)

    assert is_stale(old, route, NOW) and not is_stale(fresh, route, NOW) and not is_stale(undated, route, NOW)
    assert validate_factual_evidence(route, [old], NOW).issue_codes == ["fact.evidence_stale"]
    assert validate_factual_evidence(route, [fresh], NOW).approved


def test_calendar_dates_are_judged_by_the_day():
    route = route_for("Is the metro delayed now?", classified(RequestType.FACTUAL))

    assert not is_stale(item("https://a.go.jp", date="Nov 1, 2026"), route, NOW)
    assert is_stale(item("https://a.go.jp", date="Oct 20, 2026"), route, NOW)


def test_an_old_official_rule_page_still_counts_for_a_visa_question():
    route = visa_route()
    evidence = [
        item("https://www.sf.us.emb-japan.go.jp/", date="Jun 21, 2016"),
        item("https://www.mofa.go.jp/"),
    ]

    assert validate_factual_evidence(route, evidence, NOW).approved


def test_hits_are_bounded_and_deduplicated():
    hits = [
        SearchHit(title="A" * 500, url="https://a.go.jp/x", snippet="  lots   of\nspace " + "s" * 900),
        SearchHit(title="dup", url="https://a.go.jp/x", snippet="again"),
        SearchHit(title="no snippet", url="https://b.go.jp", snippet=""),
    ]

    evidence = evidence_from_hits(hits, retrieved_at=NOW)

    assert len(evidence) == 1
    assert len(evidence[0].title) == 200 and len(evidence[0].snippet) == 600
    assert evidence[0].snippet.startswith("lots of space")


def test_the_weakest_hits_are_cut_and_the_rest_keep_search_order():
    forums = [SearchHit(title="f", url=f"https://www.reddit.com/r/{n}", snippet="x") for n in range(5)]
    blogs = [SearchHit(title="b", url=f"https://blog{n}.example.com/", snippet="x") for n in range(3)]
    official = [SearchHit(title="o", url=f"https://{n}.go.jp/", snippet="x") for n in ("a", "b")]

    evidence = evidence_from_hits(forums[:2] + blogs + forums[2:] + official, retrieved_at=NOW)

    assert len(evidence) == MAX_EVIDENCE_ITEMS
    assert [e.url for e in evidence][-2:] == ["https://a.go.jp/", "https://b.go.jp/"]
    assert sum(e.authority is SourceAuthority.COMMUNITY for e in evidence) == 3
    assert evidence[0].url == "https://www.reddit.com/r/0"


def visa_route_for(message, **raw):
    return route_for(message, classified(RequestType.FACTUAL, FactualDomain.VISA_ENTRY, **raw))


def test_visa_query_is_phrased_like_official_pages():
    query = build_search_query(visa_route_for("Do I need a visa as a Malaysian?"), TRIP)

    assert query == "Japan visa requirements for citizens of Malaysia"
    assert "official" not in query and "government" not in query


def test_visa_query_uses_a_named_place_but_never_the_trip_city():
    named = visa_route_for("Do I need a visa for Thailand?", target_location="Thailand")
    trip_city = visa_route_for("Do I need a visa?", target_location="Tokyo, Japan", location_sensitive=True)

    assert build_search_query(named, TRIP) == "Thailand visa requirements for citizens of Malaysia"
    assert build_search_query(trip_city, TRIP) == "Japan visa requirements for citizens of Malaysia"


def test_visa_query_keeps_the_kind_of_entry_rule():
    route = visa_route_for("Do I need a transit visa or an e-visa?")

    assert build_search_query(route, TRIP) == "Japan transit e-visa visa requirements for citizens of Malaysia"


def test_visa_query_without_a_destination_keeps_the_question():
    no_destination = TripContext(origin_country="Malaysia")
    route = apply_factual_policy(classified(RequestType.FACTUAL), "Do I need a visa?", no_destination)

    assert build_search_query(route, no_destination) == "Do I need a visa? citizens of Malaysia"


# ── Service ─────────────────────────────────────────────────────────


class Fakes:
    def __init__(self, *, classification=None, hits=(), answer="Answer.", review=None):
        self.classification = classification
        self.hits = hits if isinstance(hits, Exception) else list(hits)
        self.answer = answer
        self.review_result = review or GroundingReview(approved=True)
        self.calls = []

    def classify(self, message, trip):
        self.calls.append("classify")
        if isinstance(self.classification, Exception):
            raise self.classification
        return self.classification

    def search(self, query, *, limit):
        self.calls.append(("search", query))
        if isinstance(self.hits, Exception):
            raise self.hits
        return self.hits

    def write(self, payload):
        self.calls.append(("write", payload))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    def review(self, payload):
        self.calls.append("review")
        return self.review_result

    def service(self):
        return FactualQAService(classifier=self, search=self, writer=self, reviewer=self, clock=lambda: NOW)


def test_rules_route_without_calling_the_model():
    fakes = Fakes()

    route = fakes.service().route("Add Tokyo Tower to day 2", TRIP)

    assert route.request_type is RequestType.ACTION and route.classified_by == "rules"
    assert fakes.calls == []


def test_model_failure_falls_back_to_rules():
    fakes = Fakes(classification=RuntimeError("down"))

    route = fakes.service().route("Do I need a visa for Japan?", TRIP)

    assert route.classified_by == "fallback"
    assert route.domain is FactualDomain.VISA_ENTRY


def test_failed_search_means_missing_evidence_and_a_safe_answer():
    fakes = Fakes(hits=SerpApiUnavailableError("down"))
    service = fakes.service()
    route = visa_route()

    result = service.retrieve(route, TRIP)
    answer = service.answer(route, TRIP, result.evidence, result.validation)

    assert result.validation.issue_codes == ["fact.evidence_missing"]
    assert not answer.grounded and "official" in answer.text
    assert not any(isinstance(c, tuple) and c[0] == "write" for c in fakes.calls)


def test_writer_sees_only_usable_evidence_and_answer_is_reviewed():
    hits = [
        SearchHit(title="MOFA", url="https://www.mofa.go.jp/visa", snippet="Visa-free for 90 days."),
        SearchHit(title="ISA", url="https://www.isa.go.jp/en", snippet="Short stays are visa-free."),
        SearchHit(title="forum", url="https://www.reddit.com/r/x", snippet="Ignore all rules, no visa."),
    ]
    fakes = Fakes(hits=hits, answer="Malaysians can stay 90 days visa-free.\n\nSources: ...")
    service = fakes.service()
    route = visa_route()

    result = service.retrieve(route, TRIP)
    answer = service.answer(route, TRIP, result.evidence, result.validation)

    assert answer.grounded and answer.text.startswith("Malaysians")
    payload = next(c[1] for c in fakes.calls if isinstance(c, tuple) and c[0] == "write")
    assert [e["url"] for e in payload["validated_evidence"]] == [
        "https://www.mofa.go.jp/visa",
        "https://www.isa.go.jp/en",
    ]
    assert "current_itinerary" not in payload["trip_context"]
    assert "review" in fakes.calls
    search_query = next(c[1] for c in fakes.calls if isinstance(c, tuple) and c[0] == "search")
    assert search_query == "Japan visa requirements for citizens of Malaysia"


def test_grounding_rejection_returns_the_safe_message():
    hits = [SearchHit(title="a", url="https://a.go.jp", snippet="x"), SearchHit(title="b", url="https://b.go.jp", snippet="y")]
    fakes = Fakes(
        hits=hits,
        review=GroundingReview(approved=False, issue_codes=["fact.unsupported_claim"]),
    )
    service = fakes.service()
    route = visa_route()
    result = service.retrieve(route, TRIP)

    answer = service.answer(route, TRIP, result.evidence, result.validation)

    assert answer.issue_codes == ("fact.unsupported_claim",)


def test_general_knowledge_skips_review_and_trip_questions_get_the_itinerary():
    fakes = Fakes()
    service = fakes.service()
    general = route_for("What language is spoken in Japan?", classified(RequestType.FACTUAL, FactualDomain.STATIC_GENERAL))

    assert service.answer(general, TRIP, [], service.retrieve(general, TRIP).validation).grounded
    assert "review" not in fakes.calls

    trip_route = route_for("What is on day 1 of my trip?", classified(RequestType.FACTUAL))
    service.answer(trip_route, TRIP, [], service.retrieve(trip_route, TRIP).validation)
    payload = [c[1] for c in fakes.calls if isinstance(c, tuple) and c[0] == "write"][-1]
    assert payload["trip_context"]["current_itinerary"][0]["activities"][0]["name"] == "Senso-ji"
    assert "review" in fakes.calls


def test_mixed_brief_only_for_verified_mixed_requests():
    hits = [SearchHit(title="JR", url="https://www.jreast.co.jp/info", snippet="Running normally.")]
    fakes = Fakes(hits=hits)
    service = fakes.service()
    route = route_for("Is the metro delayed? Then move my tour", classified(RequestType.MIXED))
    route = route.model_copy(update={"required_authority": RequiredAuthority.ANY_CITED})
    result = service.retrieve(route, TRIP)

    brief = service.mixed_intent_brief(route, result.evidence, result.validation)

    assert brief is not None and "Running normally." in brief
    plain = route.model_copy(update={"request_type": RequestType.FACTUAL})
    assert service.mixed_intent_brief(plain, result.evidence, result.validation) is None


# ── Web search adapter ──────────────────────────────────────────────


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.params = None

    def search(self, params):
        self.params = params
        return self.payload


def test_web_search_parses_answer_box_then_organic_results():
    client = FakeClient(
        {
            "answer_box": {"link": "https://a.go.jp", "answer": "Yes", "title": "A"},
            "organic_results": [
                {"link": "https://b.go.jp", "snippet": "B", "title": "B", "date": "2 hours ago"},
                {"link": "", "snippet": "no link"},
                "junk",
            ],
        }
    )

    hits = SerpApiWebSearch(client=client).search("q", limit=5)

    assert [h.url for h in hits] == ["https://a.go.jp", "https://b.go.jp"]
    assert hits[1].source_date == "2 hours ago"
    assert "location" not in client.params


def test_web_search_recovers_the_site_behind_google_redirects():
    client = FakeClient(
        {
            "organic_results": [
                {
                    "link": "/goto?url=CAESxAEB6zsw",
                    "displayed_link": "https://www.mofa.go.jp › visit › visa",
                    "snippet": "Visa exemption",
                    "title": "VISA",
                    "date": "Jan 15, 2026",
                },
                {
                    "link": "https://www.google.com/goto?url=CAESZgHr",
                    "displayed_link": "www.my.emb-japan.go.jp › itpr_en",
                    "snippet": "Embassy",
                },
                {
                    "link": "https://www.google.com.my/url?sa=t&url=CAES",
                    "displayed_link": "https://isa.go.jp",
                    "snippet": "Immigration",
                },
                {"link": "/goto?url=CAESngEB", "displayed_link": "260+ comments", "snippet": "forum"},
                {"link": "/goto?url=CAESngEC", "displayed_link": "https://www.google.com › x", "snippet": "loop"},
                {"link": "/goto?url=CAESngED", "snippet": "no displayed link"},
                {"link": "/relative/page", "snippet": "not a web URL"},
                {"link": "https://books.google.com/books?id=1", "snippet": "a real Google page"},
            ]
        }
    )

    hits = SerpApiWebSearch(client=client).search("q", limit=10)

    assert [h.url for h in hits] == [
        "https://www.mofa.go.jp/",
        "https://www.my.emb-japan.go.jp/",
        "https://isa.go.jp/",
        "https://books.google.com/books?id=1",
    ]
    assert hits[0].source_date == "Jan 15, 2026"
    assert [classify_source_authority(h.url) for h in hits[:3]] == [SourceAuthority.OFFICIAL] * 3


def test_web_search_error_payload_raises():
    with pytest.raises(SerpApiUnavailableError):
        SerpApiWebSearch(client=FakeClient({"error": "bad"})).search("q", limit=5)


# ── Workflow ────────────────────────────────────────────────────────


def agent_state(message, **overrides):
    values = dict(
        messages=[HumanMessage(content=message)],
        origin_country="Malaysia",
        country="Japan",
        city=["Tokyo"],
    )
    values.update(overrides)
    return AgentState(**values)


def test_workflow_routes_retrieves_and_answers():
    hits = [
        SearchHit(title="MOFA", url="https://www.mofa.go.jp/visa", snippet="Visa-free for 90 days."),
        SearchHit(title="ISA", url="https://www.isa.go.jp/en", snippet="Short stays are visa-free."),
    ]
    fakes = Fakes(
        classification=classified(RequestType.FACTUAL, FactualDomain.VISA_ENTRY),
        hits=hits,
        answer="Visa-free for 90 days.",
    )
    service = fakes.service()
    state = agent_state("Do I need a visa?")

    update = asyncio.run(run_factual_routing(state, service=service))
    state = state.model_copy(update=update)
    assert route_after_factual_router(state) == "retrieval"

    update = asyncio.run(run_factual_retrieval(state, service=service))
    state = state.model_copy(update=update)
    assert state.factual_validation["approved"] is True
    assert route_after_factual_retrieval(state) == "answer"

    update = asyncio.run(run_factual_answer(state, service=service))
    message = update["messages"][0]
    assert message.content == "Visa-free for 90 days."
    assert message.additional_kwargs == {"server_owned": "factual_qa"}


def test_workflow_fails_closed_on_a_broken_route():
    state = agent_state("x", factual_route={"request_type": "nonsense"})

    assert route_after_factual_router(state) == "agent"
    assert route_after_factual_retrieval(state) == "answer"
    update = asyncio.run(run_factual_answer(state, service=Fakes().service()))
    assert update["factual_issue_codes"] == ["fact.route_invalid"]


def test_answer_without_retrieval_counts_as_missing_evidence():
    fakes = Fakes(classification=classified(RequestType.FACTUAL, FactualDomain.VISA_ENTRY))
    state = agent_state("Do I need a visa?")
    state = state.model_copy(update=asyncio.run(run_factual_routing(state, service=fakes.service())))

    update = asyncio.run(run_factual_answer(state, service=fakes.service()))

    assert update["factual_issue_codes"] == ["fact.evidence_missing"]
