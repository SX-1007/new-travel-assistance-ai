"""Exa search adapter and the Exa-then-SerpAPI fallback, with fakes."""

import pytest

from app.infrastructure.factual_qa.exa_search import ExaWebSearch
from app.infrastructure.factual_qa.fallback_search import FallbackWebSearch
from app.infrastructure.search.exa import ExaCoolingDown, ExaUnavailableError
from app.modules.factual_qa.models import SearchHit, SourceAuthority
from app.modules.factual_qa.retrieval import classify_source_authority


class FakeExaClient:
    def __init__(self, results=()):
        self.results = list(results)
        self.bodies = []

    def search(self, body):
        self.bodies.append(body)
        return {"results": self.results}


def result(url, *highlights, title="T", published=None):
    return {"url": url, "title": title, "highlights": list(highlights), "publishedDate": published}


# ── Request ─────────────────────────────────────────────────────────


def test_asks_for_passages_within_the_base_price():
    client = FakeExaClient()

    ExaWebSearch(client=client).search("Japan visa requirements", limit=25)

    assert client.bodies == [
        {
            "query": "Japan visa requirements",
            "type": "auto",
            "numResults": 10,
            "contents": {"highlights": {"maxCharacters": 600}},
        }
    ]


@pytest.mark.parametrize(
    ("minutes", "hours"),
    [(0, 0), (5, 0), (30, 0), (59, 0), (60, 1), (180, 3), (24 * 60, 24), (10**6, 720), (-3, 0)],
)
def test_page_age_limit_becomes_whole_hours_rounded_down(minutes, hours):
    client = FakeExaClient()

    ExaWebSearch(client=client).search("q", limit=10, max_age_minutes=minutes)

    assert client.bodies[0]["contents"]["maxAgeHours"] == hours


def test_no_page_age_limit_keeps_exa_cache():
    client = FakeExaClient()

    ExaWebSearch(client=client).search("q", limit=10, max_age_minutes=None)

    assert "maxAgeHours" not in client.bodies[0]["contents"]


def test_no_results_wanted_makes_no_request():
    client = FakeExaClient()

    assert ExaWebSearch(client=client).search("q", limit=0) == []
    assert client.bodies == []


def test_client_failure_is_raised():
    class Down:
        def search(self, body):
            raise ExaUnavailableError("down")

    with pytest.raises(ExaUnavailableError):
        ExaWebSearch(client=Down()).search("q", limit=10)


# ── Parsing ─────────────────────────────────────────────────────────


def test_parses_real_page_urls_passages_and_dates():
    client = FakeExaClient(
        [
            result(
                "https://www.my.emb-japan.go.jp/itpr_en/visainfo_c.html",
                " Visa exemption for Malaysia nationals ",
                "",
                "holding biometric passports.",
                title=" Visa Information ",
                published="2023-08-23T00:00:00.000Z",
            ),
            result("https://www.mofa.go.jp/j_info/visit/visa/", "Short-term stay.", published="not a date"),
        ]
    )

    hits = ExaWebSearch(client=client).search("q", limit=10)

    assert hits == [
        SearchHit(
            title="Visa Information",
            url="https://www.my.emb-japan.go.jp/itpr_en/visainfo_c.html",
            snippet="Visa exemption for Malaysia nationals … holding biometric passports.",
            source_date="2023-08-23",
        ),
        SearchHit(title="T", url="https://www.mofa.go.jp/j_info/visit/visa/", snippet="Short-term stay."),
    ]
    # The page itself is cited, and its host is judged as before.
    assert classify_source_authority(hits[0].url) is SourceAuthority.OFFICIAL


def test_drops_results_with_nothing_to_cite_or_quote():
    client = FakeExaClient(
        [
            "junk",
            {"title": "no url", "highlights": ["x"]},
            result("ftp://files.example.com/x", "x"),
            result("javascript:alert(1)", "x"),
            result("https://", "x"),
            result("https://a.go.jp/no-passages"),
            {"url": "https://b.go.jp", "highlights": "not a list"},
            {"url": "https://c.go.jp", "highlights": [None, 5, "  "]},
            result("https://d.go.jp/kept", "kept"),
        ]
    )

    hits = ExaWebSearch(client=client).search("q", limit=10)

    assert [hit.url for hit in hits] == ["https://d.go.jp/kept"]


@pytest.mark.parametrize(
    ("published", "expected"),
    [
        ("2026-11-01T07:21:38.773Z", "2026-11-01"),
        ("2025-09-01", "2025-09-01"),
        ("2025-02-30T00:00:00Z", None),
        ("Sept 2025", None),
        (None, None),
        (20250901, None),
    ],
)
def test_keeps_only_a_valid_calendar_date(published, expected):
    client = FakeExaClient([result("https://a.go.jp", "x", published=published)])

    assert ExaWebSearch(client=client).search("q", limit=10)[0].source_date == expected


def test_returns_at_most_limit_hits():
    client = FakeExaClient([result(f"https://{n}.go.jp", "x") for n in "abcdef"])

    assert len(ExaWebSearch(client=client).search("q", limit=3)) == 3
    assert client.bodies[0]["numResults"] == 3


# ── Fallback ────────────────────────────────────────────────────────


class FakeSearch:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def search(self, query, *, limit, max_age_minutes=None):
        self.calls.append((query, limit, max_age_minutes))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


HIT = SearchHit(title="t", url="https://a.go.jp", snippet="s")


def test_backup_is_not_used_while_the_primary_answers():
    primary, backup = FakeSearch([HIT]), FakeSearch([])

    hits = FallbackWebSearch(primary=primary, backup=backup).search("q", limit=10, max_age_minutes=30)

    assert hits == [HIT]
    assert primary.calls == [("q", 10, 30)]
    assert backup.calls == []


@pytest.mark.parametrize(
    "primary_outcome",
    [ExaUnavailableError("402"), ExaCoolingDown("no credit"), RuntimeError("bug"), []],
)
def test_backup_answers_when_the_primary_fails_or_finds_nothing(primary_outcome):
    backup = FakeSearch([HIT])

    hits = FallbackWebSearch(primary=FakeSearch(primary_outcome), backup=backup).search(
        "q", limit=10, max_age_minutes=5
    )

    assert hits == [HIT]
    assert backup.calls == [("q", 10, 5)]


def test_backup_failure_is_raised():
    fallback = FallbackWebSearch(primary=FakeSearch([]), backup=FakeSearch(RuntimeError("serpapi down")))

    with pytest.raises(RuntimeError, match="serpapi down"):
        fallback.search("q", limit=10)
