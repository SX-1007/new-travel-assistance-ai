"""Web search through Exa, returning raw hits for factual Q&A.

Exa returns each page's own URL (not a Google redirect) and passages chosen
for the query, which often state the rule itself rather than a page teaser.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Final
from urllib.parse import urlparse

from app.infrastructure.search.exa import ExaClient
from app.modules.factual_qa.models import SearchHit


# "auto" costs $7 per 1,000 searches, about 1,400 a month on the free
# credit; passages are included at no extra charge. "lite" ($4) missed the
# embassy page that states the rule for the step-21 visa question.
_SEARCH_TYPE: Final = "auto"
# Up to 10 results cost the base price; each one beyond is billed extra.
_MAX_RESULTS: Final = 10
# A hint per page; Exa may send more. Factual Q&A keeps 600 characters of
# each snippet, so asking for more would only add response size.
_HIGHLIGHT_CHARACTERS: Final = 600
# Exa's upper bound for maxAgeHours (30 days).
_MAX_AGE_HOURS_LIMIT: Final = 720
_HIGHLIGHT_SEPARATOR: Final = " … "
_ISO_DATE: Final = re.compile(r"^(\d{4}-\d{2}-\d{2})")


class ExaWebSearch:
    """Exa search with passages. Raises when Exa fails or is cooling down."""

    def __init__(self, *, client: ExaClient) -> None:
        self._client = client

    def search(self, query: str, *, limit: int, max_age_minutes: int | None = None) -> list[SearchHit]:
        if limit < 1:
            return []

        contents: dict[str, Any] = {"highlights": {"maxCharacters": _HIGHLIGHT_CHARACTERS}}

        # Exa serves pages from its cache. A live question (weather, transit)
        # must not be answered from a copy older than its page-age limit, and
        # a cached copy carries no date that would show it. Whole hours,
        # rounded down, so the copy is never older than allowed; under an
        # hour means "fetch every page now". No limit keeps Exa's cache.
        if max_age_minutes is not None:
            contents["maxAgeHours"] = min(max(max_age_minutes, 0) // 60, _MAX_AGE_HOURS_LIMIT)

        payload = self._client.search(
            {
                "query": query,
                "type": _SEARCH_TYPE,
                "numResults": min(limit, _MAX_RESULTS),
                "contents": contents,
            }
        )

        hits: list[SearchHit] = []

        for result in payload["results"]:
            if len(hits) >= limit:
                break

            if isinstance(result, dict):
                hit = _hit(result)
                if hit is not None:
                    hits.append(hit)

        return hits


def _hit(raw: dict[str, Any]) -> SearchHit | None:
    url = _web_url(_text(raw.get("url")))
    highlights = raw.get("highlights")
    passages = (_text(item) for item in highlights) if isinstance(highlights, list) else ()
    snippet = _HIGHLIGHT_SEPARATOR.join(text for text in passages if text)

    if not url or not snippet:
        return None

    return SearchHit(
        title=_text(raw.get("title")),
        url=url,
        snippet=snippet,
        source_date=_published_date(raw.get("publishedDate")),
    )


def _web_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""

    return url if parsed.scheme in {"http", "https"} and parsed.hostname else ""


def _published_date(value: Any) -> str | None:
    """The calendar date only ("2025-09-01").

    Exa estimates when a page was written and most of its times are a bare
    midnight, so the date is all it really knows. Factual Q&A reads this
    format and allows a calendar date at least a day.
    """

    match = _ISO_DATE.match(_text(value))

    if match is None:
        return None

    try:
        return date.fromisoformat(match.group(1)).isoformat()
    except ValueError:
        return None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


__all__ = [
    "ExaWebSearch",
]
