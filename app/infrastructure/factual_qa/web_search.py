"""Google web search through SerpAPI, returning raw hits for factual Q&A."""

from __future__ import annotations

import re
from typing import Any, Final
from urllib.parse import ParseResult, urlparse

from app.infrastructure.search.serpapi import SerpApiClient, SerpApiUnavailableError
from app.modules.factual_qa.models import SearchHit


# Google rewrites many result links to google.com/goto (or /url) redirects
# whose target is encrypted, and SerpAPI passes some of them through. Such a
# link hides the real site, so every source would be judged as unknown.
_GOOGLE_HOST: Final = re.compile(r"^(?:www\.)?google\.[a-z]{2,3}(?:\.[a-z]{2})?$")
_GOOGLE_REDIRECT_PATHS: Final = frozenset({"/goto", "/url"})
_HOSTNAME: Final = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+$")


class SerpApiWebSearch:
    """Answer box first, then organic results. Raises when SerpAPI fails."""

    def __init__(self, *, client: SerpApiClient) -> None:
        self._client = client

    def search(self, query: str, *, limit: int, max_age_minutes: int | None = None) -> list[SearchHit]:
        # Google searches the live web and shows each page's date, which
        # factual Q&A judges itself, so the page-age limit needs nothing here.
        # The target location is already part of the query text. SerpAPI's
        # "location" parameter only accepts its own canonical names and
        # rejects the whole search for anything else.
        payload = self._client.search({"engine": "google", "q": query, "hl": "en", "num": limit})

        if payload.get("error"):
            raise SerpApiUnavailableError("SerpAPI returned an error payload.")

        hits: list[SearchHit] = []
        answer_box = payload.get("answer_box")

        if isinstance(answer_box, dict):
            hit = _hit(
                answer_box,
                snippet_keys=("answer", "snippet", "result"),
                title_keys=("title", "type"),
            )
            if hit is not None:
                hits.append(hit)

        for result in payload.get("organic_results") or []:
            if len(hits) >= limit:
                break

            if isinstance(result, dict):
                hit = _hit(result, snippet_keys=("snippet",), title_keys=("title",))
                if hit is not None:
                    hits.append(hit)

        return hits


def _hit(
    raw: dict[str, Any],
    *,
    snippet_keys: tuple[str, ...],
    title_keys: tuple[str, ...],
) -> SearchHit | None:
    url = _destination_url(raw)
    snippet = next((_text(raw.get(key)) for key in snippet_keys if _text(raw.get(key))), "")

    if not url or not snippet:
        return None

    title = next((_text(raw.get(key)) for key in title_keys if _text(raw.get(key))), "")

    return SearchHit(
        title=title,
        url=url,
        snippet=snippet,
        source_name=_text(raw.get("source")),
        source_date=_text(raw.get("date")) or None,
    )


def _destination_url(raw: dict[str, Any]) -> str:
    """The page's own URL, or its site when Google hid it behind a redirect.

    A redirect's target can't be decoded, but the displayed link still names
    the real site ("https://www.mofa.go.jp › visa › ..."). Its path is only a
    breadcrumb, so the site's root is cited instead. A result with neither
    ("5 comments", "179K+ views") is dropped: there is nothing to cite.
    """

    link = _text(raw.get("link"))
    parsed = _parse(link)

    if parsed is None:
        return ""

    if _is_google_redirect(parsed):
        return _site_from_displayed_link(_text(raw.get("displayed_link")))

    return link if parsed.scheme in {"http", "https"} else ""


def _is_google_redirect(parsed: ParseResult) -> bool:
    host = (parsed.hostname or "").lower()
    return parsed.path in _GOOGLE_REDIRECT_PATHS and (not host or bool(_GOOGLE_HOST.match(host)))


def _site_from_displayed_link(displayed: str) -> str:
    head = displayed.split("›", 1)[0].strip()

    if not head or " " in head:
        return ""

    parsed = _parse(head if "://" in head else f"https://{head}")

    if parsed is None or parsed.scheme not in {"http", "https"}:
        return ""

    host = (parsed.hostname or "").lower()

    if not _HOSTNAME.match(host) or _GOOGLE_HOST.match(host):
        return ""

    return f"{parsed.scheme}://{host}/"


def _parse(url: str) -> ParseResult | None:
    if not url:
        return None

    try:
        return urlparse(url)
    except ValueError:
        return None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


__all__ = [
    "SerpApiWebSearch",
]
