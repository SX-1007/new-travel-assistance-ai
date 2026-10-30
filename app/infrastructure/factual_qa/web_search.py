"""Google web search through SerpAPI, returning raw hits for factual Q&A."""

from __future__ import annotations

from typing import Any

from app.infrastructure.search.serpapi import SerpApiClient, SerpApiUnavailableError
from app.modules.factual_qa.models import SearchHit


class SerpApiWebSearch:
    """Answer box first, then organic results. Raises when SerpAPI fails."""

    def __init__(self, *, client: SerpApiClient) -> None:
        self._client = client

    def search(self, query: str, *, limit: int) -> list[SearchHit]:
        # The target location is already part of the query text. SerpAPI's
        # "location" parameter only accepts its own canonical names and
        # rejected the whole search for anything else.
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
    url = _text(raw.get("link"))
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


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


__all__ = [
    "SerpApiWebSearch",
]
