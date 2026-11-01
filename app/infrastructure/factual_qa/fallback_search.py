"""One web search tried before another, for factual Q&A."""

from __future__ import annotations

import logging

from app.modules.factual_qa.models import SearchHit
from app.modules.factual_qa.ports import WebSearchPort
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)


class FallbackWebSearch:
    """The primary search; the backup only when it fails or finds nothing.

    The backup is never called alongside the primary, so a working primary
    spends none of the backup's quota. A backup failure is raised as usual.
    """

    def __init__(self, *, primary: WebSearchPort, backup: WebSearchPort) -> None:
        self._primary = primary
        self._backup = backup

    def search(self, query: str, *, limit: int, max_age_minutes: int | None = None) -> list[SearchHit]:
        try:
            hits = self._primary.search(query, limit=limit, max_age_minutes=max_age_minutes)
        except Exception:
            # The primary's own client logs why it failed.
            safe_log(logger, logging.INFO, "factual.search.backup_used.primary_failed")
        else:
            if hits:
                return hits

            safe_log(logger, logging.INFO, "factual.search.backup_used.primary_empty")

        return self._backup.search(query, limit=limit, max_age_minutes=max_age_minutes)


__all__ = [
    "FallbackWebSearch",
]
