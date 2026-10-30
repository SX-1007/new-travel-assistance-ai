"""Per-request context the graph nodes share: who is asking and their profile.

The old graph fetched the long-term profile (a Supabase read plus a pgvector
query) separately in the agent, planning and reply nodes, and again on every
pass of the agent ↔ tools loop. ``ProfileLoader`` fetches it once per short
window and is invalidated when memory extraction saves something new.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, Final

from cachetools import TTLCache

from app.orchestration.workflows.planning import log_context_from_config
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)

# ``config["configurable"]`` key the routers put the authenticated user under.
USER_ID_KEY: Final = "__user_id"

_PROFILE_TTL_SECONDS: Final = 60.0
_PROFILE_CACHE_USERS: Final = 512


def user_id_from_config(config: Mapping[str, Any] | None) -> str | None:
    configurable = config.get("configurable") if isinstance(config, Mapping) else None
    user_id = configurable.get(USER_ID_KEY) if isinstance(configurable, Mapping) else None
    return user_id if isinstance(user_id, str) and user_id.strip() else None


def node_log_context(config: Mapping[str, Any] | None) -> LogContext:
    return log_context_from_config(config)


class ProfileLoader:
    """Long-term user profile, fetched at most once per user per short window.

    A failed fetch is not cached and never fails the turn: the nodes simply
    plan and reply without personalisation.
    """

    def __init__(
        self,
        fetch: Callable[[str], Mapping[str, Any]],
        *,
        ttl_seconds: float = _PROFILE_TTL_SECONDS,
        max_users: int = _PROFILE_CACHE_USERS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch
        # Touched only from the event loop thread; the fetch itself runs in a
        # worker thread and its result is stored after the await.
        self._cache: TTLCache[str, dict[str, Any]] = TTLCache(
            maxsize=max_users,
            ttl=ttl_seconds,
            timer=clock,
        )

    async def load(
        self,
        user_id: str | None,
        *,
        log_context: LogContext | None = None,
    ) -> dict[str, Any] | None:
        if not user_id:
            return None

        cached = self._cache.get(user_id)

        if cached is not None:
            return cached

        try:
            profile = dict(await asyncio.to_thread(self._fetch, user_id) or {})
        except Exception:
            log_event(
                logger,
                "profile.load",
                context=log_context,
                stage="profile",
                outcome="unavailable",
                issue_codes=["profile.unavailable"],
            )
            return None

        self._cache[user_id] = profile
        return profile

    def invalidate(self, user_id: str) -> None:
        """Drop the cached profile, e.g. after new preferences were saved."""

        self._cache.pop(user_id, None)


__all__ = [
    "USER_ID_KEY",
    "ProfileLoader",
    "node_log_context",
    "user_id_from_config",
]
