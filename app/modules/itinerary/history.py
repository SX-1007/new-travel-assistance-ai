"""Bounded history of previously accepted itinerary revisions.

The live itinerary changes with every accepted edit. History keeps a small
copy of each earlier accepted revision, so a later question such as "what
was Day 2 activity 3 before the change?" is answered from trusted state, not
from conversation prose.

Entries are stored compact (``compact_itinerary``): names, types, order,
costs, bookings and travel times; no coordinates, images or descriptions.
History only serves read-only questions, and full payloads in up to 20
revisions made every checkpoint write needlessly large.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Final

from app.modules.itinerary.projection import compact_itinerary


MAX_HISTORY_REVISIONS: Final = 20


def history_entry(value: Any) -> dict[str, Any] | None:
    """A normalised ``{"revision", "draft_itinerary"}`` entry, or None.

    Accepts history entries and accepted-plan snapshots (``plan_revision``).
    """

    if not isinstance(value, Mapping):
        return None

    revision = value.get("revision", value.get("plan_revision"))
    itinerary = value.get("draft_itinerary")

    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision <= 0
        or not isinstance(itinerary, list)
        or not itinerary
    ):
        return None

    return {"revision": revision, "draft_itinerary": compact_itinerary(itinerary)}


def append_previous_itinerary(
    history: Iterable[Any] | None,
    accepted_snapshot: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Add one previously accepted revision, without duplicates.

    The first revision is kept permanently, then the newest 19, so the
    original baseline survives while checkpoint growth stays bounded.
    """

    by_revision: dict[int, dict[str, Any]] = {}

    for raw in (*(history or ()), accepted_snapshot):
        entry = history_entry(raw)

        if entry is not None:
            by_revision.setdefault(entry["revision"], entry)

    entries = [by_revision[revision] for revision in sorted(by_revision)]

    if len(entries) > MAX_HISTORY_REVISIONS:
        entries = [entries[0], *entries[-(MAX_HISTORY_REVISIONS - 1):]]

    return entries


def recover_itinerary_history(
    checkpoint_states: Iterable[Any] | None,
    *,
    current_revision: int,
) -> list[dict[str, Any]]:
    """Rebuild history for sessions saved before the history field existed.

    The current revision itself is excluded: history holds earlier ones only.
    """

    if isinstance(current_revision, bool) or not isinstance(current_revision, int):
        return []

    if current_revision <= 1:
        return []

    snapshots: list[Mapping[str, Any]] = []

    for state in checkpoint_states or ():
        if not isinstance(state, Mapping):
            continue

        snapshot = state.get("accepted_plan_snapshot")

        if not isinstance(snapshot, Mapping):
            snapshot = {
                "plan_revision": state.get("plan_revision"),
                "draft_itinerary": state.get("draft_itinerary"),
            }

        revision = snapshot.get("plan_revision")

        if isinstance(revision, int) and not isinstance(revision, bool) and revision < current_revision:
            snapshots.append(snapshot)

    return append_previous_itinerary(snapshots, None)


__all__ = [
    "MAX_HISTORY_REVISIONS",
    "append_previous_itinerary",
    "history_entry",
    "recover_itinerary_history",
]
