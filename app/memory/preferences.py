"""Pure long-term-memory preference rules.

No database, network, LLM, or orchestration imports are allowed here.
"""

from __future__ import annotations

from typing import Any, List


MEMORY_LIST_FIELDS = frozenset(
    {
        "dietary_restrictions",
        "interests",
        "accommodation_preferences",
    }
)

MEM0_TO_PROFILE_FIELD = {
    "dietary": "dietary_restrictions",
    "interest": "interests",
    "accommodation": "accommodation_preferences",
}


def merge_unique_memory_values(
    existing: Any,
    incoming: Any,
) -> List[str]:
    """Merge list-style memory values without case-insensitive duplicates."""
    existing_values = (
        existing
        if isinstance(existing, list)
        else ([existing] if existing else [])
    )

    incoming_values = (
        incoming
        if isinstance(incoming, list)
        else ([incoming] if incoming else [])
    )

    merged: List[str] = []
    seen: set[str] = set()

    for value in [
        *existing_values,
        *incoming_values,
    ]:
        text = str(value).strip()

        if not text:
            continue

        key = text.casefold()

        if key in seen:
            continue

        seen.add(key)
        merged.append(text)

    return merged