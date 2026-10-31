"""Memory-specific observability policy.

Memory owns the severity of its own events. Generic safe log emission lives in
``app.shared.logging`` so memory does not depend on provider/tool modules.
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Final, Mapping

from app.shared.logging import safe_log


_MEMORY_EVENT_LEVELS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "memory.extractor_chain_ready": logging.INFO,
        "memory.extractor.relational_failed": logging.ERROR,
        "memory.extractor.vector_failed": logging.ERROR,
        "memory.extractor.trivial": logging.DEBUG,
        "memory.extractor.completed": logging.DEBUG,
        "memory.extractor.failed": logging.ERROR,
        "memory.extractor.retry": logging.WARNING,
        "memory.extractor.started": logging.INFO,
        "memory.extractor.no_traits": logging.INFO,
        "memory.extractor.home_country_rejected": logging.WARNING,
        "memory.extractor.relational_queued": logging.INFO,
        "memory.extractor.vector_queued": logging.INFO,
        "memory.extractor.persistence_succeeded": logging.INFO,
        "memory.extractor.persistence_failed": logging.WARNING,
        "memory.extractor.persistence_raised": logging.ERROR,
    }
)


def log_memory_event(
    target_logger: logging.Logger,
    event: str,
) -> None:
    """Emit an allowlisted memory event without affecting memory execution."""

    level = _MEMORY_EVENT_LEVELS.get(event)

    if level is None:
        return

    safe_log(
        target_logger,
        level,
        event,
    )


__all__ = [
    "log_memory_event",
]