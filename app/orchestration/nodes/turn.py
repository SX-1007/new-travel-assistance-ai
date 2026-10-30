"""Start of a chat turn: reset turn-scoped state, queue memory extraction.

The old ``memory_extraction_node`` claimed to be fire-and-forget but awaited
the extraction (an LLM call plus two database writes) before the turn could
continue, on a module-level thread pool created at import. Extraction now
runs in the background with a bound on concurrent jobs; the current turn
already sees the user's words in the conversation, so it loses nothing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from typing import Any, Final

from langchain_core.runnables import RunnableConfig

from app.orchestration.messages import message_text
from app.orchestration.nodes.context import node_log_context, user_id_from_config
from app.orchestration.nodes.transaction import planning_reset
from app.orchestration.state import AgentState
from app.orchestration.workflows.factual_qa import empty_factual_state
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)

_MAX_IN_FLIGHT: Final = 4


class MemoryExtractionQueue:
    """Background long-term-memory extraction with bounded concurrency.

    ``extract(user_id, message)`` is the blocking extraction task (it returns
    ``{"memory_updated": bool}``). When it saved something, ``on_saved`` runs
    with the user id, e.g. to drop a cached profile. Jobs beyond the bound
    are skipped and logged, never queued without limit.
    """

    def __init__(
        self,
        extract: Callable[[str, str], Mapping[str, Any] | None],
        *,
        on_saved: Callable[[str], None] | None = None,
        max_in_flight: int = _MAX_IN_FLIGHT,
    ) -> None:
        self._extract = extract
        self._on_saved = on_saved
        self._max_in_flight = max_in_flight
        # Strong references: the event loop keeps only weak ones to tasks.
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def in_flight(self) -> int:
        return len(self._tasks)

    def submit(self, user_id: str, message: str, *, log_context: LogContext | None = None) -> bool:
        """Start extraction without waiting; False when skipped."""

        if len(self._tasks) >= self._max_in_flight:
            _log(log_context, "skipped", "memory.queue_full")
            return False

        task = asyncio.get_running_loop().create_task(self._run(user_id, message, log_context))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def drain(self, timeout: float) -> None:
        """Wait up to ``timeout`` seconds for running jobs (at shutdown)."""

        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=timeout)

    async def _run(self, user_id: str, message: str, log_context: LogContext | None) -> None:
        try:
            result = await asyncio.to_thread(self._extract, user_id, message)
        except Exception:
            _log(log_context, "unavailable", "memory.persistence_failed")
            return

        saved = isinstance(result, Mapping) and bool(result.get("memory_updated"))

        if saved and self._on_saved is not None:
            try:
                self._on_saved(user_id)
            except Exception:
                _log(log_context, "failed", "memory.on_saved_failed")

        _log(log_context, "saved" if saved else "unchanged")


async def begin_chat_turn_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    memory: MemoryExtractionQueue | None = None,
) -> dict[str, Any]:
    """Reset turn-scoped state and queue memory extraction for the message.

    Any transaction a previous turn left unfinished is discarded, not only a
    failed one: the old reset kept a "validated" candidate from an
    interrupted turn, which the next tool call could then publish. A
    finished budget gate's outcome is cleared too (no recommendation is
    pending on this path), so it can't be shown again over a later reply.
    """

    user_id = user_id_from_config(config)
    last = state.messages[-1] if state.messages else None

    if memory is not None and user_id and last is not None and last.type == "human":
        text = message_text(last.content).strip()

        if text:
            memory.submit(user_id, text, log_context=node_log_context(config))

    return {
        **empty_factual_state(),
        **planning_reset(),
        "plan_overrides": None,
        "post_tool_outcome": None,
        "budget_gate_outcome": None,
        "budget_gate_reason": None,
        "budget_gate_message": None,
    }


def _log(log_context: LogContext | None, outcome: str, code: str | None = None) -> None:
    log_event(
        logger,
        "memory.extraction",
        context=log_context,
        stage="memory",
        outcome=outcome,
        issue_codes=[code] if code else [],
    )


__all__ = [
    "MemoryExtractionQueue",
    "begin_chat_turn_node",
]
