"""The compiled travel-assistant graph and the two ways a request runs it.

    New trip:  START → process_initial_form → execute_validated_transaction
               → generate_reviewed_response → promote_candidate / END
    Chat:      START → begin_chat_turn → factual_router → (factual nodes | agent)
               agent ⇄ tools → post_tool_processing → replan / review / gate / agent
    Budget:    START → assess_chat_budget | resume_accepted_budget
               → execute_validated_transaction … | budget_gate_response

``build_graph`` only wires nodes it is given; ``GraphRuntime`` checks session
ownership, runs one turn under a time limit and returns a ``TurnResult``.
Nothing here builds clients or reads settings: ``bootstrap/container.py``
assembles the services, the checkpointer and the runtime.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol

from cachetools import LRUCache
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.modules.itinerary.history import recover_itinerary_history
from app.orchestration import router
from app.orchestration.nodes.agent import AgentToolset
from app.orchestration.nodes.context import USER_ID_KEY
from app.orchestration.session_access import SessionAccessService
from app.orchestration.state import AgentState
from app.shared.exceptions import InvalidTurnRequest, SessionAccessDenied, SessionBusy
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)

RECURSION_LIMIT: Final = 50
CHAT_TIMEOUT_SECONDS: Final = 240.0
NEW_TRIP_TIMEOUT_SECONDS: Final = 480.0

_MAX_ID_LENGTH: Final = 128
_BUDGET_ACTIONS: Final = frozenset({"accept_recommended"})
_RELEASE_ATTEMPTS: Final = 3
# Checkpoints scanned to rebuild history for sessions that predate it.
_HISTORY_SCAN_LIMIT: Final = 200
_HISTORY_SCANNED_SESSIONS: Final = 1024


# ── Graph ───────────────────────────────────────────────────────────


def build_graph(nodes: Mapping[str, Callable[..., Any]], toolset: AgentToolset) -> StateGraph:
    """Wire ``nodes`` (from ``nodes.build_nodes``) and the two tool nodes."""

    graph = StateGraph(AgentState)

    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.add_node("tools", ToolNode(list(toolset.agent)))
    graph.add_node("budget_decision_tools", ToolNode(list(toolset.budget_decision)))

    graph.add_conditional_edges(START, router.route_start)

    # Planning: the form node clears the transaction itself.
    graph.add_edge("process_initial_form", "execute_validated_transaction")
    graph.add_edge("prepare_planning_transaction", "execute_validated_transaction")
    graph.add_edge("execute_validated_transaction", "generate_reviewed_response")
    graph.add_conditional_edges("generate_reviewed_response", router.route_after_review)
    graph.add_edge("promote_candidate", END)

    # Chat.
    graph.add_edge("begin_chat_turn", "factual_router")
    graph.add_conditional_edges("factual_router", router.route_after_factual_router)
    graph.add_conditional_edges("factual_retrieval", router.route_after_factual_retrieval)
    graph.add_edge("factual_answer", END)
    graph.add_conditional_edges("agent", router.route_after_agent)
    graph.add_edge("tools", "post_tool_processing")
    graph.add_conditional_edges(
        "post_tool_processing",
        router.route_after_post_tool,
        sorted(set(router.POST_TOOL_ROUTES.values())),
    )

    # Budget.
    graph.add_conditional_edges("budget_decision_agent", router.route_after_budget_decision)
    graph.add_edge("budget_decision_tools", "post_tool_processing")
    graph.add_conditional_edges("assess_chat_budget", router.route_after_budget)
    graph.add_conditional_edges("resume_accepted_budget", router.route_after_budget)
    graph.add_edge("budget_gate_response", END)

    return graph


def compile_graph(
    nodes: Mapping[str, Callable[..., Any]],
    toolset: AgentToolset,
    *,
    checkpointer: Any,
) -> Any:
    return build_graph(nodes, toolset).compile(checkpointer=checkpointer)


# ── Runtime ─────────────────────────────────────────────────────────


class CheckpointReader(Protocol):
    async def aget_tuple(self, config: Any) -> Any: ...

    def alist(self, config: Any, *, limit: int | None = None) -> Any: ...


class Workflow(Protocol):
    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class TurnResult:
    """The state after one turn, and whether a new plan revision was published.

    The old invoker mixed ``_itinerary_modified`` into the state dict.
    """

    state: dict[str, Any]
    plan_changed: bool


class GraphRuntime:
    """Runs one authenticated turn on one session at a time."""

    def __init__(
        self,
        *,
        workflow: Workflow,
        checkpointer: CheckpointReader,
        sessions: SessionAccessService,
        chat_timeout: float = CHAT_TIMEOUT_SECONDS,
        new_trip_timeout: float = NEW_TRIP_TIMEOUT_SECONDS,
        recursion_limit: int = RECURSION_LIMIT,
    ) -> None:
        self._workflow = workflow
        self._checkpointer = checkpointer
        self._sessions = sessions
        self._chat_timeout = chat_timeout
        self._new_trip_timeout = new_trip_timeout
        self._recursion_limit = recursion_limit
        # One lock per active session; entries vanish when no turn holds one.
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._history_scanned: LRUCache[str, bool] = LRUCache(maxsize=_HISTORY_SCANNED_SESSIONS)
        self._background: set[asyncio.Task[None]] = set()

    # ── New trip ────────────────────────────────────────────────────

    async def invoke_new_trip(
        self,
        initial_state: Mapping[str, Any],
        *,
        thread_id: str,
        user_id: str,
        request_id: str | None = None,
    ) -> TurnResult:
        """Plan a new trip on a session id this user reserves now."""

        _require_id(thread_id, "thread_id")
        _require_id(user_id, "user_id")
        config = self._config(thread_id, user_id, request_id)
        log_context = LogContext(request_id=request_id, session_id=thread_id)

        async with self._turn(thread_id):
            existing = await self._checkpointer.aget_tuple(_thread(thread_id))

            if existing is not None:
                # An earlier attempt with this id wrote its plan but never
                # completed the reservation: hand that result back once.
                if await self._sessions.finalize_expired_new_session_reservation(thread_id, user_id):
                    values = _channel_values(existing)

                    if values is None:
                        raise SessionAccessDenied

                    return TurnResult(state=values, plan_changed=_revision(values) > 0)

                raise SessionAccessDenied

            await self._reserve(thread_id, user_id)

            try:
                state = await self._run(dict(initial_state), config, self._new_trip_timeout)

                if not await self._sessions.complete_new_session_reservation(thread_id, user_id):
                    raise SessionAccessDenied

            except BaseException:
                await self._release_reservation(thread_id, user_id, log_context)
                raise

        return TurnResult(state=state, plan_changed=_revision(state) > 0)

    # ── Chat ────────────────────────────────────────────────────────

    async def invoke_chat(
        self,
        message: str,
        *,
        thread_id: str,
        user_id: str,
        budget_action: str | None = None,
        budget_assessment_id: str | None = None,
        request_id: str | None = None,
    ) -> TurnResult:
        """Run one chat turn on a session this user owns."""

        _require_id(thread_id, "thread_id")
        _require_id(user_id, "user_id")

        if not isinstance(message, str) or not message.strip():
            raise InvalidTurnRequest("message must not be empty")

        if budget_action is not None and budget_action not in _BUDGET_ACTIONS:
            raise InvalidTurnRequest("unsupported budget_action")

        config = self._config(thread_id, user_id, request_id)
        log_context = LogContext(request_id=request_id, session_id=thread_id)

        async with self._turn(thread_id):
            # Both reads are needed; run them together.
            verified, existing = await asyncio.gather(
                self._sessions.verify_session_owner(thread_id, user_id),
                self._checkpointer.aget_tuple(_thread(thread_id)),
            )

            if existing is None:
                raise SessionAccessDenied

            if not verified:
                await self._adopt_legacy_owner(existing, thread_id, user_id)

            before = _channel_values(existing) or {}
            turn_input: dict[str, Any] = {
                "messages": [HumanMessage(content=message)],
                "chat_budget_action": budget_action,
                "chat_budget_assessment_id": budget_assessment_id,
            }
            history = await self._legacy_history(thread_id, before, log_context)

            if history:
                turn_input["itinerary_history"] = history

            state = await self._run(turn_input, config, self._chat_timeout)

        return TurnResult(state=state, plan_changed=_revision(state) > _revision(before))

    # ── Lifecycle ───────────────────────────────────────────────────

    async def drain(self, timeout: float) -> None:
        """Wait up to ``timeout`` for background reservation releases."""

        if self._background:
            await asyncio.wait(set(self._background), timeout=timeout)

    # ── Internals ───────────────────────────────────────────────────

    def _config(self, thread_id: str, user_id: str, request_id: str | None) -> RunnableConfig:
        # ``metadata.user_id`` is persisted with every checkpoint: chat
        # history lists a user's sessions by it.
        configurable: dict[str, Any] = {"thread_id": thread_id, USER_ID_KEY: user_id}
        metadata: dict[str, Any] = {"thread_id": thread_id, "user_id": user_id}

        if request_id:
            configurable["request_id"] = metadata["request_id"] = request_id

        return {"configurable": configurable, "metadata": metadata, "recursion_limit": self._recursion_limit}

    def _turn(self, thread_id: str) -> _SessionTurn:
        lock = self._locks.get(thread_id)

        if lock is None:
            lock = asyncio.Lock()
            self._locks[thread_id] = lock

        return _SessionTurn(lock)

    async def _run(self, turn_input: dict[str, Any], config: RunnableConfig, timeout: float) -> dict[str, Any]:
        # A hung provider call surfaces as TimeoutError (the API maps it to
        # 504) instead of freezing the request.
        result = await asyncio.wait_for(self._workflow.ainvoke(turn_input, config=config), timeout=timeout)
        return dict(result) if isinstance(result, Mapping) else {}

    async def _reserve(self, thread_id: str, user_id: str) -> None:
        if await self._sessions.reserve_new_session(thread_id, user_id):
            return

        # Taken: by a concurrent initialisation that already wrote, or by our
        # own earlier attempt whose lease has expired.
        if await self._checkpointer.aget_tuple(_thread(thread_id)) is not None:
            raise SessionAccessDenied

        if not await self._sessions.reclaim_expired_new_session_reservation(thread_id, user_id):
            raise SessionAccessDenied

    async def _adopt_legacy_owner(self, existing: Any, thread_id: str, user_id: str) -> None:
        """Sessions from before the ownership table: trust the server-written
        checkpoint metadata once, then record the owner."""

        metadata = getattr(existing, "metadata", None)
        owner = metadata.get("user_id") if isinstance(metadata, Mapping) else None

        if owner != user_id or not await self._sessions.backfill_existing_session_owner(thread_id, user_id):
            raise SessionAccessDenied

    async def _release_reservation(self, thread_id: str, user_id: str, log_context: LogContext) -> None:
        """Release a failed initialisation's lease, even while being cancelled."""

        task = asyncio.ensure_future(self._release_with_retries(thread_id, user_id, log_context))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Cancelled again: the shielded release carries on by itself.
            pass

    async def _release_with_retries(self, thread_id: str, user_id: str, log_context: LogContext) -> None:
        for attempt in range(1, _RELEASE_ATTEMPTS + 1):
            try:
                await self._sessions.release_new_session_reservation(thread_id, user_id)
                return
            except Exception:
                if attempt < _RELEASE_ATTEMPTS:
                    await asyncio.sleep(0.1 * 2**attempt)

        log_event(
            logger,
            "session.release",
            context=log_context,
            stage="session_cleanup",
            outcome="error",
            attempt=_RELEASE_ATTEMPTS,
            issue_codes=["session.cleanup_exhausted"],
        )

    async def _legacy_history(
        self,
        thread_id: str,
        values: Mapping[str, Any],
        log_context: LogContext,
    ) -> list[dict[str, Any]]:
        """Rebuild revision history for a session that predates the field.

        Scanned at most once per session per process: the old invoker
        re-read up to 200 checkpoints on every turn when nothing was found.
        """

        revision = _revision(values)

        if revision <= 1 or values.get("itinerary_history") or thread_id in self._history_scanned:
            return []

        self._history_scanned[thread_id] = True
        states: list[Mapping[str, Any]] = []

        try:
            async for saved in self._checkpointer.alist(_thread(thread_id), limit=_HISTORY_SCAN_LIMIT):
                saved_values = _channel_values(saved)

                if saved_values is not None:
                    states.append(saved_values)

            history = recover_itinerary_history(states, current_revision=revision)

        except Exception:
            # Helpful context only; never a reason to fail the turn.
            log_event(
                logger,
                "session.history_recovery",
                context=log_context,
                stage="storage",
                outcome="failed",
                issue_codes=["history.recovery_failed"],
            )
            return []

        if history:
            log_event(logger, "session.history_recovery", context=log_context, stage="storage", outcome="completed")

        return history


class _SessionTurn:
    """Holds a session's lock for one turn; a second turn is refused, not queued."""

    def __init__(self, lock: asyncio.Lock) -> None:
        self._lock = lock

    async def __aenter__(self) -> None:
        # Two concurrent turns on one thread would both start from the same
        # checkpoint and overwrite each other's writes. Checked and taken
        # with no await in between, so this is atomic on the event loop.
        if self._lock.locked():
            raise SessionBusy

        await self._lock.acquire()

    async def __aexit__(self, *exc_info: Any) -> None:
        self._lock.release()


def _thread(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _channel_values(saved: Any) -> dict[str, Any] | None:
    checkpoint = getattr(saved, "checkpoint", None)
    values = checkpoint.get("channel_values") if isinstance(checkpoint, Mapping) else None
    return dict(values) if isinstance(values, Mapping) else None


def _revision(values: Mapping[str, Any]) -> int:
    revision = values.get("plan_revision", 0)
    return revision if isinstance(revision, int) and not isinstance(revision, bool) and revision > 0 else 0


def _require_id(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_ID_LENGTH:
        raise InvalidTurnRequest(f"{name} must be a non-empty string of at most {_MAX_ID_LENGTH} characters")


__all__ = [
    "CHAT_TIMEOUT_SECONDS",
    "NEW_TRIP_TIMEOUT_SECONDS",
    "RECURSION_LIMIT",
    "GraphRuntime",
    "TurnResult",
    "build_graph",
    "compile_graph",
]
