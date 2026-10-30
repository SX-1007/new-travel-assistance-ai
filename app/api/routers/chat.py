"""Chat endpoints: one turn on an existing trip, and restored history.

POST /message runs one turn through ``GraphRuntime.invoke_chat``, which
allows one turn per session at a time (a second one is answered 409) and
checks ownership. GET /history lists the user's sessions from their latest
checkpoints. Exceptions map to HTTP statuses in ``api/errors.py``.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, Response

from app.api.dependencies import CurrentUser, RequestId
from app.api.presenters.chat import ChatTurnResponse, chat_turn_response, history_session
from app.api.providers import ChatHistoryLoader, HistoryLoader, Reviewer, Runtime
from app.api.schemas.chat import ChatHistoryResponse, ChatHistorySession, ChatRequest, ChatResponse
from app.modules.output_review.service import OutputReviewService
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Send a message to the travel-planning agent",
)
async def send_chat_message(
    chat: ChatRequest,
    user_id: CurrentUser,
    request_id: RequestId,
    runtime: Runtime,
) -> ChatTurnResponse:
    log_context = LogContext(request_id=request_id, session_id=chat.session_id)

    result = await runtime.invoke_chat(
        chat.user_message,
        thread_id=chat.session_id,
        user_id=user_id,
        budget_action=chat.budget_action,
        budget_assessment_id=chat.budget_assessment_id,
        request_id=request_id,
    )
    response = await asyncio.to_thread(chat_turn_response, result)

    log_event(
        logger,
        "chat.message",
        context=log_context,
        stage="chat",
        outcome="completed" if response.status == "success" else response.status,
    )
    return response


@router.get(
    "/history",
    response_model=ChatHistoryResponse,
    summary="List the current user's saved chat sessions",
)
async def get_chat_history(
    user_id: CurrentUser,
    response: Response,
    load: ChatHistoryLoader,
    reviewer: Reviewer,
    limit: int = Query(default=10, ge=1, le=20),
) -> ChatHistoryResponse:
    # Per-user data: never cached by the browser or a shared proxy.
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Vary"] = "X-User-ID"

    sessions = await asyncio.to_thread(_load_sessions, load, reviewer, user_id, limit)
    return ChatHistoryResponse(sessions=sessions)


def _load_sessions(
    load: HistoryLoader,
    reviewer: OutputReviewService,
    user_id: str,
    limit: int,
) -> list[ChatHistorySession]:
    # Reading and projecting both block (database, validator): one thread.
    return [
        session
        for record in load(user_id, limit)
        if (session := history_session(record, reviewer)) is not None
    ]
