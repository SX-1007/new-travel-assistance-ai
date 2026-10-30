"""Read-only helpers over LangGraph message history.

Shared by the graph nodes, workflows and prompt building; nothing here
mutates messages.

The "current turn" is everything from the latest human message onwards.
Provider evidence (search results, alternatives) is only trusted within the
turn that produced it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from langchain_core.messages import BaseMessage

from app.shared.text import content_text as message_text


# Payload keys that never support anything a reply or edit needs.
HEAVY_KEYS: Final = frozenset(
    {
        "airline_logo",
        "coordinates",
        "features",
        "geometry",
        "gps_coordinates",
        "image",
        "images",
        "photos",
        "serpapi_link",
        "serpapi_property_details_link",
        "thumbnail",
        "thumbnails",
    }
)
_MAX_DEPTH: Final = 8


def latest_user_index(messages: Sequence[BaseMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].type == "human":
            return index

    return None


def latest_user_request(messages: Sequence[BaseMessage]) -> str:
    """Text of the newest non-empty human message, or ""."""

    for message in reversed(messages):
        if message.type == "human":
            text = message_text(message.content).strip()

            if text:
                return text

    return ""


def current_turn(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """The latest human message and everything after it."""

    index = latest_user_index(messages)
    return messages if index is None else messages[index:]


def current_turn_used_tool(messages: Sequence[BaseMessage], tool_name: str) -> bool:
    return any(
        message.type == "tool" and getattr(message, "name", None) == tool_name
        for message in current_turn(messages)
    )


def current_turn_tool_results(messages: Sequence[BaseMessage]) -> list[tuple[str, Any]]:
    """(tool name, parsed result) for each tool message this turn."""

    return [
        (str(getattr(message, "name", "") or ""), parse_tool_content(message.content))
        for message in current_turn(messages)
        if message.type == "tool"
    ]


def current_turn_tool_exchanges(
    messages: Sequence[BaseMessage],
    tool_name: str,
) -> list[tuple[dict[str, Any], Any]]:
    """(call arguments, parsed result) for one tool this turn.

    A result counts only when it answers a call the assistant made this
    turn (matched by tool-call id), so a stray or replayed tool message can't
    become trusted evidence.
    """

    calls: dict[str, dict[str, Any]] = {}
    exchanges: list[tuple[dict[str, Any], Any]] = []

    for message in current_turn(messages):
        if message.type == "ai":
            for call in getattr(message, "tool_calls", None) or []:
                call_id = call.get("id")
                args = call.get("args")

                if call.get("name") == tool_name and isinstance(call_id, str) and call_id and isinstance(args, dict):
                    calls[call_id] = args

        elif message.type == "tool" and getattr(message, "name", None) == tool_name:
            args = calls.get(getattr(message, "tool_call_id", None))

            if args is not None:
                exchanges.append((args, parse_tool_content(message.content)))

    return exchanges


def latest_tool_batch(messages: Sequence[BaseMessage]) -> list[tuple[str, dict[str, Any], Any]]:
    """(tool name, call arguments, parsed result) for the newest tool batch.

    The batch is the tool messages after the newest assistant message. A
    result counts only when it answers one of that message's calls (matched
    by id and name), so a stray tool message is never applied to state.
    """

    batch: list[tuple[str, dict[str, Any], Any]] = []

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]

        if message.type != "ai":
            continue

        calls = {
            call.get("id"): call
            for call in getattr(message, "tool_calls", None) or []
            if isinstance(call.get("id"), str) and call.get("id")
        }

        for result in messages[index + 1:]:
            call = calls.get(getattr(result, "tool_call_id", None)) if result.type == "tool" else None

            if call is not None and call.get("name") == getattr(result, "name", None):
                args = call.get("args")
                batch.append(
                    (str(call["name"]), dict(args) if isinstance(args, Mapping) else {}, parse_tool_content(result.content))
                )

        break

    return batch


def parse_tool_content(content: Any) -> Any:
    """JSON tool results parsed; anything else kept as text."""

    if isinstance(content, (dict, list)) and not _is_part_list(content):
        return content

    text = message_text(content)

    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def compact_payload(value: Any, *, max_chars: int | None = None) -> Any:
    """Tool output without images and map geometry, optionally size-capped.

    Over ``max_chars`` the JSON text is truncated and returned as a string.
    """

    compact = _strip_heavy(value, 0)

    if max_chars is None:
        return compact

    text = compact if isinstance(compact, str) else json.dumps(compact, ensure_ascii=False, default=str)

    if len(text) <= max_chars:
        return compact

    return text[:max_chars] + " …(truncated)"


def _strip_heavy(value: Any, depth: int) -> Any:
    if depth >= _MAX_DEPTH:
        return "…"

    if isinstance(value, Mapping):
        return {
            key: _strip_heavy(item, depth + 1)
            for key, item in value.items()
            if key not in HEAVY_KEYS
        }

    if isinstance(value, list):
        return [_strip_heavy(item, depth + 1) for item in value]

    return value


def _is_part_list(content: Any) -> bool:
    return (
        isinstance(content, list)
        and bool(content)
        and all(isinstance(part, dict) and "text" in part for part in content)
    )


__all__ = [
    "HEAVY_KEYS",
    "compact_payload",
    "current_turn",
    "current_turn_tool_exchanges",
    "current_turn_tool_results",
    "current_turn_used_tool",
    "latest_user_index",
    "latest_tool_batch",
    "latest_user_request",
    "message_text",
    "parse_tool_content",
]
