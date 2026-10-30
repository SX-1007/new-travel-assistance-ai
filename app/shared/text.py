"""Plain-text extraction from chat-model message content."""

from __future__ import annotations

from typing import Any


def content_text(content: Any) -> str:
    """Text of message content: a string, or a list of string/text parts."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))

        return " ".join(parts)

    return "" if content is None else str(content)


__all__ = [
    "content_text",
]
