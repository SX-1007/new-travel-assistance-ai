"""Bounded, structured request log events.

Used by planning, output review, factual Q&A and, later, the graph and the
form/chat routers. Every field is sanitised to a short safe token, so a log
line can never carry user text, provider payloads or unbounded data, and a
broken handler can never affect the request.

The fields are written into the message itself (``event key=value ...``).
The app's log format prints only ``%(message)s``; fields passed only through
``extra=`` would be invisible. They are also attached as
``record.event_fields`` for structured handlers.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Final


_SAFE_TOKEN: Final = re.compile(r"[^A-Za-z0-9._:-]+")
_COUNTRY_CODE: Final = re.compile(r"[A-Z]{2}")

_MAX_IDENTIFIER_LENGTH: Final = 64
_MAX_ISSUE_CODES: Final = 8
_MAX_ELAPSED_MS: Final = 86_400_000

# Outcomes that mean something went wrong are logged as warnings.
_WARNING_OUTCOMES: Final = frozenset(
    {"failed", "rejected", "unavailable", "interrupted", "error"}
)


@dataclass(frozen=True, slots=True)
class LogContext:
    """Correlation fields shared by every event of one request."""

    request_id: str | None = None
    session_id: str | None = None
    destination_country_code: str | None = None

    def with_country(self, code: str | None) -> LogContext:
        return replace(self, destination_country_code=code)


def log_event(
    target_logger: logging.Logger,
    event: str,
    *,
    context: LogContext | None = None,
    stage: str = "unknown",
    outcome: str = "unknown",
    attempt: int = 0,
    issue_codes: Sequence[str] = (),
    elapsed_ms: float | None = None,
    reason: str | None = None,
    level: int | None = None,
) -> None:
    """Emit one event; logging must never alter the request."""

    try:
        context = context or LogContext()
        record = {
            "request_id": _token(context.request_id, limit=_MAX_IDENTIFIER_LENGTH),
            "session_id": _token(context.session_id, limit=_MAX_IDENTIFIER_LENGTH),
            "country": _country(context.destination_country_code),
            "attempt": _bounded_int(attempt, maximum=99),
            "stage": _token(stage, limit=48),
            "outcome": _token(outcome, limit=32),
        }

        if elapsed_ms is not None:
            record["elapsed_ms"] = _bounded_int(elapsed_ms, maximum=_MAX_ELAPSED_MS)

        codes = _issue_codes(issue_codes)

        if codes:
            record["issues"] = ",".join(codes)

        if reason is not None:
            record["reason"] = _token(reason, limit=48)

        if level is None:
            level = (
                logging.WARNING
                if record["outcome"] in _WARNING_OUTCOMES
                else logging.INFO
            )

        target_logger.log(
            level,
            "%s %s",
            _token(event, limit=64, fallback="event.unknown"),
            " ".join(f"{key}={value}" for key, value in record.items()),
            extra={"event_fields": record},
        )

    except Exception:
        return


def _token(value: Any, *, limit: int, fallback: str = "unknown") -> str:
    if not isinstance(value, str):
        return fallback

    return _SAFE_TOKEN.sub("_", value.strip()).strip("_")[:limit] or fallback


def _country(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"

    code = value.strip().upper()
    return code if _COUNTRY_CODE.fullmatch(code) else "unknown"


def _bounded_int(value: Any, *, maximum: int) -> int:
    if isinstance(value, bool):
        return 0

    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0

    return max(0, min(number, maximum))


def _issue_codes(values: Sequence[Any]) -> tuple[str, ...]:
    """Distinct codes in first-seen order, at most ``_MAX_ISSUE_CODES``."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        return ()

    codes: list[str] = []

    for value in values:
        code = _token(value, limit=_MAX_IDENTIFIER_LENGTH)

        if code not in codes:
            codes.append(code)

            if len(codes) == _MAX_ISSUE_CODES:
                break

    return tuple(codes)


__all__ = [
    "LogContext",
    "log_event",
]
