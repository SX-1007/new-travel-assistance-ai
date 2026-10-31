"""Generic logging helpers; no feature-specific event definitions here."""

from __future__ import annotations

import logging
from typing import Any


def bounded_http_status(value: Any) -> int | None:
    """Return a valid HTTP status code or None."""

    if isinstance(value, bool):
        return None

    try:
        status_code = int(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if 100 <= status_code <= 599:
        return status_code

    return None


def safe_log(
    target_logger: logging.Logger,
    level: int,
    event: str,
    *,
    status_code: Any = None,
) -> None:
    """Log a fixed event name (and an HTTP status), never data; a logging
    failure is swallowed so it can't affect the request."""

    try:
        bounded_status = bounded_http_status(status_code)

        if bounded_status is None:
            target_logger.log(level, event)
        else:
            target_logger.log(
                level,
                "%s status_code=%d",
                event,
                bounded_status,
            )

    except Exception:
        return