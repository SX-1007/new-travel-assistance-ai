"""Stopping blocking work that runs in a worker thread.

``asyncio`` can cancel a coroutine, but not the thread it handed work to:
the thread carries on until it returns. Work that spends paid quota in a
thread (trip planning) takes a ``CancellationToken`` and checks it before
each paid call, so cancelling the request stops the spending at the next
call instead of minutes later.

``OperationCancelled`` derives from ``BaseException``, like
``asyncio.CancelledError``: the many ``except Exception`` blocks that turn a
provider failure into "no result" must not swallow it and carry on.
"""

from __future__ import annotations

import threading


class OperationCancelled(BaseException):
    """The work was cancelled; raised at the next check after ``cancel()``."""


class CancellationToken:
    """Thread-safe, one-way: once cancelled it stays cancelled."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise OperationCancelled


__all__ = ["CancellationToken", "OperationCancelled"]
