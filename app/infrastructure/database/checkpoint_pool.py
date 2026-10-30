"""Async PostgreSQL pool for the LangGraph checkpointer.

Separate from the main sync pool (``postgres.py``) on purpose: the saver
needs autocommit (its ``CREATE INDEX CONCURRENTLY`` migrations can't run in a
transaction) and ``dict_row`` rows, and checkpoint writes must not compete
with request queries for connections. Created at startup from the
composition root; nothing here reads settings.
"""

from __future__ import annotations

from typing import Any, Final

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


CHECKPOINT_POOL_MAX_SIZE: Final = 4
# Retire idle connections before the Supabase pooler or a NAT drops them.
_MAX_IDLE_SECONDS: Final = 240.0
_CHECKOUT_TIMEOUT_SECONDS: Final = 30.0


async def open_checkpoint_pool(
    conninfo: str,
    *,
    max_size: int = CHECKPOINT_POOL_MAX_SIZE,
) -> AsyncConnectionPool[Any]:
    """Open and return the pool; the caller closes it at shutdown."""

    if not conninfo or not conninfo.strip():
        raise ValueError("conninfo must not be empty")

    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
        conninfo,
        min_size=1,
        max_size=max_size,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            # No server-side prepared statements, so PgBouncer ports work.
            "prepare_threshold": None,
            # The graph can idle on a connection-free LLM call for minutes;
            # keepalives surface dead peers instead of "SSL error: bad length".
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 3,
        },
        check=AsyncConnectionPool.check_connection,
        max_idle=_MAX_IDLE_SECONDS,
        timeout=_CHECKOUT_TIMEOUT_SECONDS,
        open=False,
    )
    await pool.open(wait=True)
    return pool


__all__ = [
    "CHECKPOINT_POOL_MAX_SIZE",
    "open_checkpoint_pool",
]
