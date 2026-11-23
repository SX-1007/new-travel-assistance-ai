"""The shared PostgreSQL connection pool.

The pool is created by ``init_connection_pool`` at application startup, not at
import time, and receives its connection string from the composition root.
Importing this module therefore never needs configuration or a network.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any

from psycopg import Connection
from psycopg_pool import ConnectionPool

from app.shared.logging import safe_log

logger = logging.getLogger(__name__)


# ── Pool configuration ──────────────────────────────────────────────
STATEMENT_TIMEOUT_MS = 15_000
POOL_MIN_SIZE = 2
POOL_MAX_SIZE = 20
POOL_MAX_IDLE = 300
POOL_TIMEOUT = 30.0


def _configure_conn(conn: Connection[Any]) -> None:
    """Configure each newly-created PostgreSQL connection."""
    conn.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")

    # psycopg_pool expects configure hooks to return an idle connection.
    conn.commit()


# Keep the concrete pool private. Other modules use get_connection().
_connection_pool: ConnectionPool | None = None
_pool_lock = Lock()


def init_connection_pool(conninfo: str) -> None:
    """Create and open the PostgreSQL pool during application startup."""
    global _connection_pool

    if not conninfo or not conninfo.strip():
        raise ValueError("conninfo must not be empty")

    with _pool_lock:
        if _connection_pool is not None:
            return

        pool = ConnectionPool(
            conninfo=conninfo,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            max_idle=POOL_MAX_IDLE,
            timeout=POOL_TIMEOUT,
            configure=_configure_conn,
            kwargs={
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 3,
            },
            check=ConnectionPool.check_connection,
            open=False,
        )
        pool.open(wait=True)
        _connection_pool = pool

    safe_log(logger, logging.INFO, "storage.postgres.pool_opened")


def close_connection_pool() -> None:
    """Close the PostgreSQL pool during application shutdown."""
    global _connection_pool

    with _pool_lock:
        pool, _connection_pool = _connection_pool, None

    if pool is None:
        return

    try:
        pool.close()
        safe_log(logger, logging.INFO, "storage.postgres.pool_closed")
    except Exception:
        safe_log(logger, logging.ERROR, "storage.postgres.pool_close_failed")


def verify_connection_pool() -> bool:
    """Return True when PostgreSQL can execute a trivial query."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                return cur.fetchone() is not None
    except Exception:
        safe_log(logger, logging.ERROR, "storage.postgres.pool_health_failed")
        return False


@contextmanager
def get_connection() -> Iterator[Connection[Any]]:
    """A pooled connection: the only way other modules reach the database, so
    the pool can change without touching them.
    """
    pool = _connection_pool

    if pool is None:
        raise RuntimeError(
            "PostgreSQL pool is not initialised; call init_connection_pool() first"
        )

    with pool.connection() as conn:
        yield conn
