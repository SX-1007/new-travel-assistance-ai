"""LangGraph checkpoint persistence.

Read side: chat-history reads over the main sync pool (safe there because
the saver opens its own ``dict_row`` cursors).

Write side: the graph's ``AsyncPostgresSaver`` on its own autocommit pool
(``infrastructure/database/checkpoint_pool.py``). ``setup_checkpointer``
creates and migrates its tables at startup; ``setup()`` runs
``CREATE INDEX CONCURRENTLY``, which PostgreSQL rejects inside a
transaction, so it can't run on the main pool.
"""

from __future__ import annotations

import asyncio
import logging

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
)

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import OperationalError
from psycopg.errors import DuplicateColumn, DuplicateObject, DuplicateTable
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.infrastructure.database.postgres import (
    get_connection,
)
from app.shared.logging import safe_log
from app.shared.observability import log_event


logger = logging.getLogger(__name__)


@contextmanager
def get_checkpointer() -> Generator[
    PostgresSaver,
    None,
    None,
]:
    """Yield a read-only-use PostgresSaver backed by one pooled connection."""
    with get_connection() as conn:
        yield PostgresSaver(
            conn=conn
        )


def _load_chat_history_checkpoint(
    row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Load one persisted checkpoint using its own pooled connection."""
    with get_connection() as conn:
        saver = PostgresSaver(
            conn=conn
        )

        saved = saver.get_tuple(
            {
                "configurable": {
                    "thread_id":
                        row["thread_id"],
                    "checkpoint_ns":
                        row["checkpoint_ns"],
                    "checkpoint_id":
                        row["checkpoint_id"],
                }
            }
        )

    if saved is None:
        return None

    checkpoint = saved.checkpoint

    return {
        "session_id":
            row["thread_id"],
        "updated_at":
            checkpoint.get(
                "ts",
                "",
            ),
        "state":
            checkpoint.get(
                "channel_values",
                {},
            )
            or {},
    }


def fetch_chat_history_checkpoints(
    user_id: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Return the newest persisted LangGraph state for each user session."""
    if not user_id:
        return []

    safe_limit = max(
        1,
        min(
            int(limit),
            20,
        ),
    )

    with get_connection() as conn:
        with conn.cursor(
            row_factory=dict_row
        ) as cur:
            cur.execute(
                """
                SELECT
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id
                FROM (
                    SELECT DISTINCT ON (thread_id)
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id
                    FROM checkpoints
                    WHERE checkpoint_ns = ''
                      AND metadata @> %s
                    ORDER BY
                        thread_id,
                        checkpoint_id DESC
                ) AS latest
                ORDER BY checkpoint_id DESC
                LIMIT %s
                """,
                (
                    Jsonb(
                        {
                            "user_id":
                                user_id,
                        }
                    ),
                    safe_limit,
                ),
            )

            rows = cur.fetchall()

    if not rows:
        return []

    worker_count = min(
        4,
        len(rows),
    )

    try:
        with ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:

            futures = [
                executor.submit(
                    _load_chat_history_checkpoint,
                    row,
                )
                for row in rows
            ]

            return [
                record
                for future in futures
                if (
                    record :=
                    future.result()
                )
                is not None
            ]

    except Exception:
        safe_log(
            logger,
            logging.ERROR,
            (
                "orchestration.checkpoints."
                "history_fetch_failed"
            ),
        )
        raise


# ── Write side: the graph's saver ───────────────────────────────────

# "Already exists" errors mean the schema is ahead of the recorded version.
_MIGRATION_DESYNC_ERRORS = (DuplicateColumn, DuplicateTable, DuplicateObject)


class ResilientAsyncPostgresSaver(AsyncPostgresSaver):
    """``AsyncPostgresSaver`` that retries a dropped connection.

    The Supabase pooler (or a NAT) can drop a socket after the pool's
    checkout check, so a checkpoint write may hit ``OperationalError``.
    Reads and writes are idempotent upserts keyed by checkpoint id, so a
    retry on a fresh connection is safe. ``alist`` is not retried: a partly
    consumed iterator can't be replayed safely.
    """

    retry_attempts: int = 3
    retry_base_delay_seconds: float = 0.25

    async def aget_tuple(self, config: Any) -> Any:
        return await self._retry(super().aget_tuple, config)

    async def aput(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        return await self._retry(super().aput, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config: Any, writes: Any, task_id: str, task_path: str = "") -> None:
        await self._retry(super().aput_writes, config, writes, task_id, task_path)

    async def _retry(self, operation: Any, *args: Any) -> Any:
        delay = self.retry_base_delay_seconds

        for attempt in range(1, self.retry_attempts + 1):
            try:
                return await operation(*args)
            except OperationalError:
                if attempt == self.retry_attempts:
                    raise

                log_event(
                    logger,
                    "checkpoint.retry",
                    stage="storage",
                    outcome="retrying",
                    attempt=attempt,
                    issue_codes=["checkpoint.connection_dropped"],
                )
                await asyncio.sleep(delay)
                delay *= 2

        raise AssertionError("unreachable")


async def setup_checkpointer(saver: AsyncPostgresSaver) -> None:
    """Create/migrate the checkpoint tables, healing a desynced version counter.

    If a run applied a migration's DDL but died before recording its version,
    ``setup()`` replays it and PostgreSQL reports "already exists". That
    error proves the next pending migration is applied (migrations run in
    order from the highest recorded version), so it is recorded and setup is
    retried. Each pass advances the counter by one, so the loop is bounded;
    any other error propagates.
    """

    for _ in range(len(saver.MIGRATIONS) + 1):
        try:
            await saver.setup()
            return
        except _MIGRATION_DESYNC_ERRORS:
            if not await _record_next_migration(saver):
                raise

            log_event(
                logger,
                "checkpoint.migration",
                stage="storage",
                outcome="retrying",
                issue_codes=["checkpoint.migration_desync"],
            )

    raise RuntimeError("Checkpoint setup did not converge after reconciling migration versions.")


async def _record_next_migration(saver: AsyncPostgresSaver) -> bool:
    """Record ``max + 1`` as applied; False when nothing is left to record."""

    async with saver.conn.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT COALESCE(MAX(v), -1) AS v FROM checkpoint_migrations")
            row = await cur.fetchone()
            current = row["v"] if isinstance(row, dict) else row[0]

            if current + 1 >= len(saver.MIGRATIONS):
                return False

            # ON CONFLICT keeps concurrent startups idempotent.
            await cur.execute(
                "INSERT INTO checkpoint_migrations (v) VALUES (%s) ON CONFLICT (v) DO NOTHING",
                (current + 1,),
            )

    return True


__all__ = [
    "ResilientAsyncPostgresSaver",
    "fetch_chat_history_checkpoints",
    "get_checkpointer",
    "setup_checkpointer",
]
