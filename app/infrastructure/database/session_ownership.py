"""Durable PostgreSQL persistence for trip-session ownership.

A row ties a session id to its user. A new trip first reserves the id
(``initializing``); once its first checkpoint is saved the row becomes
``ready``, and only ready rows grant access. An ``initializing`` row whose
lease expired (its request died) can be reclaimed by the same user.

``updated_at`` is when the session last changed (set on reserve, complete
and after every chat turn), with an index on ``(user_id, updated_at)``: chat
history lists a user's newest sessions from this table alone, instead of
searching every user's checkpoints by a JSON field.

Sessions saved before ownership rows existed get their row once, from
their checkpoints' metadata (``backfill_history``, run at start-up and
recorded in ``app_migrations`` so it never runs twice).

Database concerns only: no LangGraph, FastAPI or access policy here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Final

from psycopg import Connection

from app.infrastructure.database.postgres import get_connection


logger = logging.getLogger(__name__)

ConnectionFactory = Callable[[], AbstractContextManager[Connection[Any]]]

_SCHEMA_SQL: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS trip_session_owners (
        session_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('initializing', 'ready')),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # Nullable: rows from before this column get their time from the
    # backfill below; every write from now on sets it.
    "ALTER TABLE trip_session_owners ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ",
    """
    CREATE INDEX IF NOT EXISTS trip_session_owners_user_recent_idx
    ON trip_session_owners (user_id, updated_at DESC NULLS LAST)
    """,
    """
    CREATE TABLE IF NOT EXISTS app_migrations (
        name TEXT PRIMARY KEY,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
)

# Serialises schema changes and the backfill across processes.
_SCHEMA_LOCK_SQL: Final = "SELECT pg_advisory_xact_lock(794523017)"

HISTORY_BACKFILL: Final = "trip_session_owners.history_v1"

# The newest checkpoint of every session, with its owner and time, for the
# one-time backfill. Each session's checkpoints sit together in the primary
# key (thread_id, checkpoint_ns, checkpoint_id).
_LEGACY_SESSIONS_SQL: Final = """
SELECT DISTINCT ON (thread_id)
    thread_id,
    metadata ->> 'user_id' AS user_id,
    checkpoint ->> 'ts' AS ts
FROM checkpoints
WHERE checkpoint_ns = ''
ORDER BY thread_id, checkpoint_id DESC
"""


class SessionOwnershipRepository:
    """PostgreSQL repository for durable trip-session ownership records."""

    def __init__(self, *, connect: ConnectionFactory = get_connection) -> None:
        self._connect = connect
        self._schema_ready = False
        self._schema_guard = Lock()

    def ensure_schema(self) -> None:
        """Create or update the table once per instance, safely across processes."""

        if self._schema_ready:
            return

        with self._schema_guard:
            if self._schema_ready:
                return

            with self._transaction() as cursor:
                cursor.execute(_SCHEMA_LOCK_SQL)

                for statement in _SCHEMA_SQL:
                    cursor.execute(statement)

            self._schema_ready = True

    def backfill_history(self) -> int:
        """Once ever: give every session saved without an ownership row one
        (owner from its checkpoints' server-written metadata), and give
        every row without ``updated_at`` its newest checkpoint's time.
        Returns the number of rows added; 0 when already done."""

        self.ensure_schema()

        with self._transaction() as cursor:
            cursor.execute(_SCHEMA_LOCK_SQL)
            cursor.execute("SELECT 1 FROM app_migrations WHERE name = %s", (HISTORY_BACKFILL,))

            if cursor.fetchone() is not None:
                return 0

            cursor.execute("SELECT to_regclass('checkpoints') IS NOT NULL")
            row = cursor.fetchone()
            sessions = []

            if row is not None and row[0]:
                cursor.execute(_LEGACY_SESSIONS_SQL)
                sessions = [
                    (thread_id, user_id, _timestamp(ts))
                    for thread_id, user_id, ts in cursor.fetchall()
                ]

            added = 0

            for thread_id, user_id, updated_at in sessions:
                if not isinstance(user_id, str) or not user_id:
                    continue

                cursor.execute(
                    """
                    INSERT INTO trip_session_owners (session_id, user_id, state, updated_at)
                    VALUES (%s, %s, 'ready', %s)
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    (thread_id, user_id, updated_at),
                )
                added += cursor.rowcount

                # A row that existed before ``updated_at`` gets its time.
                cursor.execute(
                    """
                    UPDATE trip_session_owners
                    SET updated_at = %s
                    WHERE session_id = %s AND updated_at IS NULL
                    """,
                    (updated_at, thread_id),
                )

            cursor.execute("UPDATE trip_session_owners SET updated_at = created_at WHERE updated_at IS NULL")
            cursor.execute("INSERT INTO app_migrations (name) VALUES (%s)", (HISTORY_BACKFILL,))

        logger.info("session_ownership.history_backfill added=%d", added)
        return added

    def reserve(self, session_id: str, user_id: str) -> bool:
        """Atomically reserve a new session ID for one authenticated user."""

        return self._returns_row(
            """
            INSERT INTO trip_session_owners (session_id, user_id, state, updated_at)
            VALUES (%s, %s, 'initializing', NOW())
            ON CONFLICT (session_id) DO NOTHING
            RETURNING session_id
            """,
            (session_id, user_id),
        )

    def complete(self, session_id: str, user_id: str) -> bool:
        """Mark an initializing reservation ready when ownership still matches."""

        return self._returns_row(
            """
            UPDATE trip_session_owners
            SET state = 'ready', updated_at = NOW()
            WHERE session_id = %s
              AND user_id = %s
              AND state = 'initializing'
            RETURNING session_id
            """,
            (session_id, user_id),
        )

    def reclaim_expired(self, session_id: str, user_id: str, lease_seconds: int) -> bool:
        """Refresh an expired initializing lease belonging to the same user."""

        return self._returns_row(
            """
            UPDATE trip_session_owners
            SET created_at = NOW(), updated_at = NOW()
            WHERE session_id = %s
              AND user_id = %s
              AND state = 'initializing'
              AND created_at <= NOW() - (%s * INTERVAL '1 second')
            RETURNING session_id
            """,
            (session_id, user_id, lease_seconds),
        )

    def finalize_expired(self, session_id: str, user_id: str, lease_seconds: int) -> bool:
        """Mark an expired initializing lease ready for the same owner."""

        return self._returns_row(
            """
            UPDATE trip_session_owners
            SET state = 'ready', updated_at = NOW()
            WHERE session_id = %s
              AND user_id = %s
              AND state = 'initializing'
              AND created_at <= NOW() - (%s * INTERVAL '1 second')
            RETURNING session_id
            """,
            (session_id, user_id, lease_seconds),
        )

    def release(self, session_id: str, user_id: str) -> None:
        """Delete only an initializing reservation owned by the same user."""

        self._execute(
            """
            DELETE FROM trip_session_owners
            WHERE session_id = %s
              AND user_id = %s
              AND state = 'initializing'
            """,
            (session_id, user_id),
        )

    def backfill(self, session_id: str, user_id: str) -> bool:
        """Record ownership for a session that has checkpoints but no row (saved
        before ownership rows existed), after the caller verified the owner."""

        return self._returns_row(
            """
            INSERT INTO trip_session_owners (session_id, user_id, state, updated_at)
            VALUES (%s, %s, 'ready', NOW())
            ON CONFLICT (session_id) DO UPDATE
            SET state = 'ready',
                updated_at = COALESCE(trip_session_owners.updated_at, NOW())
            WHERE trip_session_owners.user_id = EXCLUDED.user_id
            RETURNING session_id
            """,
            (session_id, user_id),
        )

    def delete(self, session_id: str, user_id: str) -> bool:
        """Delete a session's ownership row when it belongs to the same user."""

        return self._returns_row(
            """
            DELETE FROM trip_session_owners
            WHERE session_id = %s
              AND user_id = %s
            RETURNING session_id
            """,
            (session_id, user_id),
        )

    def verify(self, session_id: str, user_id: str) -> bool:
        """Return whether a ready session belongs to the supplied user."""

        return self._returns_row(
            """
            SELECT session_id
            FROM trip_session_owners
            WHERE session_id = %s
              AND user_id = %s
              AND state = 'ready'
            """,
            (session_id, user_id),
        )

    def touch(self, session_id: str, user_id: str) -> bool:
        """Record that the user's session just changed (a chat turn)."""

        return self._returns_row(
            """
            UPDATE trip_session_owners
            SET updated_at = NOW()
            WHERE session_id = %s
              AND user_id = %s
            RETURNING session_id
            """,
            (session_id, user_id),
        )

    def recent_sessions(self, user_id: str, limit: int) -> list[str]:
        """The user's session ids, newest change first (an index scan)."""

        self.ensure_schema()

        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT session_id
                FROM trip_session_owners
                WHERE user_id = %s
                ORDER BY updated_at DESC NULLS LAST, session_id
                LIMIT %s
                """,
                (user_id, limit),
            )
            return [row[0] for row in cursor.fetchall()]

    # ── Internals ───────────────────────────────────────────────────

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        """A cursor inside one transaction."""

        with self._connect() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    yield cursor

    def _returns_row(self, sql: str, params: tuple[Any, ...]) -> bool:
        self.ensure_schema()

        with self._transaction() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchone() is not None

    def _execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self.ensure_schema()

        with self._transaction() as cursor:
            cursor.execute(sql, params)


def _timestamp(value: Any) -> datetime | None:
    """A checkpoint's ``ts`` as a time; None (then ``created_at``) when unreadable."""

    if not isinstance(value, str):
        return None

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None

    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


__all__ = [
    "HISTORY_BACKFILL",
    "SessionOwnershipRepository",
]
