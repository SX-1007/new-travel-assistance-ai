"""Durable PostgreSQL persistence for trip-session ownership.

This module contains database concerns only. It does not know about LangGraph,
FastAPI, request objects, or orchestration policy.
"""

from __future__ import annotations

from threading import Lock
from typing import Final

from app.infrastructure.database.postgres import get_connection


_RESERVATION_TABLE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS trip_session_owners (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('initializing', 'ready')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_RESERVATION_SCHEMA_LOCK_SQL: Final[str] = (
    "SELECT pg_advisory_xact_lock(794523017)"
)


class SessionOwnershipRepository:
    """PostgreSQL repository for durable trip-session ownership records."""

    _schema_ready = False
    _schema_guard = Lock()

    @classmethod
    def _ensure_schema(cls) -> None:
        """Create the ownership table once per process, safely across processes."""
        if cls._schema_ready:
            return

        with cls._schema_guard:
            if cls._schema_ready:
                return

            with get_connection() as connection:
                with connection.transaction():
                    with connection.cursor() as cursor:
                        cursor.execute(_RESERVATION_SCHEMA_LOCK_SQL)
                        cursor.execute(_RESERVATION_TABLE_SQL)

            cls._schema_ready = True

    def reserve(self, session_id: str, user_id: str) -> bool:
        """Atomically reserve a new session ID for one authenticated user."""
        self._ensure_schema()

        with get_connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO trip_session_owners (session_id, user_id, state)
                        VALUES (%s, %s, 'initializing')
                        ON CONFLICT (session_id) DO NOTHING
                        RETURNING session_id
                        """,
                        (session_id, user_id),
                    )
                    return cursor.fetchone() is not None

    def complete(self, session_id: str, user_id: str) -> bool:
        """Mark an initializing reservation ready when ownership still matches."""
        self._ensure_schema()

        with get_connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE trip_session_owners
                        SET state = 'ready'
                        WHERE session_id = %s
                          AND user_id = %s
                          AND state = 'initializing'
                        RETURNING session_id
                        """,
                        (session_id, user_id),
                    )
                    return cursor.fetchone() is not None

    def reclaim_expired(
        self,
        session_id: str,
        user_id: str,
        lease_seconds: int,
    ) -> bool:
        """Refresh an expired initializing lease belonging to the same user."""
        self._ensure_schema()

        with get_connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE trip_session_owners
                        SET created_at = NOW()
                        WHERE session_id = %s
                          AND user_id = %s
                          AND state = 'initializing'
                          AND created_at <= NOW() - (%s * INTERVAL '1 second')
                        RETURNING session_id
                        """,
                        (session_id, user_id, lease_seconds),
                    )
                    return cursor.fetchone() is not None

    def finalize_expired(
        self,
        session_id: str,
        user_id: str,
        lease_seconds: int,
    ) -> bool:
        """Mark an expired initializing lease ready for the same owner."""
        self._ensure_schema()

        with get_connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE trip_session_owners
                        SET state = 'ready'
                        WHERE session_id = %s
                          AND user_id = %s
                          AND state = 'initializing'
                          AND created_at <= NOW() - (%s * INTERVAL '1 second')
                        RETURNING session_id
                        """,
                        (session_id, user_id, lease_seconds),
                    )
                    return cursor.fetchone() is not None

    def release(self, session_id: str, user_id: str) -> None:
        """Delete only an initializing reservation owned by the same user."""
        self._ensure_schema()

        with get_connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        DELETE FROM trip_session_owners
                        WHERE session_id = %s
                          AND user_id = %s
                          AND state = 'initializing'
                        """,
                        (session_id, user_id),
                    )

    def backfill(self, session_id: str, user_id: str) -> bool:
        """Persist ownership for a legacy checkpoint already verified elsewhere."""
        self._ensure_schema()

        with get_connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO trip_session_owners (session_id, user_id, state)
                        VALUES (%s, %s, 'ready')
                        ON CONFLICT (session_id) DO UPDATE
                        SET state = 'ready'
                        WHERE trip_session_owners.user_id = EXCLUDED.user_id
                        RETURNING session_id
                        """,
                        (session_id, user_id),
                    )
                    return cursor.fetchone() is not None

    def verify(self, session_id: str, user_id: str) -> bool:
        """Return whether a ready session belongs to the supplied user."""
        self._ensure_schema()

        with get_connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT session_id
                        FROM trip_session_owners
                        WHERE session_id = %s
                          AND user_id = %s
                          AND state = 'ready'
                        """,
                        (session_id, user_id),
                    )
                    return cursor.fetchone() is not None