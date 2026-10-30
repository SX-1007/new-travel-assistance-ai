"""PostgreSQL persistence for the relational user profile.

One row per user in ``user_profiles``: the onboarding fields plus the traits
memory extraction learns from chat. Profile and memory rules live in
``app.modules.profile`` and ``app.memory``; this module only stores rows.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Any, Dict, Final

from psycopg import Connection, sql
from psycopg.rows import dict_row

from app.infrastructure.database.postgres import get_connection
from app.shared.logging import safe_log

logger = logging.getLogger(__name__)

ConnectionFactory = Callable[[], AbstractContextManager[Connection[Any]]]

# Every writable column. Anything else is a caller bug and is refused, so a
# column name is never built from unchecked input.
PROFILE_COLUMNS: Final[tuple[str, ...]] = (
    "name",
    "home_country",
    "home_state",
    "travel_pacing",
    "dietary_restrictions",
    "interests",
    "accommodation_preferences",
)

_TABLE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS user_profiles (
    id TEXT PRIMARY KEY,
    name TEXT,
    home_country TEXT,
    home_state TEXT,
    travel_pacing TEXT,
    dietary_restrictions TEXT[],
    interests TEXT[],
    accommodation_preferences TEXT[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_SCHEMA_LOCK_SQL: Final[str] = "SELECT pg_advisory_xact_lock(794523018)"

# Timestamps stay in the table; callers get only the profile fields, as
# plain JSON-friendly values.
_SELECT_SQL: Final = sql.SQL("SELECT {columns} FROM user_profiles WHERE id = %s").format(
    columns=sql.SQL(", ").join(sql.Identifier(c) for c in ("id", *PROFILE_COLUMNS))
)


class PostgresUserProfileStore:
    """``user_profiles`` table over the shared connection pool."""

    def __init__(self, *, connect: ConnectionFactory = get_connection) -> None:
        self._connect = connect

    def ensure_schema(self) -> None:
        """Create the table; safe to run from several processes at once."""
        with self._connect() as conn:
            with conn.transaction():
                conn.execute(_SCHEMA_LOCK_SQL)
                conn.execute(_TABLE_SQL)

    def fetch_profile(
        self,
        user_id: str,
        *,
        raise_on_error: bool = True,
    ) -> Dict[str, Any]:
        """The user's profile row, or ``{}`` when there is none.

        Raises by default so memory updates fail closed; pass
        ``raise_on_error=False`` where an empty profile is acceptable.
        """
        try:
            with self._connect() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(_SELECT_SQL, (user_id,))
                    row = cur.fetchone()
            return dict(row) if row else {}
        except Exception:
            safe_log(logger, logging.ERROR, "storage.postgres.profile_fetch_failed")
            if raise_on_error:
                raise
            return {}

    def create_profile(
        self,
        user_id: str,
        profile: Mapping[str, Any],
    ) -> bool:
        """Insert the row, or update only the given columns of an existing one."""
        columns = _checked_columns(profile)
        if not user_id or not columns:
            return False

        statement = sql.SQL(
            "INSERT INTO user_profiles (id, {columns}) VALUES (%s, {values}) "
            "ON CONFLICT (id) DO UPDATE SET {updates}, updated_at = NOW()"
        ).format(
            columns=sql.SQL(", ").join(map(sql.Identifier, columns)),
            values=sql.SQL(", ").join([sql.Placeholder()] * len(columns)),
            updates=sql.SQL(", ").join(
                sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(c)) for c in columns
            ),
        )
        try:
            with self._connect() as conn:
                with conn.transaction():
                    conn.execute(statement, (user_id, *(profile[c] for c in columns)))
            safe_log(logger, logging.INFO, "storage.postgres.profile_created")
            return True
        except Exception:
            safe_log(logger, logging.ERROR, "storage.postgres.profile_create_failed")
            return False

    def update_profile(
        self,
        user_id: str,
        updates: Mapping[str, Any],
    ) -> bool:
        """Persist already-prepared fields of an existing row.

        False when the user has no row. Memory merge rules belong to
        ``app.memory.service``, not here.
        """
        columns = _checked_columns(updates)
        if not user_id or not columns:
            return False

        statement = sql.SQL(
            "UPDATE user_profiles SET {updates}, updated_at = NOW() WHERE id = %s"
        ).format(
            updates=sql.SQL(", ").join(
                sql.SQL("{} = %s").format(sql.Identifier(c)) for c in columns
            ),
        )
        try:
            with self._connect() as conn:
                with conn.transaction():
                    updated = conn.execute(statement, (*(updates[c] for c in columns), user_id)).rowcount
        except Exception:
            safe_log(logger, logging.ERROR, "storage.postgres.profile_update_failed")
            return False

        if updated:
            safe_log(logger, logging.INFO, "storage.postgres.profile_updated")
            return True

        safe_log(logger, logging.WARNING, "storage.postgres.profile_update_empty")
        return False


def _checked_columns(fields: Mapping[str, Any]) -> list[str]:
    """The field names in table order; empty when any name is not a column."""
    unknown = set(fields) - set(PROFILE_COLUMNS)
    if unknown:
        safe_log(logger, logging.ERROR, "storage.postgres.profile_unknown_column")
        return []
    return [c for c in PROFILE_COLUMNS if c in fields]


__all__ = ["PROFILE_COLUMNS", "PostgresUserProfileStore"]
