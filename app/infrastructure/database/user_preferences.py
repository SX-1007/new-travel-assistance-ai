"""PostgreSQL store for each user's learned preference sentences.

Implements ``app.memory.ports.SemanticPreferenceStorePort`` structurally; the
memory package depends only on that contract, never on this module.

The table keeps its old name, ``mem0_vector``, so existing rows stay where
they are. It used to hold a 3,072-dimension embedding per sentence, paid for
with one embedding call per save, but nothing ever searched by it: the
preferences are read back as plain text, newest first. New rows no longer
get one. An existing ``embedding`` column is made optional (its old values
are kept, never dropped) so new rows can be written without it.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Any, Dict, Final, List

from psycopg import Connection
from psycopg.rows import dict_row

from app.infrastructure.database.postgres import get_connection
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

ConnectionFactory = Callable[[], AbstractContextManager[Connection[Any]]]

_FETCH_LIMIT: Final = 200

_TABLE_SQL: Final = """
CREATE TABLE IF NOT EXISTS mem0_vector (
    id              UUID PRIMARY KEY,
    user_id         TEXT NOT NULL,
    category        TEXT NOT NULL,
    preference_text TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, category, preference_text)
)
"""

_INDEX_SQL: Final = """
CREATE INDEX IF NOT EXISTS idx_mem0_vector_user_category
ON mem0_vector (user_id, category, created_at DESC)
"""

# Tables created before this version have a required embedding column.
_OPTIONAL_EMBEDDING_SQL: Final = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'mem0_vector'
          AND column_name = 'embedding'
          AND is_nullable = 'NO'
    ) THEN
        ALTER TABLE mem0_vector ALTER COLUMN embedding DROP NOT NULL;
    END IF;
END $$
"""

# Serialises schema changes across processes (each table has its own lock id).
_SCHEMA_LOCK_SQL: Final = "SELECT pg_advisory_xact_lock(794523021)"

_INSERT_SQL: Final = """
INSERT INTO mem0_vector (id, user_id, category, preference_text)
VALUES (%s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    category = EXCLUDED.category,
    preference_text = EXCLUDED.preference_text
"""

_SELECT_SQL: Final = """
SELECT category, preference_text
FROM mem0_vector
WHERE user_id = %s
ORDER BY created_at DESC
LIMIT %s
"""


class PostgresPreferenceStore:
    """``mem0_vector``: each user's learned preference sentences."""

    def __init__(self, *, connect: ConnectionFactory = get_connection) -> None:
        self._connect = connect

    def ensure_schema(self) -> None:
        """Create or update the table; safe to run from several processes."""

        with self._connect() as conn:
            with conn.transaction():
                conn.execute(_SCHEMA_LOCK_SQL)
                conn.execute(_TABLE_SQL)
                conn.execute(_OPTIONAL_EMBEDDING_SQL)
                conn.execute(_INDEX_SQL)

        safe_log(logger, logging.INFO, "storage.preferences.ready")

    def fetch_preferences(
        self,
        user_id: str,
        *,
        raise_on_error: bool = False,
    ) -> Dict[str, List[str]]:
        """A user's preferences by category, newest first, without duplicates."""

        if not user_id:
            return {}

        try:
            with self._connect() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(_SELECT_SQL, (user_id, _FETCH_LIMIT))
                    rows = cur.fetchall()

        except Exception:
            safe_log(logger, logging.ERROR, "storage.preferences.fetch_failed")

            if raise_on_error:
                raise

            return {}

        grouped: Dict[str, List[str]] = {}
        seen: Dict[str, set[str]] = {}

        for row in rows:
            category = str(row.get("category") or "").strip()
            text = str(row.get("preference_text") or "").strip()

            if not category or not text:
                continue

            key = text.casefold()
            category_seen = seen.setdefault(category, set())

            if key in category_seen:
                continue

            category_seen.add(key)
            grouped.setdefault(category, []).append(text)

        return grouped

    def insert_preferences(
        self,
        user_id: str,
        categorized_preferences: Mapping[str, List[str]],
    ) -> bool:
        """Save preference sentences; True when at least one was written.

        The id is derived from (user, category, sentence), so saving the same
        sentence again updates its row instead of adding a duplicate.
        """

        if not user_id or not categorized_preferences:
            return False

        records = [
            (_preference_id(user_id, category, text), user_id, category, text)
            for category, preferences in categorized_preferences.items()
            for text in (str(item).strip() for item in preferences or ())
            if text
        ]

        if not records:
            return False

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.executemany(_INSERT_SQL, records)

        except Exception:
            safe_log(logger, logging.ERROR, "storage.preferences.insert_failed")
            return False

        return True


def _preference_id(user_id: str, category: str, text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}|{category}|{text.casefold()}"))


__all__ = ["PostgresPreferenceStore"]
