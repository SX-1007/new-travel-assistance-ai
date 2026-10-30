"""PostgreSQL/pgvector store for long-term semantic user preferences.

Implements ``app.memory.ports.SemanticPreferenceStorePort`` structurally; the
memory package depends only on that contract, never on this module.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from typing import Dict, List, Protocol

from psycopg.rows import dict_row

from app.infrastructure.database.postgres import (
    ensure_vector_extension,
    get_connection,
    prepare_vector_parameter,
    vector_cast_suffix,
)
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

_FETCH_LIMIT = 200


class _Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def _log_info(event: str) -> None:
    safe_log(logger, logging.INFO, event)


def _log_error(event: str) -> None:
    safe_log(logger, logging.ERROR, event)


class PgVectorSemanticPreferenceStore:
    """
    PostgreSQL/pgvector implementation of the
    semantic-memory persistence contract.
    """

    def __init__(self, *, embedder: _Embedder) -> None:
        self._embedder = embedder
        self._dimension = int(embedder.dimension)

    def ensure_schema(self) -> None:
        """RUN ONCE AT STARTUP: create the ``mem0_vector`` table and index."""
        ensure_vector_extension()

        with get_connection() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS mem0_vector (
                    id              UUID PRIMARY KEY,
                    user_id         TEXT NOT NULL,
                    category        TEXT NOT NULL,
                    preference_text TEXT NOT NULL,
                    embedding       vector({self._dimension}) NOT NULL,
                    created_at      TIMESTAMP WITH TIME ZONE
                                    NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE (
                        user_id,
                        category,
                        preference_text
                    )
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mem0_vector_user_category
                ON mem0_vector (
                    user_id,
                    category,
                    created_at DESC
                );
            """)

        _log_info("vector.semantic_preferences.ready")

    def fetch_preferences(
        self,
        user_id: str,
        *,
        raise_on_error: bool = False,
    ) -> Dict[str, List[str]]:
        """Fetch all persisted semantic preferences for one user."""
        if not user_id:
            return {}

        try:
            with get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT category, preference_text, created_at
                        FROM mem0_vector
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s;
                        """,
                        (user_id, _FETCH_LIMIT),
                    )
                    rows = cur.fetchall()

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

            _log_info("vector.semantic_preferences.fetch_succeeded")
            return grouped

        except Exception:
            _log_error("vector.semantic_preferences.fetch_failed")
            if raise_on_error:
                raise
            return {}

    def insert_preferences(
        self,
        user_id: str,
        categorized_preferences: Mapping[str, List[str]],
    ) -> bool:
        """Embed and persist long-term semantic preferences in PostgreSQL."""
        if not user_id or not categorized_preferences:
            return False

        try:
            all_texts: List[str] = []
            text_categories: List[str] = []

            for category, preferences in categorized_preferences.items():
                for preference_text in preferences:
                    text = str(preference_text).strip()
                    if not text:
                        continue
                    all_texts.append(text)
                    text_categories.append(category)

            if not all_texts:
                return False

            vectors = self._embedder.embed_documents(all_texts)
            if len(vectors) != len(all_texts):
                raise RuntimeError("Embedding count mismatch.")

            cast = vector_cast_suffix()
            records: List[tuple] = [
                (
                    str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{user_id}|{category}|{text.casefold()}",
                        )
                    ),
                    user_id,
                    category,
                    text,
                    prepare_vector_parameter(vector),
                )
                for text, category, vector in zip(all_texts, text_categories, vectors)
            ]

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        f"""
                        INSERT INTO mem0_vector
                            (id, user_id, category, preference_text, embedding)
                        VALUES (%s, %s, %s, %s, %s{cast})
                        ON CONFLICT (id) DO UPDATE SET
                            category = EXCLUDED.category,
                            preference_text = EXCLUDED.preference_text,
                            embedding = EXCLUDED.embedding;
                        """,
                        records,
                    )

            _log_info("vector.semantic_preferences.insert_succeeded")
            return True

        except Exception:
            _log_error("vector.semantic_preferences.insert_failed")
            return False


__all__ = [
    "PgVectorSemanticPreferenceStore",
]
