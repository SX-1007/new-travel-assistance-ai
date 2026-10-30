"""Session-scoped semantic search over provider travel options (pgvector).

Hotels, flights and activities found for one planning session are embedded and
stored with a TTL so later turns can rank them against user preferences. This
is retrieval infrastructure, not long-term user memory.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Protocol

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.infrastructure.database.postgres import (
    ensure_vector_extension,
    get_connection,
    prepare_vector_parameter,
    vector_cast_suffix,
)
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 200
DEFAULT_TTL_HOURS = 1
HNSW_EF_SEARCH = 40


class _Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _log_info(event: str) -> None:
    safe_log(logger, logging.INFO, event)


def _log_warning(event: str) -> None:
    safe_log(logger, logging.WARNING, event)


def _log_error(event: str) -> None:
    safe_log(logger, logging.ERROR, event)


# ══════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════
def _deterministic_id(session_id: str, category: str, item: Dict[str, Any]) -> str:
    """Deterministic UUID via uuid5 — prevents duplicate vectors on re-index."""
    if category == "hotel":
        identity = str(item.get("hotel_name", "")).strip().lower()
    elif category == "flight":
        identity = (
            f"{str(item.get('airline', '')).strip().lower()}_"
            f"{str(item.get('flight_number', '')).strip().lower()}"
        )
    elif category == "activity":
        identity = str(item.get("name", "")).strip().lower()
    else:
        identity = ""

    if not identity:
        identity = json.dumps(item, sort_keys=True, default=str, ensure_ascii=False)

    raw_key = f"{session_id}|{category}|{identity}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_key))


def _format_text_payload(category: str, item: Dict[str, Any]) -> str:
    """Format a travel item into a text string optimal for embedding."""
    if category == "hotel":
        amenities = ", ".join(item.get("amenities", []) or [])
        return (
            f"Hotel: {item.get('hotel_name', '')}. "
            f"Description: {item.get('description', '')}. "
            f"Amenities: {amenities}."
        )
    if category == "activity":
        return (
            f"Activity: {item.get('name', '')}. "
            f"Type: {item.get('type', '')}. "
            f"Description: {item.get('description', '')}."
        )
    if category == "flight":
        return (
            f"Flight {item.get('flight_number', '')} by {item.get('airline', '')} "
            f"departing {item.get('departure_time', '')} "
            f"arriving {item.get('arrival_time', '')} "
            f"price {item.get('price', '')}."
        )
    return json.dumps(item, default=str, ensure_ascii=False)


def _decode_raw_data(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw if isinstance(raw, dict) else None


# ══════════════════════════════════════════════════════════════════════
#  pgvector Store
# ══════════════════════════════════════════════════════════════════════
class PgVectorTravelOptionStore:
    """Embed, store and rank per-session travel options in ``travel_vectors``."""

    def __init__(self, *, embedder: _Embedder) -> None:
        self._embedder = embedder
        self._dimension = int(embedder.dimension)

    def ensure_schema(self) -> None:
        """
        RUN ONCE AT STARTUP: enables pgvector, creates table & indexes.

        Indexes:
          • B-tree on (session_id, category, expires_at) — session filtering
          • HNSW  on embedding (halfvec_cosine_ops)      — ANN search
        """
        ensure_vector_extension()

        with get_connection() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS travel_vectors (
                    id          UUID PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    category    TEXT NOT NULL,
                    raw_data    JSONB NOT NULL,
                    expires_at  DOUBLE PRECISION NOT NULL,
                    created_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    embedding   vector({self._dimension})
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_travel_vectors_session_cat_expires
                ON travel_vectors (session_id, category, expires_at);
            """)

        # HNSW ANN index. pgvector's ``vector`` type only supports HNSW indexes
        # up to 2000 dimensions, but our embeddings are 3072-dim — so we index
        # a ``halfvec`` cast instead (supported up to 4000 dims). The search
        # query casts to the same halfvec expression so the planner can use
        # this index. Non-fatal: if index creation fails, semantic search
        # still works via an exact scan.
        try:
            with get_connection() as conn:
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_travel_vectors_embedding
                    ON travel_vectors
                    USING hnsw ((embedding::halfvec({self._dimension})) halfvec_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
        except Exception:
            _log_warning("vector.travel_options.index_unavailable")

        _log_info("vector.travel_options.ready")

    def verify(self) -> bool:
        """Health check: verify pgvector extension and travel_vectors table exist."""
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
                    if not cur.fetchone():
                        _log_error("vector.travel_options.extension_missing")
                        return False

                    cur.execute(
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_name = 'travel_vectors';"
                    )
                    if not cur.fetchone():
                        _log_error("vector.travel_options.table_missing")
                        return False
            return True
        except Exception:
            _log_error("vector.travel_options.verify_failed")
            return False

    def upsert(
        self,
        session_id: str,
        category: str,
        items: List[Dict[str, Any]],
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> bool:
        """
        Embed + upsert travel items into the pgvector table.

        Uses deterministic UUIDs (uuid5) so re-indexing the same item updates
        it in-place instead of creating duplicates.

        Returns True on success, False on failure.
        """
        if not items:
            return False

        expires_at = time.time() + (ttl_hours * 3600)
        cast = vector_cast_suffix()

        texts_to_embed = [_format_text_payload(category, item) for item in items]

        try:
            vectors = self._embedder.embed_documents(texts_to_embed)
            if len(vectors) != len(texts_to_embed):
                raise RuntimeError(
                    f"Embedding count mismatch: {len(vectors)} != {len(texts_to_embed)}"
                )

            records: List[tuple] = [
                (
                    _deterministic_id(session_id, category, item),
                    session_id,
                    category,
                    Jsonb(item),
                    expires_at,
                    prepare_vector_parameter(vector),
                )
                for item, vector in zip(items, vectors)
            ]

            # Batch INSERT with ON CONFLICT upsert
            with get_connection() as conn:
                with conn.cursor() as cur:
                    for i in range(0, len(records), UPSERT_BATCH_SIZE):
                        batch = records[i : i + UPSERT_BATCH_SIZE]
                        row_ph = f"(%s, %s, %s, %s, %s, %s{cast})"
                        placeholders = ", ".join([row_ph] * len(batch))

                        query = f"""
                            INSERT INTO travel_vectors
                                (id, session_id, category, raw_data, expires_at, embedding)
                            VALUES {placeholders}
                            ON CONFLICT (id) DO UPDATE SET
                                raw_data   = EXCLUDED.raw_data,
                                expires_at = EXCLUDED.expires_at,
                                embedding  = EXCLUDED.embedding;
                        """
                        params: list = []
                        for rec in batch:
                            params.extend(rec)

                        cur.execute(query, params)

            _log_info("vector.travel_options.upsert_succeeded")
            return True

        except Exception:
            _log_error("vector.travel_options.upsert_failed")
            return False

    def search(
        self,
        session_id: str,
        user_preferences: str,
        category: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over the user's session data using cosine distance.

        Uses HNSW index with ef_search=40 for a good accuracy/speed tradeoff.

        Returns ranked list of original item dicts with a 'relevance_score' field.
        """
        if not user_preferences.strip():
            return []

        try:
            query_vector = self._embedder.embed_query(user_preferences)
            if not query_vector:
                return []
            query_param = prepare_vector_parameter(query_vector)
            # Match the halfvec HNSW index built in ensure_schema() so the
            # planner can use it (vector<=> would force an exact scan on
            # 3072-dim embeddings, which cannot be HNSW-indexed as a plain vector).
            hv = f"::halfvec({self._dimension})"

            with get_connection() as conn:
                with conn.transaction():
                    with conn.cursor(row_factory=dict_row) as cur:
                        cur.execute(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}")
                        cur.execute(
                            f"""
                            SELECT
                                raw_data,
                                1 - (embedding{hv} <=> %s{hv}) AS relevance_score
                            FROM travel_vectors
                            WHERE session_id = %s
                            AND category   = %s
                            AND expires_at > %s
                            ORDER BY embedding{hv} <=> %s{hv}
                            LIMIT %s;
                            """,
                            (
                                query_param,
                                session_id,
                                category,
                                time.time(),
                                query_param,
                                top_k,
                            ),
                        )
                        rows = cur.fetchall()

            ranked_items: List[Dict[str, Any]] = []
            for row in rows:
                raw = _decode_raw_data(row["raw_data"])
                if raw is None:
                    continue
                raw["relevance_score"] = round(float(row["relevance_score"]), 3)
                ranked_items.append(raw)

            return ranked_items

        except Exception:
            _log_error("vector.travel_options.search_failed")
            return []

    def fetch_all(
        self,
        session_id: str,
        category: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve all non-expired items for a session+category (no semantic ranking)."""
        try:
            with get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT raw_data
                        FROM travel_vectors
                        WHERE session_id = %s
                          AND category   = %s
                          AND expires_at > %s
                        ORDER BY created_at DESC
                        LIMIT %s;
                        """,
                        (session_id, category, time.time(), limit),
                    )
                    rows = cur.fetchall()

            return [
                raw
                for row in rows
                if (raw := _decode_raw_data(row["raw_data"])) is not None
            ]

        except Exception:
            _log_error("vector.travel_options.fetch_failed")
            return []

    def clear_session(self, session_id: str) -> bool:
        """Delete all vectors associated with a specific session ID."""
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM travel_vectors WHERE session_id = %s;",
                        (session_id,),
                    )
            _log_info("vector.travel_options.clear_succeeded")
            return True
        except Exception:
            _log_error("vector.travel_options.clear_failed")
            return False

    def clear_expired(self, session_id: Optional[str] = None) -> int:
        """
        Best-effort sweep of expired vectors (call periodically via cron).

        Returns number of deleted rows.
        """
        try:
            now = time.time()
            with get_connection() as conn:
                with conn.cursor() as cur:
                    if session_id:
                        cur.execute(
                            "DELETE FROM travel_vectors "
                            "WHERE expires_at < %s AND session_id = %s;",
                            (now, session_id),
                        )
                    else:
                        cur.execute(
                            "DELETE FROM travel_vectors WHERE expires_at < %s;",
                            (now,),
                        )
                    deleted = cur.rowcount

            if deleted > 0:
                _log_info("vector.travel_options.sweep_succeeded")
            return deleted

        except Exception:
            _log_error("vector.travel_options.sweep_failed")
            return 0


__all__ = [
    "PgVectorTravelOptionStore",
]
