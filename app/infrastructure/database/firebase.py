"""
Firebase Firestore cache layer.

Provides a TTL-based key-value cache backed by Firestore documents.
Keys are deterministic SHA-256 hashes of a normalised key mapping, so
'USD' and 'usd' collide (case-insensitive).

The credential is injected through ``FirestoreCacheStore``'s constructor and
the Firestore client is created lazily on first use, so importing this module
needs neither configuration nor a network.

Thread-safety:
  • The process-wide firebase_admin app is initialised under _init_lock.
  • Firestore client itself is thread-safe for reads & writes.

Resilience:
  • Handles the credential as either a file path or raw JSON string.
  • Every public operation fails soft (None/False) instead of raising.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

import firebase_admin
from firebase_admin import credentials, firestore

from app.shared.logging import safe_log

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────
DEFAULT_COLLECTION = "api_cache"
DEFAULT_TTL_HOURS = 1
FIRESTORE_BATCH_LIMIT = 450  # Firestore batch max is 500; leave headroom
FIRESTORE_DOC_MAX_BYTES = 900_000  # Firestore doc max is ~1 MiB
MAX_CLEAR_PAGES = 50

# firebase_admin keeps one default app per process, so its initialisation is
# guarded at module level even though stores are separate instances.
_init_lock: threading.Lock = threading.Lock()

ClaimStatus = Literal["cached", "claimed", "busy", "error"]


# ── Cache key generation ────────────────────────────────────────────


def _generate_cache_key(prefix: str, key: Mapping[str, Any]) -> str:
    """
    Deterministic SHA-256 hash of the normalised key mapping → document ID.

    String values are lowercased + trimmed so 'USD' and 'usd' collide.
    """
    normalised = {
        k: (str(v).strip().lower() if isinstance(v, str) else v)
        for k, v in key.items()
    }
    param_string = json.dumps(normalised, sort_keys=True, default=str)
    digest = hashlib.sha256(param_string.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


def _payload_too_large(payload: Any) -> bool:
    try:
        payload_bytes = len(json.dumps(payload, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        payload_bytes = 0
    return payload_bytes > FIRESTORE_DOC_MAX_BYTES


class FirestoreCacheStore:
    """TTL cache and creator-lease primitives over Firestore documents."""

    def __init__(self, *, credential_json: str) -> None:
        if not credential_json or not credential_json.strip():
            raise ValueError("credential_json must not be empty")

        self._credential_json = credential_json.strip()
        self._client: Optional[firestore.Client] = None
        self._claim_condition = threading.Condition()

    # ── Client ──────────────────────────────────────────────────────

    def _get_db(self) -> firestore.Client:
        """Return the lazily created Firestore client."""
        if self._client is not None:
            return self._client

        with _init_lock:
            if self._client is not None:  # double-checked locking
                return self._client

            if not firebase_admin._apps:
                cred_str = self._credential_json

                if os.path.isfile(cred_str):
                    # It's a file path
                    cred = credentials.Certificate(cred_str)
                else:
                    # Assume it's a raw JSON string
                    try:
                        cred = credentials.Certificate(json.loads(cred_str))
                    except (json.JSONDecodeError, ValueError):
                        _log("cache.firebase.credentials_invalid")
                        raise

                firebase_admin.initialize_app(cred)

            self._client = firestore.client()
            _log("cache.firebase.initialized")
            return self._client

    def _doc_ref(self, collection: str, prefix: str, key: Mapping[str, Any]):
        return (
            self._get_db()
            .collection(collection)
            .document(_generate_cache_key(prefix, key))
        )

    # ── Cache read / write ──────────────────────────────────────────

    def get(
        self,
        collection: str,
        prefix: str,
        key: Mapping[str, Any],
    ) -> Optional[Any]:
        """
        Retrieve cached data from Firestore.

        Returns the payload on hit, None on miss / expiry / error.
        Expired documents are deleted best-effort.
        """
        try:
            doc_ref = self._doc_ref(collection, prefix, key)
            doc = doc_ref.get()
        except Exception:
            _log("cache.firebase.read_failed")
            return None

        if not doc.exists:
            _log("cache.firebase.miss")
            return None

        data = doc.to_dict() or {}
        expires_at = data.get("expires_at")

        # TTL expired?
        now = datetime.now(timezone.utc)
        if expires_at is None or now > expires_at:
            _log("cache.firebase.expired")
            claim_expires_at = data.get("claim_expires_at")
            active_claim = (
                data.get("claim_id")
                and isinstance(claim_expires_at, datetime)
                and claim_expires_at > now
            )
            if not active_claim:
                # Best-effort cleanup. An active creator lease owns an
                # expired predecessor until it publishes.
                try:
                    doc_ref.delete()
                except Exception:
                    pass
            return None

        _log("cache.firebase.hit")
        return data.get("payload")

    def set(
        self,
        collection: str,
        prefix: str,
        key: Mapping[str, Any],
        payload: Any,
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> bool:
        """
        Write payload to Firestore with an expiry timestamp.

        Returns True on success, False on failure.
        """
        if ttl_hours <= 0:
            _log("cache.firebase.invalid_ttl")
            return False

        # Size guard — Firestore doc max is ~1 MiB
        if _payload_too_large(payload):
            _log("cache.firebase.payload_too_large")
            return False

        now = datetime.now(timezone.utc)

        try:
            self._doc_ref(collection, prefix, key).set(
                {
                    "payload": payload,
                    "expires_at": now + timedelta(hours=ttl_hours),
                    "created_at": now,
                }
            )
            _log("cache.firebase.write_succeeded")
            return True
        except Exception:
            _log("cache.firebase.write_failed")
            return False

    # ── Creator leases (single-flight construction) ─────────────────

    def claim(
        self,
        collection: str,
        prefix: str,
        key: Mapping[str, Any],
        *,
        lease_seconds: int,
    ) -> tuple[ClaimStatus, Any, str | None]:
        """Atomically return a cache hit or lease one empty key for construction."""
        if lease_seconds <= 0:
            return "error", None, None

        now = datetime.now(timezone.utc)
        claim_id = str(uuid.uuid4())
        claim_expires_at = now + timedelta(seconds=lease_seconds)

        try:
            database = self._get_db()
            doc_ref = self._doc_ref(collection, prefix, key)
            transaction = database.transaction()

            @firestore.transactional
            def _claim(transaction):
                snapshot = doc_ref.get(transaction=transaction)
                data = (snapshot.to_dict() or {}) if snapshot.exists else {}
                expires_at = data.get("expires_at")
                if (
                    data.get("payload") is not None
                    and isinstance(expires_at, datetime)
                    and expires_at > now
                ):
                    return "cached", data["payload"], None
                current_claim_expiry = data.get("claim_expires_at")
                if (
                    data.get("claim_id")
                    and isinstance(current_claim_expiry, datetime)
                    and current_claim_expiry > now
                ):
                    return "busy", None, None
                transaction.set(
                    doc_ref,
                    {
                        "claim_id": claim_id,
                        "claim_expires_at": claim_expires_at,
                        "claim_created_at": now,
                    },
                )
                return "claimed", None, claim_id

            return _claim(transaction)
        except Exception:
            _log("cache.firebase.claim_failed")
            return "error", None, None

    def complete_claim(
        self,
        collection: str,
        prefix: str,
        key: Mapping[str, Any],
        payload: Any,
        *,
        claim_id: str,
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> bool:
        """Publish payload only if this caller still owns the atomic cache lease."""
        if ttl_hours <= 0 or not claim_id:
            return False
        if _payload_too_large(payload):
            return False

        now = datetime.now(timezone.utc)
        try:
            database = self._get_db()
            doc_ref = self._doc_ref(collection, prefix, key)
            transaction = database.transaction()

            @firestore.transactional
            def _complete(transaction):
                snapshot = doc_ref.get(transaction=transaction)
                data = (snapshot.to_dict() or {}) if snapshot.exists else {}
                if data.get("claim_id") != claim_id:
                    return False
                transaction.set(
                    doc_ref,
                    {
                        "payload": payload,
                        "expires_at": now + timedelta(hours=ttl_hours),
                        "created_at": now,
                    },
                )
                return True

            completed = _complete(transaction)
        except Exception:
            _log("cache.firebase.publish_failed")
            completed = False
        if completed:
            with self._claim_condition:
                self._claim_condition.notify_all()
        return completed

    def release_claim(
        self,
        collection: str,
        prefix: str,
        key: Mapping[str, Any],
        *,
        claim_id: str,
    ) -> bool:
        """Release only the matching failed creator lease so another caller retries."""
        if not claim_id:
            return False
        try:
            database = self._get_db()
            doc_ref = self._doc_ref(collection, prefix, key)
            transaction = database.transaction()

            @firestore.transactional
            def _release(transaction):
                snapshot = doc_ref.get(transaction=transaction)
                data = (snapshot.to_dict() or {}) if snapshot.exists else {}
                if data.get("claim_id") != claim_id:
                    return False
                transaction.delete(doc_ref)
                return True

            released = _release(transaction)
        except Exception:
            _log("cache.firebase.release_failed")
            released = False
        if released:
            with self._claim_condition:
                self._claim_condition.notify_all()
        return released

    def wait_for_claim_change(self, *, timeout_seconds: float) -> None:
        """Wait briefly for a local claimant; remote claimants are re-polled safely."""
        with self._claim_condition:
            self._claim_condition.wait(timeout=max(0.0, timeout_seconds))

    # ── Invalidation ────────────────────────────────────────────────

    def clear_collection(self, collection: str = DEFAULT_COLLECTION) -> int:
        """
        Delete all documents in `collection` in paginated batches.

        Returns the total number of documents deleted.
        """
        _log("cache.firebase.clear_started")
        total_deleted = 0

        try:
            client = self._get_db()
            for _page in range(MAX_CLEAR_PAGES):
                docs = list(
                    client.collection(collection).limit(FIRESTORE_BATCH_LIMIT).stream()
                )

                if not docs:
                    break

                batch = client.batch()
                for doc in docs:
                    batch.delete(doc.reference)
                batch.commit()
                total_deleted += len(docs)

                if len(docs) < FIRESTORE_BATCH_LIMIT:
                    break

            _log("cache.firebase.clear_completed")
        except Exception:
            _log("cache.firebase.clear_failed")

        return total_deleted

    def invalidate(
        self,
        collection: str,
        prefix: str,
        key: Mapping[str, Any],
    ) -> bool:
        """
        Delete a single cached entry identified by prefix + key.

        Returns True on success, False on failure.
        """
        try:
            self._doc_ref(collection, prefix, key).delete()
            _log("cache.firebase.invalidate_succeeded")
            return True
        except Exception:
            _log("cache.firebase.invalidate_failed")
            return False


# -------- Logging ------------------------------
_FIREBASE_EVENT_LEVELS = {
    "cache.firebase.credentials_invalid": logging.ERROR,
    "cache.firebase.initialized": logging.INFO,
    "cache.firebase.read_failed": logging.WARNING,
    "cache.firebase.miss": logging.DEBUG,
    "cache.firebase.expired": logging.DEBUG,
    "cache.firebase.hit": logging.DEBUG,
    "cache.firebase.invalid_ttl": logging.WARNING,
    "cache.firebase.payload_too_large": logging.ERROR,
    "cache.firebase.write_succeeded": logging.DEBUG,
    "cache.firebase.write_failed": logging.ERROR,
    "cache.firebase.claim_failed": logging.ERROR,
    "cache.firebase.publish_failed": logging.ERROR,
    "cache.firebase.release_failed": logging.ERROR,
    "cache.firebase.clear_started": logging.INFO,
    "cache.firebase.clear_completed": logging.INFO,
    "cache.firebase.clear_failed": logging.ERROR,
    "cache.firebase.invalidate_succeeded": logging.DEBUG,
    "cache.firebase.invalidate_failed": logging.ERROR,
}


def _log(event: str) -> None:
    level = _FIREBASE_EVENT_LEVELS.get(event)

    if level is None:
        return

    safe_log(
        logger,
        level,
        event,
    )


__all__ = [
    "DEFAULT_COLLECTION",
    "ClaimStatus",
    "FirestoreCacheStore",
]
