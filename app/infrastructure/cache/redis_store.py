"""Redis-backed TTL cache, plus creator leases.

An entry lives at ``<keyspace>:cache:<prefix>:<digest>`` and expires on its
own (Redis TTL), so nothing has to clean up after it. The digest is a
SHA-256 of the normalised key mapping: string values are trimmed and
lowercased, so 'USD' and 'usd' share an entry. Payloads are stored as
strict JSON; anything else is refused rather than silently changed.

Leases ("claims") let one caller build an expensive entry while the others
wait for it, instead of every caller asking the provider at once. A lease is
its own key (``<keyspace>:lease:...``) that expires by itself, so a caller
that crashes never blocks the entry for longer than its lease. Claiming,
publishing and releasing each run as one Lua script, so no other caller can
act between the check and the change.

Every public operation fails soft (None / False / "error") instead of
raising: a cache problem must never fail a request. Without a client the
store is disabled and behaves as an empty cache that can't save anything.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal, Protocol

from app.infrastructure.cache.upstash import CacheCoolingDown
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

DEFAULT_KEYSPACE: Final = "travel"
# Far below Upstash's 10 MB request limit; keeps one runaway payload from
# filling the free plan's 256 MB.
MAX_PAYLOAD_BYTES: Final = 1_000_000

ClaimStatus = Literal["cached", "claimed", "busy", "error"]

# KEYS: entry, lease. ARGV: claim id, lease milliseconds.
_CLAIM_SCRIPT: Final = """
local value = redis.call('GET', KEYS[1])
if value then return {'cached', value} end
if redis.call('SET', KEYS[2], ARGV[1], 'NX', 'PX', ARGV[2]) then return {'claimed'} end
return {'busy'}
"""
# KEYS: entry, lease. ARGV: claim id, payload, TTL seconds.
_PUBLISH_SCRIPT: Final = """
if redis.call('GET', KEYS[2]) ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
redis.call('DEL', KEYS[2])
return 1
"""
# KEYS: lease. ARGV: claim id.
_RELEASE_SCRIPT: Final = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('DEL', KEYS[1])
return 1
"""


class RedisCommandClient(Protocol):
    """Runs one Redis command and returns its result; raises on failure."""

    def execute(self, command: Sequence[str | int]) -> Any:
        ...


def cache_digest(key: Mapping[str, Any]) -> str:
    """SHA-256 of the normalised key mapping."""

    normalised = {
        name: (value.strip().lower() if isinstance(value, str) else value)
        for name, value in key.items()
    }
    text = json.dumps(normalised, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RedisCacheStore:
    """TTL cache and creator-lease primitives over Redis keys."""

    def __init__(self, *, client: RedisCommandClient | None, keyspace: str = DEFAULT_KEYSPACE) -> None:
        if not keyspace.strip() or ":" in keyspace:
            raise ValueError("keyspace must be a non-empty name without ':'")

        self._client = client
        self._keyspace = keyspace.strip()
        self._claim_condition = threading.Condition()

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def close(self) -> None:
        close = getattr(self._client, "close", None)

        if callable(close):
            close()

    # ── Cache read / write ──────────────────────────────────────────

    def get(self, prefix: str, key: Mapping[str, Any]) -> Any | None:
        """The payload, or None on a miss, expiry or error."""

        entry = self._entry_key(prefix, key)
        raw = self._run(["GET", entry], "cache.redis.read_failed")

        if raw is None:
            return None

        payload = _decode(raw)

        if payload is None:
            # Unreadable entries would otherwise be read and ignored until
            # they expire.
            self._run(["DEL", entry], "cache.redis.invalidate_failed")
            safe_log(logger, logging.WARNING, "cache.redis.entry_unreadable")

        return payload

    def set(self, prefix: str, key: Mapping[str, Any], payload: Any, *, ttl_hours: int) -> bool:
        """Store ``payload`` for ``ttl_hours``; False when it wasn't saved."""

        encoded = _encode(payload)

        if ttl_hours <= 0 or encoded is None:
            return False

        result = self._run(
            ["SET", self._entry_key(prefix, key), encoded, "EX", ttl_hours * 3600],
            "cache.redis.write_failed",
        )
        return result == "OK"

    def invalidate(self, prefix: str, key: Mapping[str, Any]) -> bool:
        """Delete one entry; False when that failed."""

        return self._run(["DEL", self._entry_key(prefix, key)], "cache.redis.invalidate_failed") is not None

    # ── Creator leases (single-flight construction) ─────────────────

    def claim(
        self,
        prefix: str,
        key: Mapping[str, Any],
        *,
        lease_seconds: int,
    ) -> tuple[ClaimStatus, Any, str | None]:
        """Atomically: return a fresh entry, or lease the key.

        ``cached``: the payload is returned. ``claimed``: the caller builds the
        entry, then calls ``complete_claim`` or ``release_claim`` with the
        returned claim id. ``busy``: someone else holds the lease. ``error``:
        the cache failed or is disabled.
        """

        if lease_seconds <= 0:
            return "error", None, None

        claim_id = uuid.uuid4().hex
        result = self._run(
            [
                "EVAL",
                _CLAIM_SCRIPT,
                2,
                self._entry_key(prefix, key),
                self._lease_key(prefix, key),
                claim_id,
                lease_seconds * 1000,
            ],
            "cache.redis.claim_failed",
        )

        if not isinstance(result, list) or not result:
            return "error", None, None

        if result[0] == "claimed":
            return "claimed", None, claim_id

        if result[0] == "busy":
            return "busy", None, None

        payload = _decode(result[1]) if result[0] == "cached" and len(result) == 2 else None

        if payload is None:
            return "error", None, None

        return "cached", payload, None

    def complete_claim(
        self,
        prefix: str,
        key: Mapping[str, Any],
        payload: Any,
        *,
        claim_id: str,
        ttl_hours: int,
    ) -> bool:
        """Publish ``payload``, but only if this caller still holds the lease."""

        encoded = _encode(payload)

        if ttl_hours <= 0 or not claim_id or encoded is None:
            return False

        result = self._run(
            [
                "EVAL",
                _PUBLISH_SCRIPT,
                2,
                self._entry_key(prefix, key),
                self._lease_key(prefix, key),
                claim_id,
                encoded,
                ttl_hours * 3600,
            ],
            "cache.redis.publish_failed",
        )
        return self._notify_if(result == 1)

    def release_claim(self, prefix: str, key: Mapping[str, Any], *, claim_id: str) -> bool:
        """Drop this caller's lease (it failed to build), so another caller
        can try; someone else's lease is left alone."""

        if not claim_id:
            return False

        result = self._run(
            ["EVAL", _RELEASE_SCRIPT, 1, self._lease_key(prefix, key), claim_id],
            "cache.redis.release_failed",
        )
        return self._notify_if(result == 1)

    def wait_for_claim_change(self, *, timeout_seconds: float) -> None:
        """Wake early when a lease in this process completes or is released.
        Leases held by other processes are noticed when the caller polls again."""

        with self._claim_condition:
            self._claim_condition.wait(timeout=max(0.0, timeout_seconds))

    # ── Internals ───────────────────────────────────────────────────

    def _entry_key(self, prefix: str, key: Mapping[str, Any]) -> str:
        return f"{self._keyspace}:cache:{_prefix(prefix)}:{cache_digest(key)}"

    def _lease_key(self, prefix: str, key: Mapping[str, Any]) -> str:
        return f"{self._keyspace}:lease:{_prefix(prefix)}:{cache_digest(key)}"

    def _run(self, command: Sequence[str | int], failure_event: str) -> Any:
        """The command's result, or None when it couldn't run."""

        if self._client is None:
            return None

        try:
            return self._client.execute(command)
        except CacheCoolingDown:
            # The client logged when the cooldown started.
            return None
        except Exception:
            safe_log(logger, logging.WARNING, failure_event)
            return None

    def _notify_if(self, changed: bool) -> bool:
        if changed:
            with self._claim_condition:
                self._claim_condition.notify_all()
        return changed


def _prefix(prefix: str) -> str:
    cleaned = prefix.strip()

    if not cleaned or ":" in cleaned:
        raise ValueError("prefix must be a non-empty name without ':'")

    return cleaned


def _encode(payload: Any) -> str | None:
    """Strict JSON, or None for a payload that can't be stored as-is.

    None itself isn't stored: a read couldn't tell it from a miss.
    """

    if payload is None:
        return None

    try:
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        safe_log(logger, logging.ERROR, "cache.redis.payload_not_json")
        return None

    if len(text.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        safe_log(logger, logging.ERROR, "cache.redis.payload_too_large")
        return None

    return text


def _decode(raw: Any) -> Any | None:
    if not isinstance(raw, str):
        return None

    try:
        return json.loads(raw)
    except ValueError:
        return None


__all__ = [
    "DEFAULT_KEYSPACE",
    "MAX_PAYLOAD_BYTES",
    "ClaimStatus",
    "RedisCacheStore",
    "RedisCommandClient",
    "cache_digest",
]
