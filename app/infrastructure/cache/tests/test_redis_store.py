"""Redis cache store over an in-memory Redis: entries, TTLs, leases, failures.

The fake runs the store's three Lua scripts as Python with the same meaning;
the scripts themselves are exercised against the real database in the live
check (docs/migration/28-redis-cache.md).
"""

import json
import threading

import pytest

from app.infrastructure.booking.cache import RedisBookingSearchCache
from app.infrastructure.budget.assessment_cache import RedisBudgetAssessmentCache
from app.infrastructure.cache import redis_store
from app.infrastructure.cache.redis_store import MAX_PAYLOAD_BYTES, RedisCacheStore, cache_digest
from app.infrastructure.cache.upstash import CacheCoolingDown, CacheUnavailableError
from app.infrastructure.places.cache import RedisPlaceSearchCache


class FakeRedis:
    """GET / SET (EX, PX, NX) / DEL and the store's scripts, with a clock."""

    def __init__(self):
        self.now = 0.0
        self.data = {}  # key -> (value, expires_at or None)
        self.commands = []
        self.fail = None

    def _alive(self, key):
        item = self.data.get(key)
        if item is None or (item[1] is not None and item[1] <= self.now):
            self.data.pop(key, None)
            return None
        return item[0]

    def _set(self, key, value, *options):
        options = [str(o).upper() if isinstance(o, str) else o for o in options]
        if "NX" in options and self._alive(key) is not None:
            return None
        expires = None
        if "EX" in options:
            expires = self.now + int(options[options.index("EX") + 1])
        if "PX" in options:
            expires = self.now + int(options[options.index("PX") + 1]) / 1000
        self.data[key] = (value, expires)
        return "OK"

    def execute(self, command):
        self.commands.append(list(command))
        if self.fail is not None:
            raise self.fail
        name, *args = [str(part) for part in command]

        if name == "GET":
            return self._alive(args[0])
        if name == "SET":
            return self._set(*args)
        if name == "DEL":
            return int(self.data.pop(args[0], None) is not None)
        if name == "EVAL":
            script, count, rest = args[0], int(args[1]), args[2:]
            keys, argv = rest[:count], rest[count:]
            return self._script(script, keys, argv)
        raise AssertionError(f"unexpected command {name}")

    def _script(self, script, keys, argv):
        if script == redis_store._CLAIM_SCRIPT:
            value = self._alive(keys[0])
            if value is not None:
                return ["cached", value]
            return ["claimed"] if self._set(keys[1], argv[0], "NX", "PX", argv[1]) else ["busy"]
        if script == redis_store._PUBLISH_SCRIPT:
            if self._alive(keys[1]) != argv[0]:
                return 0
            self._set(keys[0], argv[1], "EX", argv[2])
            self.data.pop(keys[1], None)
            return 1
        if script == redis_store._RELEASE_SCRIPT:
            if self._alive(keys[0]) != argv[0]:
                return 0
            self.data.pop(keys[0], None)
            return 1
        raise AssertionError("unknown script")


KEY = {"origin": " Malaysia ", "currency": "USD", "adults": 2}


@pytest.fixture
def redis():
    return FakeRedis()


@pytest.fixture
def store(redis):
    return RedisCacheStore(client=redis)


# ── Keys ────────────────────────────────────────────────────────────


def test_keys_are_normalised_namespaced_and_hashed(store, redis):
    store.set("flight", KEY, [{"price": 1}], ttl_hours=1)

    [(entry, _)] = redis.data.items()
    assert entry == f"travel:cache:flight:{cache_digest(KEY)}"
    assert cache_digest(KEY) == cache_digest({"adults": 2, "currency": "usd", "origin": "malaysia"})
    assert "Malaysia" not in entry


def test_the_digest_matches_the_old_firestore_document_ids():
    # Same normalisation as before, so the same trip finds the same entry.
    import hashlib

    old = hashlib.sha256(
        json.dumps({"origin": "malaysia", "currency": "usd", "adults": 2}, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert cache_digest(KEY) == old


@pytest.mark.parametrize("prefix", ["", " ", "a:b"])
def test_rejects_a_prefix_that_could_collide(store, prefix):
    with pytest.raises(ValueError):
        store.get(prefix, KEY)


def test_rejects_a_bad_keyspace():
    for keyspace in ("", "a:b"):
        with pytest.raises(ValueError):
            RedisCacheStore(client=None, keyspace=keyspace)


# ── Entries ─────────────────────────────────────────────────────────


def test_round_trip_and_expiry(store, redis):
    payload = [{"name": "Hotel Ōkura", "price": 120.5, "tags": ["a", "b"], "ok": True, "none": None}]

    assert store.set("hotel_v2", KEY, payload, ttl_hours=2)
    assert store.get("hotel_v2", KEY) == payload
    assert store.get("place_search_v2", KEY) is None  # another feature's entry

    redis.now += 2 * 3600
    assert store.get("hotel_v2", KEY) is None


def test_set_uses_the_redis_ttl(store, redis):
    store.set("flight", KEY, [1], ttl_hours=24)

    assert redis.commands[-1][-2:] == ["EX", 24 * 3600]


@pytest.mark.parametrize(
    "payload",
    [None, {"when": object()}, {"x": float("nan")}, "x" * (MAX_PAYLOAD_BYTES + 1)],
    ids=["none", "not-json", "nan", "too-large"],
)
def test_refuses_payloads_it_cannot_store_faithfully(store, redis, payload):
    assert store.set("flight", KEY, payload, ttl_hours=1) is False
    assert redis.commands == []


def test_refuses_a_non_positive_ttl(store, redis):
    assert store.set("flight", KEY, [1], ttl_hours=0) is False
    assert redis.commands == []


def test_an_unreadable_entry_is_a_miss_and_is_deleted(store, redis):
    entry = f"travel:cache:flight:{cache_digest(KEY)}"
    redis.data[entry] = ("{not json", None)

    assert store.get("flight", KEY) is None
    assert entry not in redis.data


def test_invalidate_deletes_the_entry(store):
    store.set("flight", KEY, [1], ttl_hours=1)

    assert store.invalidate("flight", KEY)
    assert store.get("flight", KEY) is None


# ── Failures and the disabled store ─────────────────────────────────


@pytest.mark.parametrize("failure", [CacheUnavailableError("down"), CacheCoolingDown("wait"), RuntimeError("bug")])
def test_every_operation_fails_soft(store, redis, failure):
    redis.fail = failure

    assert store.get("flight", KEY) is None
    assert store.set("flight", KEY, [1], ttl_hours=1) is False
    assert store.invalidate("flight", KEY) is False
    assert store.claim("budget_assessment", KEY, lease_seconds=60) == ("error", None, None)
    assert store.complete_claim("budget_assessment", KEY, [1], claim_id="c", ttl_hours=1) is False
    assert store.release_claim("budget_assessment", KEY, claim_id="c") is False


def test_without_a_client_the_store_is_an_empty_cache():
    store = RedisCacheStore(client=None)

    assert not store.enabled
    assert store.get("flight", KEY) is None
    assert store.set("flight", KEY, [1], ttl_hours=1) is False
    assert store.claim("budget_assessment", KEY, lease_seconds=60) == ("error", None, None)
    store.close()


def test_close_closes_the_client():
    class Closable(FakeRedis):
        closed = False

        def close(self):
            self.closed = True

    client = Closable()
    RedisCacheStore(client=client).close()
    assert client.closed


# ── Leases ──────────────────────────────────────────────────────────


def test_one_caller_claims_the_others_are_busy_then_see_the_entry(store, redis):
    status, payload, claim_id = store.claim("budget_assessment", KEY, lease_seconds=120)
    assert (status, payload) == ("claimed", None) and claim_id

    assert store.claim("budget_assessment", KEY, lease_seconds=120) == ("busy", None, None)

    assert store.complete_claim("budget_assessment", KEY, {"id": "a1"}, claim_id=claim_id, ttl_hours=1)
    assert store.claim("budget_assessment", KEY, lease_seconds=120) == ("cached", {"id": "a1"}, None)
    assert store.get("budget_assessment", KEY) == {"id": "a1"}
    assert not any(key.startswith("travel:lease:") for key in redis.data)


def test_only_the_lease_holder_can_publish_or_release(store):
    _, _, claim_id = store.claim("budget_assessment", KEY, lease_seconds=120)

    assert not store.complete_claim("budget_assessment", KEY, {"id": "x"}, claim_id="someone-else", ttl_hours=1)
    assert not store.release_claim("budget_assessment", KEY, claim_id="someone-else")
    assert store.get("budget_assessment", KEY) is None

    assert store.release_claim("budget_assessment", KEY, claim_id=claim_id)
    assert store.claim("budget_assessment", KEY, lease_seconds=120)[0] == "claimed"


def test_an_abandoned_lease_expires_by_itself(store, redis):
    _, _, first = store.claim("budget_assessment", KEY, lease_seconds=120)

    redis.now += 120
    status, _, second = store.claim("budget_assessment", KEY, lease_seconds=120)

    assert status == "claimed" and second != first
    # The first caller lost its lease and can no longer publish over the second.
    assert not store.complete_claim("budget_assessment", KEY, {"id": "late"}, claim_id=first, ttl_hours=1)


def test_a_lease_needs_a_positive_length(store, redis):
    assert store.claim("budget_assessment", KEY, lease_seconds=0) == ("error", None, None)
    assert redis.commands == []


@pytest.mark.parametrize("answer", [None, [], ["cached"], ["cached", "{bad"], ["surprise"], "OK"])
def test_an_unexpected_claim_answer_is_an_error(store, redis, answer):
    redis.execute = lambda command: answer

    assert store.claim("budget_assessment", KEY, lease_seconds=60) == ("error", None, None)


def test_waiters_wake_when_a_lease_in_this_process_ends(store):
    _, _, claim_id = store.claim("budget_assessment", KEY, lease_seconds=120)
    woke = threading.Event()

    def wait():
        store.wait_for_claim_change(timeout_seconds=5)
        woke.set()

    waiter = threading.Thread(target=wait)
    waiter.start()
    while not store._claim_condition._waiters:
        pass
    store.complete_claim("budget_assessment", KEY, {"id": "a"}, claim_id=claim_id, ttl_hours=1)
    waiter.join(timeout=5)

    assert woke.is_set()


# ── Feature adapters ────────────────────────────────────────────────


def test_feature_adapters_keep_their_own_prefixes(store, redis):
    RedisBookingSearchCache(store=store).set(namespace="flight", key=KEY, payload=[1], ttl_hours=1)
    RedisPlaceSearchCache(store=store).set(namespace="place_search_v2", key=KEY, payload=[2], ttl_hours=24)
    budget = RedisBudgetAssessmentCache(store=store)
    _, _, claim_id = budget.claim(KEY, lease_seconds=60)
    budget.publish(KEY, {"id": 3}, claim_id=claim_id, ttl_hours=1)

    assert {key.split(":")[2] for key in redis.data} == {"flight", "place_search_v2", "budget_assessment"}
    assert RedisBookingSearchCache(store=store).get(namespace="flight", key=KEY) == [1]
    assert RedisPlaceSearchCache(store=store).get(namespace="place_search_v2", key=KEY) == [2]
    assert budget.get(KEY) == {"id": 3}


def test_booking_and_place_adapters_swallow_store_bugs():
    class Broken:
        def get(self, *args, **kwargs):
            raise RuntimeError("bug")

        set = get

    for cache in (RedisBookingSearchCache(store=Broken()), RedisPlaceSearchCache(store=Broken())):
        assert cache.get(namespace="flight", key=KEY) is None
        assert cache.set(namespace="flight", key=KEY, payload=[1], ttl_hours=1) is False
