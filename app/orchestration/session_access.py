"""Orchestration-facing access to durable trip-session ownership."""

from __future__ import annotations

import asyncio
from typing import Final, Protocol


NEW_SESSION_INITIALIZATION_LEASE_SECONDS: Final[int] = 960


class SessionOwnershipStore(Protocol):
    """Storage contract required by session-access orchestration."""

    def reserve(self, session_id: str, user_id: str) -> bool: ...

    def complete(self, session_id: str, user_id: str) -> bool: ...

    def reclaim_expired(
        self,
        session_id: str,
        user_id: str,
        lease_seconds: int,
    ) -> bool: ...

    def finalize_expired(
        self,
        session_id: str,
        user_id: str,
        lease_seconds: int,
    ) -> bool: ...

    def release(self, session_id: str, user_id: str) -> None: ...

    def backfill(self, session_id: str, user_id: str) -> bool: ...

    def verify(self, session_id: str, user_id: str) -> bool: ...


class SessionAccessService:
    """Async orchestration facade over session-ownership persistence."""

    def __init__(
        self,
        store: SessionOwnershipStore,
        initialization_lease_seconds: int = (
            NEW_SESSION_INITIALIZATION_LEASE_SECONDS
        ),
    ) -> None:
        if initialization_lease_seconds <= 0:
            raise ValueError(
                "initialization_lease_seconds must be greater than zero"
            )

        self._store = store
        self._initialization_lease_seconds = (
            initialization_lease_seconds
        )

    async def reserve_new_session(
        self,
        session_id: str,
        user_id: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._store.reserve,
            session_id,
            user_id,
        )

    async def complete_new_session_reservation(
        self,
        session_id: str,
        user_id: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._store.complete,
            session_id,
            user_id,
        )

    async def reclaim_expired_new_session_reservation(
        self,
        session_id: str,
        user_id: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._store.reclaim_expired,
            session_id,
            user_id,
            self._initialization_lease_seconds,
        )

    async def finalize_expired_new_session_reservation(
        self,
        session_id: str,
        user_id: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._store.finalize_expired,
            session_id,
            user_id,
            self._initialization_lease_seconds,
        )

    async def release_new_session_reservation(
        self,
        session_id: str,
        user_id: str,
    ) -> None:
        await asyncio.to_thread(
            self._store.release,
            session_id,
            user_id,
        )

    async def backfill_existing_session_owner(
        self,
        session_id: str,
        user_id: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._store.backfill,
            session_id,
            user_id,
        )

    async def verify_session_owner(
        self,
        session_id: str,
        user_id: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._store.verify,
            session_id,
            user_id,
        )
