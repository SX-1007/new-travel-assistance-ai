"""Read and save the onboarding profile.

The profile row also holds long-term memory columns (interests, dietary
needs, ...); only the three onboarding columns are read or written here.
"""

from __future__ import annotations

from collections.abc import Callable

from app.modules.profile.exceptions import ProfileUnavailable
from app.modules.profile.models import OnboardingProfile, onboarding_profile
from app.modules.profile.ports import ProfileStorePort


class OnboardingService:
    def __init__(
        self,
        *,
        store: ProfileStorePort,
        on_saved: Callable[[str], None] | None = None,
    ) -> None:
        self._store = store
        # Lets caches of this user's profile drop their copy (wired in the
        # container); never allowed to fail the save.
        self._on_saved = on_saved

    def get(self, user_id: str) -> OnboardingProfile | None:
        """The user's onboarding profile, or None until it is complete.

        Raises ``ProfileUnavailable`` when the store can't be read, so an
        outage is never mistaken for "not onboarded yet".
        """

        try:
            record = self._store.fetch_profile(user_id)
        except Exception as exc:
            raise ProfileUnavailable("Profile could not be read.") from exc

        return onboarding_profile(record)

    def save(self, user_id: str, profile: OnboardingProfile) -> None:
        """Store the onboarding fields; other profile columns are kept."""

        try:
            saved = self._store.create_profile(user_id, profile.as_record())
        except Exception as exc:
            raise ProfileUnavailable("Profile could not be saved.") from exc

        if not saved:
            raise ProfileUnavailable("Profile could not be saved.")

        if self._on_saved is not None:
            try:
                self._on_saved(user_id)
            except Exception:
                pass


__all__ = [
    "OnboardingService",
]
