import pytest

from app.modules.profile.exceptions import ProfileUnavailable
from app.modules.profile.models import OnboardingProfile, onboarding_profile
from app.modules.profile.service import OnboardingService


ANA = OnboardingProfile(name="Ana", home_country="Malaysia", home_state="Selangor")


class Store:
    def __init__(self, row=None, *, read_error=None, write_ok=True, write_error=None):
        self.row = row or {}
        self.read_error = read_error
        self.write_ok = write_ok
        self.write_error = write_error
        self.writes = []

    def fetch_profile(self, user_id):
        if self.read_error:
            raise self.read_error
        return self.row

    def create_profile(self, user_id, profile):
        if self.write_error:
            raise self.write_error
        self.writes.append((user_id, dict(profile)))
        return self.write_ok


def test_all_three_fields_are_required_to_be_onboarded():
    assert onboarding_profile({"name": " Ana ", "home_country": "Malaysia", "home_state": "Selangor"}) == ANA
    assert onboarding_profile({"name": "Ana", "home_country": "Malaysia", "home_state": "  "}) is None
    assert onboarding_profile(None) is None


def test_get_projects_only_the_onboarding_fields():
    row = {**ANA.as_record(), "interests": ["temples"], "id": "user-1"}

    assert OnboardingService(store=Store(row)).get("user-1") == ANA


def test_an_unreadable_store_is_not_mistaken_for_a_new_user():
    with pytest.raises(ProfileUnavailable):
        OnboardingService(store=Store(read_error=RuntimeError("down"))).get("user-1")


def test_save_writes_only_onboarding_columns_and_notifies_caches():
    store, saved = Store(), []

    OnboardingService(store=store, on_saved=saved.append).save("user-1", ANA)

    assert store.writes == [("user-1", ANA.as_record())]
    assert saved == ["user-1"]


def test_a_failed_save_raises_and_a_failing_listener_does_not():
    for store in (Store(write_ok=False), Store(write_error=RuntimeError("down"))):
        with pytest.raises(ProfileUnavailable):
            OnboardingService(store=store).save("user-1", ANA)

    def broken(user_id):
        raise RuntimeError("cache gone")

    OnboardingService(store=Store(), on_saved=broken).save("user-1", ANA)
