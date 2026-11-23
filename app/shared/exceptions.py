class SessionAccessDenied(PermissionError):
    """The requested thread is missing or is not owned by the current user."""


class SessionBusy(RuntimeError):
    """Another request is already running a turn on this session."""


class InvalidTurnRequest(ValueError):
    """A turn was asked for with arguments the runtime rejects before any I/O.

    Its own type so the API can answer 422 for it without treating every
    ``ValueError`` (pydantic's included) raised deep in a turn as the
    client's fault.
    """


class ProviderQuotaExhausted(RuntimeError):
    """A paid provider's quota (searches, credit) is used up for now.

    Different from "the provider failed": retrying right away can't help,
    and an empty result must not be read as "nothing found". Infrastructure
    errors subclass it next to their own failure type, so code that only
    knows the failure type keeps working.
    """


class RateLimitExceeded(RuntimeError):
    """The user has used up a request allowance for now."""

    def __init__(self, message: str, *, retry_after_seconds: int) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after_seconds = max(int(retry_after_seconds), 1)


class RateLimitUnavailable(RuntimeError):
    """The request limits couldn't be checked, and this request may not
    go ahead unchecked (production)."""


class PlanAlreadyRunning(RuntimeError):
    """This user already has a trip being planned."""


class PlanningCapacityFull(RuntimeError):
    """As many trips as the server plans at once are being planned now."""
