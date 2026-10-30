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
