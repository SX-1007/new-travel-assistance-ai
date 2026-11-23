"""Gemini failures exactly as ``langchain-google-genai`` raises them.

The library turns the ``google-genai`` SDK's ``ClientError`` / ``ServerError``
into LangChain's provider-neutral model errors (``GoogleRateLimitError`` is a
``ModelRateLimitError``, …). Building them through the library's own
conversion keeps these tests honest: when an upgrade changes the types or
the message, the tests that depend on them fail.
"""

from __future__ import annotations

from google.genai.errors import ClientError, ServerError
from langchain_google_genai.chat_models import _handle_client_error, _handle_server_error

_STATUS = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    503: "UNAVAILABLE",
}

PER_MINUTE_QUOTA = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
PER_DAY_QUOTA = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"


def gemini_error(code: int, message: str, *, quota_id: str | None = None) -> Exception:
    """The exception a Gemini call raises for this HTTP failure."""

    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]}]
    body = {
        "error": {
            "code": code,
            "message": message,
            "status": _STATUS[code],
            "details": details if quota_id else [],
        }
    }

    try:
        if code >= 500:
            _handle_server_error(ServerError(code, body))
        else:
            _handle_client_error(ClientError(code, body), {"model": "gemini-test"})
    except Exception as error:  # noqa: BLE001 - the conversion always raises
        return error

    raise AssertionError("the library did not convert the error")


def per_minute_quota(retry_hint: str = "Please retry in 22.555s.") -> Exception:
    return gemini_error(429, f"You exceeded your current quota. {retry_hint}", quota_id=PER_MINUTE_QUOTA)


def per_day_quota() -> Exception:
    return gemini_error(429, "You exceeded your current quota.", quota_id=PER_DAY_QUOTA)
