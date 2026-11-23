"""The deployment values ``main.py`` needs before any settings load.

``main.py`` must stay importable without a complete ``.env`` (the tests
import it with none), so it can't load ``Settings``, whose required values
would then fail. It reads only these few values, with the same precedence
as ``Settings``: the process environment first, then ``.env``.
"""

from __future__ import annotations

import os
from typing import Final, Literal

from dotenv import dotenv_values


Environment = Literal["development", "production"]

_ENV_FILE: Final = ".env"
_ENVIRONMENTS: Final = ("development", "production")


def deployment_value(name: str, default: str = "") -> str:
    """One setting from the environment, else ``.env``, else ``default``."""

    value = os.environ.get(name)

    if value is None:
        value = dotenv_values(_ENV_FILE).get(name) if os.path.exists(_ENV_FILE) else None

    return (value if value is not None else default).strip()


def app_environment() -> Environment:
    """``ENVIRONMENT``: "development" (the default) or "production".

    Anything else stops the app: a typo must not quietly run production
    with development settings.
    """

    value = deployment_value("ENVIRONMENT", "development")

    if value not in _ENVIRONMENTS:
        raise ValueError(f"ENVIRONMENT must be 'development' or 'production', not {value!r}.")

    return value  # type: ignore[return-value]


__all__ = [
    "Environment",
    "app_environment",
    "deployment_value",
]
