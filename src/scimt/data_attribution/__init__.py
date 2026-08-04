"""Data-attribution methods with optional, lazily imported dependencies."""

from typing import Any as _Any

__all__ = ["SOURCE_REPOSITORY", "SOURCE_COMMIT", "MIGRATED_MODULES"]


def __getattr__(name: str) -> _Any:
    if name in __all__:
        from . import _migration

        return getattr(_migration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
