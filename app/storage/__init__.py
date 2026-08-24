"""Persistent storage adapters for completed research runs."""

from app.storage.run_repository import RunRepository
from app.storage.sqlite_run_repository import (
    SQLiteRunRepository,
    get_history_repository,
    reset_history_repository,
)

__all__ = [
    "RunRepository",
    "SQLiteRunRepository",
    "get_history_repository",
    "reset_history_repository",
]
