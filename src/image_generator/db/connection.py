"""DuckDB connection manager.

DuckDB is single-writer per file but supports multiple read transactions. For a
Streamlit app that's one user at a time, the simplest correct pattern is a
process-global connection opened lazily. We expose it through `get_database()`
so tests can use an in-memory DB.
"""

from __future__ import annotations

import threading
from importlib.resources import files
from typing import TYPE_CHECKING, Final

import duckdb

from image_generator.config import settings
from image_generator.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)

_SCHEMA_PATH: Final = files("image_generator.db").joinpath("schema.sql")


class Database:
    """Thin wrapper around a DuckDB connection. Thread-safe via a single lock."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        self._path = str(path)
        self._lock = threading.RLock()
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(self._path)
        self._apply_schema()

    def _apply_schema(self) -> None:
        schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self._lock:
            self._conn.execute(schema_sql)
        log.info("db.schema_applied", path=self._path)

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Raw connection. Callers must hold `self.lock` for writes."""
        return self._conn

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_database_singleton: Database | None = None
_singleton_lock = threading.Lock()


def get_database() -> Database:
    """Return the process-global `Database`. Created on first call.

    Tests should construct `Database(":memory:")` directly instead of using this.
    """
    global _database_singleton
    if _database_singleton is None:
        with _singleton_lock:
            if _database_singleton is None:
                settings.ensure_dirs()
                _database_singleton = Database(settings.imggen_db_path)
    return _database_singleton
