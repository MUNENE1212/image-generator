"""DuckDB persistence layer for runs, selfies, and trainings."""

from image_generator.db.connection import Database, get_database
from image_generator.db.repository import (
    RunsRepository,
    SelfiesRepository,
    SweepsRepository,
    TrainingsRepository,
)

__all__ = [
    "Database",
    "RunsRepository",
    "SelfiesRepository",
    "SweepsRepository",
    "TrainingsRepository",
    "get_database",
]
