"""Structured logging setup. Call `configure_logging()` once at app entry."""

from __future__ import annotations

import logging
import sys

import structlog

from image_generator.config import settings


def configure_logging() -> None:
    """Configure structlog + stdlib logging based on `settings`.

    Idempotent — safe to call from both CLI entry points and Streamlit page reloads.
    """
    level = getattr(logging, settings.imggen_log_level)

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.imggen_log_format == "json":
        renderer: structlog.typing.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=level,
        stream=sys.stderr,
        force=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    # structlog.get_logger() is dynamically typed; the runtime type matches.
    return structlog.get_logger(name)  # type: ignore[no-any-return]
