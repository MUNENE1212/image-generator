"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from image_generator.db.connection import Database
from image_generator.models.enums import Backbone, BackendName, Strategy
from image_generator.models.requests import GenerationRequest
from image_generator.models.results import EvalMetrics, GenerationResult

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def db() -> Iterator[Database]:
    """In-memory DuckDB with schema applied."""
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def sample_request() -> GenerationRequest:
    return GenerationRequest(
        strategy=Strategy.INSTANT_ID,
        backbone=Backbone.SDXL,
        prompt="portrait of a person",
        selfie_sha256="a" * 64,
        seed=42,
    )


@pytest.fixture
def sample_result(sample_request: GenerationRequest) -> GenerationResult:
    now = datetime.now(UTC)
    return GenerationResult(
        run_id=uuid4(),
        request=sample_request,
        image_path=Path("data/outputs/test.png"),
        backend=BackendName.REPLICATE,
        model_version="stability-ai/sdxl:test",
        started_at=now,
        completed_at=now,
        cost_usd=0.005,
        metrics=EvalMetrics(identity_arcface=0.72),
    )
