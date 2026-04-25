"""DuckDB schema + repository tests using in-memory databases."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from image_generator.db.repository import RunsRepository, SelfiesRepository
from image_generator.models.results import GenerationResult, RunRecord
from image_generator.models.selfie import FaceEmbedding, Selfie

if TYPE_CHECKING:
    from image_generator.db.connection import Database


class TestSchema:
    def test_tables_exist(self, db: Database) -> None:
        tables = {row[0] for row in db.conn.execute("SHOW TABLES").fetchall()}
        assert {"runs", "selfies", "face_embeddings", "sweeps", "sweep_runs", "trainings"} <= tables


class TestRunsRepository:
    def test_insert_then_count(self, db: Database, sample_result: GenerationResult) -> None:
        repo = RunsRepository(db)
        assert repo.count() == 0
        repo.insert(RunRecord.from_result(sample_result))
        assert repo.count() == 1

    def test_recent_orders_by_completed_at(self, db: Database, sample_result: GenerationResult) -> None:
        repo = RunsRepository(db)
        repo.insert(RunRecord.from_result(sample_result))
        repo.insert(
            RunRecord.from_result(
                sample_result.model_copy(
                    update={"run_id": uuid4(), "completed_at": datetime.now(UTC)}
                )
            )
        )
        rows = repo.recent(limit=5)
        assert len(rows) == 2

    def test_update_metrics(self, db: Database, sample_result: GenerationResult) -> None:
        repo = RunsRepository(db)
        repo.insert(RunRecord.from_result(sample_result))
        repo.update_metrics(
            run_id=str(sample_result.run_id),
            prompt_siglip=0.31,
            aesthetic_laion=6.4,
        )
        rows = repo.recent(limit=1)
        assert rows[0]["prompt_siglip"] == 0.31
        assert rows[0]["aesthetic_laion"] == 6.4
        # Previously-set identity metric should survive the COALESCE.
        assert rows[0]["identity_arcface"] == sample_result.metrics.identity_arcface


class TestSelfiesRepository:
    def test_upsert_idempotent(self, db: Database) -> None:
        repo = SelfiesRepository(db)
        selfie = Selfie(
            sha256="b" * 64,
            path=Path("data/selfies/b.png"),
            uploaded_at=datetime.now(UTC),
            width=512,
            height=512,
        )
        repo.upsert(selfie)
        repo.upsert(selfie)  # should not raise
        got = repo.get("b" * 64)
        assert got is not None and got.sha256 == selfie.sha256

    def test_face_embedding_roundtrip(self, db: Database) -> None:
        repo = SelfiesRepository(db)
        emb = FaceEmbedding(
            selfie_sha256="c" * 64,
            model_name="arcface_r100",
            vector=[0.1, 0.2, 0.3, 0.4],
            computed_at=datetime.now(UTC),
        )
        repo.put_embedding(emb)
        got = repo.get_embedding("c" * 64, "arcface_r100")
        assert got is not None
        # DuckDB FLOAT[] stores float32; tolerate the precision delta.
        assert got.vector == pytest.approx(emb.vector)
