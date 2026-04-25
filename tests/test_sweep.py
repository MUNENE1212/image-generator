"""Sweep expansion + persistence tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from image_generator.db.repository import RunsRepository, SweepsRepository
from image_generator.models.requests import SweepAxis, SweepConfig
from image_generator.models.results import RunRecord
from image_generator.services.sweep import SweepRunner

if TYPE_CHECKING:
    from image_generator.db.connection import Database
    from image_generator.models.requests import GenerationRequest
    from image_generator.models.results import GenerationResult


# ----------------------------- Expansion -----------------------------


class TestSweepExpand:
    def test_single_axis(self, sample_request: GenerationRequest) -> None:
        config = SweepConfig(
            base=sample_request,
            axes=[SweepAxis(field="guidance_scale", values=[3.0, 5.0, 7.0])],
        )
        cells = SweepRunner.expand(config)
        assert len(cells) == 3
        assert {c.request.guidance_scale for c in cells} == {3.0, 5.0, 7.0}

    def test_cartesian_product(self, sample_request: GenerationRequest) -> None:
        config = SweepConfig(
            base=sample_request,
            axes=[
                SweepAxis(field="guidance_scale", values=[3.0, 7.0]),
                SweepAxis(field="num_inference_steps", values=[20, 30, 40]),
            ],
        )
        cells = SweepRunner.expand(config)
        assert len(cells) == 6
        # Every combination is present exactly once
        combos = {(c.request.guidance_scale, c.request.num_inference_steps) for c in cells}
        assert combos == {(3.0, 20), (3.0, 30), (3.0, 40), (7.0, 20), (7.0, 30), (7.0, 40)}

    def test_int_fields_coerced(self, sample_request: GenerationRequest) -> None:
        # Even if the user supplies floats, steps must be int for GenerationRequest.
        config = SweepConfig(
            base=sample_request,
            axes=[SweepAxis(field="num_inference_steps", values=[20.0, 30.0])],
        )
        cells = SweepRunner.expand(config)
        for c in cells:
            assert isinstance(c.request.num_inference_steps, int)

    def test_seed_axis_passes_through(self, sample_request: GenerationRequest) -> None:
        config = SweepConfig(
            base=sample_request,
            axes=[SweepAxis(field="seed", values=[0, 1, 42])],
        )
        cells = SweepRunner.expand(config)
        assert {c.request.seed for c in cells} == {0, 1, 42}

    def test_seed_derived_when_not_swept(self, sample_request: GenerationRequest) -> None:
        # Each cell should still get a deterministic, distinct seed even when seed
        # isn't in the axes — so cells aren't accidentally sharing initial noise.
        config = SweepConfig(
            base=sample_request,
            axes=[SweepAxis(field="guidance_scale", values=[3.0, 5.0, 7.0])],
        )
        cells = SweepRunner.expand(config)
        seeds = [c.request.seed for c in cells]
        assert len(set(seeds)) == 3  # all distinct

    def test_overrides_track_axes_only(self, sample_request: GenerationRequest) -> None:
        # `overrides` records what the user swept, not the derived seed.
        config = SweepConfig(
            base=sample_request,
            axes=[SweepAxis(field="guidance_scale", values=[3.0])],
        )
        cells = SweepRunner.expand(config)
        assert cells[0].overrides == (("guidance_scale", 3.0),)


# ----------------------------- Persistence roundtrip -----------------------------


class TestSweepsRepository:
    def test_insert_and_list(self, db: Database) -> None:
        repo = SweepsRepository(db)
        repo.insert(
            sweep_id="s1",
            name="my-sweep",
            config_json=json.dumps({"axes": []}),
            created_at=datetime.now(UTC),
        )
        results = repo.recent()
        assert len(results) == 1
        assert results[0]["name"] == "my-sweep"
        assert results[0]["run_count"] == 0

    def test_link_run_and_get_runs(self, db: Database, sample_result: GenerationResult) -> None:
        runs_repo = RunsRepository(db)
        sweeps_repo = SweepsRepository(db)

        runs_repo.insert(RunRecord.from_result(sample_result))
        sweeps_repo.insert(
            sweep_id="s1",
            name="t",
            config_json="{}",
            created_at=datetime.now(UTC),
        )
        sweeps_repo.link_run("s1", str(sample_result.run_id))

        runs = sweeps_repo.get_runs("s1")
        assert len(runs) == 1
        assert runs[0]["run_id"] == str(sample_result.run_id)

    def test_link_run_idempotent(self, db: Database, sample_result: GenerationResult) -> None:
        runs_repo = RunsRepository(db)
        sweeps_repo = SweepsRepository(db)
        runs_repo.insert(RunRecord.from_result(sample_result))
        sweeps_repo.insert(
            sweep_id="s1", name="t", config_json="{}", created_at=datetime.now(UTC)
        )
        sweeps_repo.link_run("s1", str(sample_result.run_id))
        sweeps_repo.link_run("s1", str(sample_result.run_id))  # must not raise
        assert len(sweeps_repo.get_runs("s1")) == 1
