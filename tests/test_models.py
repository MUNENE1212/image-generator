"""Pydantic model validation + round-trip tests."""

from __future__ import annotations

import pytest

from image_generator.models.enums import Backbone, Strategy
from image_generator.models.requests import GenerationRequest, SweepAxis, SweepConfig
from image_generator.models.results import RunRecord


class TestGenerationRequest:
    def test_default_fields(self) -> None:
        req = GenerationRequest(
            strategy=Strategy.PROMPT_ONLY,
            backbone=Backbone.SDXL,
            prompt="a cat",
        )
        assert req.num_inference_steps == 30
        assert req.guidance_scale == 5.0

    def test_prompt_only_needs_no_selfie(self) -> None:
        req = GenerationRequest(
            strategy=Strategy.PROMPT_ONLY,
            backbone=Backbone.SDXL,
            prompt="a cat",
        )
        req.validate_request_consistency()  # no raise

    def test_instant_id_requires_selfie(self) -> None:
        req = GenerationRequest(
            strategy=Strategy.INSTANT_ID,
            backbone=Backbone.SDXL,
            prompt="portrait",
        )
        with pytest.raises(ValueError, match="requires a selfie"):
            req.validate_request_consistency()

    def test_lora_requires_name(self, sample_request: GenerationRequest) -> None:
        bad = sample_request.model_copy(update={"strategy": Strategy.LORA, "lora_name": None})
        with pytest.raises(ValueError, match="requires lora_name"):
            bad.validate_request_consistency()

    def test_dimensions_must_be_divisible_by_eight(self, sample_request: GenerationRequest) -> None:
        bad = sample_request.model_copy(update={"width": 1023})
        with pytest.raises(ValueError, match="multiples of 8"):
            bad.validate_request_consistency()

    def test_is_frozen(self, sample_request: GenerationRequest) -> None:
        with pytest.raises(Exception):  # noqa: B017 — pydantic frozen model
            sample_request.seed = 99

    def test_json_roundtrip(self, sample_request: GenerationRequest) -> None:
        dumped = sample_request.model_dump_json()
        restored = GenerationRequest.model_validate_json(dumped)
        assert restored == sample_request


class TestSweepConfig:
    def test_total_cells(self, sample_request: GenerationRequest) -> None:
        config = SweepConfig(
            base=sample_request,
            axes=[
                SweepAxis(field="guidance_scale", values=[3.0, 5.0, 7.0]),
                SweepAxis(field="seed", values=[0, 1]),
            ],
        )
        assert config.total_cells == 6

    def test_empty_axis_rejected(self, sample_request: GenerationRequest) -> None:
        with pytest.raises(ValueError):
            SweepConfig(base=sample_request, axes=[SweepAxis(field="seed", values=[])])


class TestRunRecord:
    def test_from_result_flattens(self, sample_result: object) -> None:
        # sample_result is GenerationResult — typed as object here to match conftest signature
        from image_generator.models.results import GenerationResult

        assert isinstance(sample_result, GenerationResult)
        record = RunRecord.from_result(sample_result)
        assert record.strategy == sample_result.request.strategy.value
        assert record.backbone == sample_result.request.backbone.value
        assert record.identity_arcface == sample_result.metrics.identity_arcface
        assert record.duration_seconds == sample_result.duration_seconds
