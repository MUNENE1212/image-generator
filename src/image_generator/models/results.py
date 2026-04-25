"""Result models: what the backend returns + what we store."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from image_generator.models.enums import BackendName
from image_generator.models.requests import GenerationRequest


class EvalMetrics(BaseModel):
    """Automated metrics attached to a generation result.

    All fields are optional because metrics are computed asynchronously after
    the image is written — a fresh row may have identity_arcface=None until the
    harness fills it in.
    """

    model_config = ConfigDict(frozen=True)

    identity_arcface: float | None = Field(
        default=None, description="Cosine similarity to selfie (ArcFace). Higher = more similar."
    )
    identity_adaface: float | None = Field(
        default=None, description="Cosine similarity to selfie (AdaFace). Cross-check for ArcFace."
    )
    prompt_siglip: float | None = Field(
        default=None, description="Image-text similarity (SigLIP). Higher = better adherence."
    )
    aesthetic_laion: float | None = Field(
        default=None, description="LAION aesthetic score, 1–10."
    )
    aesthetic_qalign: float | None = Field(
        default=None, description="Q-Align aesthetic score (secondary check)."
    )
    diversity_lpips: float | None = Field(
        default=None,
        description="Mean LPIPS distance vs. sibling seeds. Higher = more diverse (no mode collapse).",
    )


class GenerationResult(BaseModel):
    """Successful generation output. Persisted as one row in the `runs` table."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID = Field(default_factory=uuid4)
    request: GenerationRequest
    image_path: Path

    # Execution metadata
    backend: BackendName
    model_version: str = Field(..., description="Pinned model id on the backend, e.g. replicate:owner/model:hash")
    started_at: datetime
    completed_at: datetime
    cost_usd: float = Field(..., ge=0.0)

    # Metrics filled in by the eval harness after generation
    metrics: EvalMetrics = Field(default_factory=EvalMetrics)

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()


class RunRecord(BaseModel):
    """Flat row shape used for DuckDB insertion and Experiments-page queries.

    Mirrors `GenerationResult` but with every GenerationRequest field promoted
    to a column so analytical queries (`GROUP BY strategy`, `WHERE cfg > 5`)
    stay trivial.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    # Request fields
    strategy: str
    backbone: str
    prompt: str
    negative_prompt: str | None
    selfie_sha256: str | None
    seed: int
    num_inference_steps: int
    guidance_scale: float
    width: int
    height: int
    identity_strength: float
    lora_name: str | None
    # Execution
    backend: str
    model_version: str
    image_path: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    cost_usd: float
    # Metrics
    identity_arcface: float | None
    identity_adaface: float | None
    prompt_siglip: float | None
    aesthetic_laion: float | None
    aesthetic_qalign: float | None
    diversity_lpips: float | None

    @classmethod
    def from_result(cls, result: GenerationResult) -> RunRecord:
        req = result.request
        m = result.metrics
        return cls(
            run_id=str(result.run_id),
            strategy=req.strategy.value,
            backbone=req.backbone.value,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            selfie_sha256=req.selfie_sha256,
            seed=req.seed,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            identity_strength=req.identity_strength,
            lora_name=req.lora_name,
            backend=result.backend.value,
            model_version=result.model_version,
            image_path=str(result.image_path),
            started_at=result.started_at,
            completed_at=result.completed_at,
            duration_seconds=result.duration_seconds,
            cost_usd=result.cost_usd,
            identity_arcface=m.identity_arcface,
            identity_adaface=m.identity_adaface,
            prompt_siglip=m.prompt_siglip,
            aesthetic_laion=m.aesthetic_laion,
            aesthetic_qalign=m.aesthetic_qalign,
            diversity_lpips=m.diversity_lpips,
        )
