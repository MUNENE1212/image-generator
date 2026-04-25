"""Request models: what the UI sends to the backend."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from image_generator.models.enums import Backbone, BackendName, Strategy


class GenerationRequest(BaseModel):
    """A single generation request.

    This is the contract between the Streamlit UI and the `ComputeBackend`.
    It is serialized into the `runs` table so any run can be replayed by id.
    """

    model_config = ConfigDict(frozen=True)

    # Required
    strategy: Strategy
    backbone: Backbone
    prompt: str = Field(..., min_length=1, max_length=2000)
    selfie_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        description="None only for Strategy.PROMPT_ONLY.",
    )

    # Routing
    backend: BackendName | None = Field(
        default=None,
        description="If None, the backend registry picks one based on capability + config.",
    )

    # Sampling
    negative_prompt: str | None = Field(default=None, max_length=2000)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    num_inference_steps: int = Field(default=30, ge=1, le=150)
    guidance_scale: float = Field(default=5.0, ge=0.0, le=20.0)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)

    # Strategy-specific knobs (flat so the DB schema stays simple)
    identity_strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.5,
        description="Identity preservation weight. Semantics vary by strategy.",
    )
    lora_name: str | None = Field(
        default=None,
        description="Required iff strategy == LORA. Names a trained LoRA artifact.",
    )

    @field_validator("selfie_sha256")
    @classmethod
    def _require_selfie_for_id_strategies(cls, v: str | None, info: object) -> str | None:
        # Cross-field validation needs model_validator; kept here as a hint. The
        # full cross-field check happens in `validate_request_consistency`.
        return v

    def validate_request_consistency(self) -> None:
        """Raise ValueError if the combination is invalid.

        Called explicitly after construction because field_validators can't see
        sibling fields in Pydantic v2 without model_validator gymnastics.
        """
        if self.strategy is not Strategy.PROMPT_ONLY and self.selfie_sha256 is None:
            raise ValueError(f"strategy={self.strategy} requires a selfie")
        if self.strategy is Strategy.LORA and not self.lora_name:
            raise ValueError("strategy=lora requires lora_name")
        if self.width % 8 != 0 or self.height % 8 != 0:
            raise ValueError("width and height must be multiples of 8")


class SweepAxis(BaseModel):
    """One axis of an Experiments-page sweep."""

    model_config = ConfigDict(frozen=True)

    field: str  # e.g. "guidance_scale", "num_inference_steps", "seed"
    values: list[float | int | str]

    @field_validator("values")
    @classmethod
    def _nonempty(cls, v: list[float | int | str]) -> list[float | int | str]:
        if not v:
            raise ValueError("sweep axis must have at least one value")
        return v


class SweepConfig(BaseModel):
    """A full Experiments sweep: base request + axes to vary."""

    base: GenerationRequest
    axes: list[SweepAxis] = Field(..., min_length=1)

    @property
    def total_cells(self) -> int:
        total = 1
        for axis in self.axes:
            total *= len(axis.values)
        return total
