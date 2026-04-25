"""`ComputeBackend` protocol — the boundary between the app and external GPU APIs.

All methods are `async`. Backends are expected to use their vendor's async SDK
(replicate.async_run, fal_client.submit_async) rather than wrapping sync calls
in run_in_executor — that's why this protocol is async-native.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from image_generator.models.enums import Backbone, BackendName, Strategy, TrainingMethod
    from image_generator.models.requests import GenerationRequest
    from image_generator.models.results import GenerationResult


class BackendError(Exception):
    """Raised for any backend-level failure. Wraps vendor exceptions."""


@dataclass(frozen=True, slots=True)
class Quote:
    """Price + latency estimate for a request on a given backend."""

    backend: BackendName
    estimated_cost_usd: float
    estimated_seconds: float
    model_version: str


@runtime_checkable
class ComputeBackend(Protocol):
    """A remote compute provider capable of generation and optionally training."""

    name: BackendName

    def supports(self, strategy: Strategy, backbone: Backbone) -> bool:
        """Return True if this backend can serve (strategy, backbone)."""
        ...

    def quote(self, request: GenerationRequest) -> Quote:
        """Synchronous price/latency estimate. No network call."""
        ...

    async def generate(self, request: GenerationRequest, selfie_bytes: bytes | None) -> GenerationResult:
        """Execute one generation. Raises BackendError on failure."""
        ...

    async def train_lora(
        self,
        *,
        method: TrainingMethod,
        archive_url: str,
        name: str,
        destination: str,
        rank: int = 16,
        steps: int = 1500,
        learning_rate: float = 1e-4,
    ) -> str:
        """Kick off a LoRA training job.

        `archive_url` points to a publicly readable zip of selfies (uploaded
        beforehand via the backend's file API). `destination` is the user's
        pre-created Replicate/Fal model where the LoRA artifact will be pushed.
        Returns a backend-specific job id for `training_status` to poll.
        """
        ...

    async def training_status(self, job_id: str) -> dict[str, object]:
        """Poll training progress. Shape: {status, progress, lora_url, error}."""
        ...

    async def health(self) -> bool:
        """Cheap liveness check (token validity, typically)."""
        ...
