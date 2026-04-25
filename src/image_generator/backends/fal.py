"""Fal.ai adapter — wires `fal_client.subscribe_async` into the ComputeBackend protocol.

Fal differs from Replicate in two ways that matter here:
  1. Selfies must be uploaded to Fal storage first; the model takes a URL, not bytes.
  2. Each application has its own input schema. We map per-strategy below.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import fal_client
import httpx

from image_generator.backends.base import BackendError, ComputeBackend, Quote
from image_generator.logging import get_logger
from image_generator.models.enums import Backbone, BackendName, Strategy, TrainingMethod
from image_generator.models.results import GenerationResult
from image_generator.storage.local import LocalStorage

if TYPE_CHECKING:
    from image_generator.models.requests import GenerationRequest
    from image_generator.storage.base import Storage

log = get_logger(__name__)

FAL_MODELS: dict[tuple[Strategy, Backbone], str] = {
    (Strategy.PROMPT_ONLY, Backbone.SDXL): "fal-ai/fast-sdxl",
    (Strategy.PROMPT_ONLY, Backbone.FLUX_DEV): "fal-ai/flux/dev",
    (Strategy.INSTANT_ID, Backbone.SDXL): "fal-ai/instant-id",
    (Strategy.IP_ADAPTER_FACEID, Backbone.SDXL): "fal-ai/ip-adapter-face-id",
    (Strategy.PHOTOMAKER, Backbone.SDXL): "fal-ai/photomaker",
    (Strategy.PULID, Backbone.FLUX_DEV): "fal-ai/flux-pulid",
    (Strategy.LORA, Backbone.SDXL): "fal-ai/lora",
    (Strategy.LORA, Backbone.FLUX_DEV): "fal-ai/flux-lora",
}

FAL_PRICES: dict[Backbone, float] = {
    Backbone.SDXL: 0.004,
    Backbone.FLUX_DEV: 0.025,
}


def _build_arguments(request: GenerationRequest, selfie_url: str | None) -> dict[str, Any]:
    """Map a GenerationRequest to fal arguments for the chosen application."""
    common: dict[str, Any] = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt or "",
        "image_size": {"width": request.width, "height": request.height},
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "seed": request.seed,
        "num_images": 1,
    }

    if request.strategy is Strategy.PROMPT_ONLY:
        return common
    if selfie_url is None:
        raise BackendError(f"strategy={request.strategy.value} requires a selfie")

    if request.strategy is Strategy.INSTANT_ID:
        return {**common, "face_image_url": selfie_url, "ip_adapter_scale": request.identity_strength}
    if request.strategy is Strategy.IP_ADAPTER_FACEID:
        return {**common, "face_image_url": selfie_url, "scale": request.identity_strength}
    if request.strategy is Strategy.PHOTOMAKER:
        return {**common, "image_archive_url": selfie_url}
    if request.strategy is Strategy.PULID:
        return {**common, "reference_image_url": selfie_url, "id_weight": request.identity_strength}
    if request.strategy is Strategy.LORA:
        if not request.lora_name:
            raise BackendError("strategy=lora requires lora_name")
        return {
            **common,
            "loras": [{"path": request.lora_name, "scale": request.identity_strength}],
        }

    raise BackendError(f"Unhandled strategy {request.strategy}")


def _extract_image_url(result: dict[str, Any]) -> str:
    """Fal applications return image URLs under varying keys.

    Most use `images: [{url, ...}]`; some use `image: {url}`. Cover both.
    """
    if isinstance(result.get("images"), list) and result["images"]:
        url = result["images"][0].get("url")
        if isinstance(url, str):
            return url
    if isinstance(result.get("image"), dict):
        url = result["image"].get("url")
        if isinstance(url, str):
            return url
    raise BackendError(f"Could not find image URL in fal result: keys={list(result.keys())}")


class FalBackend(ComputeBackend):
    name = BackendName.FAL

    def __init__(self, api_key: str, storage: Storage | None = None) -> None:
        if not api_key:
            raise BackendError("FAL_KEY is required")
        self._api_key = api_key
        # Also set in env so any code path that falls back to module-level
        # fal_client functions still authenticates.
        os.environ["FAL_KEY"] = api_key
        self._storage: Storage = storage or LocalStorage()

    def _new_client(self) -> fal_client.AsyncClient:
        # Construct per-call. Same reason as ReplicateBackend._new_client:
        # the cached httpx.AsyncClient inside binds to the current event loop,
        # which Streamlit destroys between asyncio.run() calls.
        return fal_client.AsyncClient(key=self._api_key)

    def supports(self, strategy: Strategy, backbone: Backbone) -> bool:
        return (strategy, backbone) in FAL_MODELS

    def quote(self, request: GenerationRequest) -> Quote:
        if not self.supports(request.strategy, request.backbone):
            raise BackendError(f"Unsupported: {request.strategy}/{request.backbone}")
        return Quote(
            backend=self.name,
            estimated_cost_usd=FAL_PRICES[request.backbone],
            estimated_seconds=10.0 if request.backbone is Backbone.SDXL else 25.0,
            model_version=FAL_MODELS[(request.strategy, request.backbone)],
        )

    async def generate(self, request: GenerationRequest, selfie_bytes: bytes | None) -> GenerationResult:
        if not self.supports(request.strategy, request.backbone):
            raise BackendError(f"Unsupported: {request.strategy}/{request.backbone}")

        application = FAL_MODELS[(request.strategy, request.backbone)]
        run_id = uuid4()
        started = datetime.now(UTC)
        client = self._new_client()

        selfie_url: str | None = None
        if selfie_bytes is not None:
            try:
                selfie_url = await client.upload(selfie_bytes, "image/png")
            except Exception as e:
                raise BackendError(f"Fal selfie upload failed: {e}") from e

        arguments = _build_arguments(request, selfie_url)
        log.info(
            "fal.generate.start",
            strategy=request.strategy.value,
            backbone=request.backbone.value,
            application=application,
            run_id=str(run_id),
        )

        try:
            result = await client.subscribe(application, arguments=arguments)
        except Exception as e:
            raise BackendError(f"Fal call failed for {application}: {e}") from e

        image_url = _extract_image_url(result)
        async with httpx.AsyncClient(timeout=60) as http:
            resp = await http.get(image_url)
            resp.raise_for_status()
            image_bytes = resp.content

        image_path = self._storage.put_output(image_bytes, str(run_id))
        completed = datetime.now(UTC)

        log.info(
            "fal.generate.done",
            run_id=str(run_id),
            duration_s=(completed - started).total_seconds(),
        )

        return GenerationResult(
            run_id=run_id,
            request=request,
            image_path=image_path,
            backend=self.name,
            model_version=application,
            started_at=started,
            completed_at=completed,
            cost_usd=FAL_PRICES[request.backbone],
        )

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
        # Fal.ai LoRA training (e.g. fal-ai/flux-lora-trainer) is conceptually similar
        # but takes a different input schema. Wired in a later phase — for now,
        # users training LoRAs go through Replicate.
        raise NotImplementedError("Fal LoRA training not wired — use REPLICATE_API_TOKEN for training.")

    async def training_status(self, job_id: str) -> dict[str, object]:
        raise NotImplementedError("Fal LoRA training not wired — use REPLICATE_API_TOKEN for training.")

    async def health(self) -> bool:
        # fal_client has no cheap ping endpoint; "health" here means credentials
        # are present (constructor enforced) and the module imports.
        return True
