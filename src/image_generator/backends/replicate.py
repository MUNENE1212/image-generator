"""Replicate adapter — wires `replicate.async_run` into the ComputeBackend protocol.

Pin model versions in `REPLICATE_MODELS` for reproducibility. A bare slug like
`"stability-ai/sdxl"` resolves to "latest" at call time, which is convenient for
development but breaks repro. Add `:<hash>` once you've validated a version.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import replicate

from image_generator.backends.base import BackendError, ComputeBackend, Quote
from image_generator.logging import get_logger
from image_generator.models.enums import Backbone, BackendName, Strategy, TrainingMethod
from image_generator.models.results import GenerationResult
from image_generator.storage.local import LocalStorage

if TYPE_CHECKING:
    from image_generator.models.requests import GenerationRequest
    from image_generator.storage.base import Storage

log = get_logger(__name__)

# Pinned model versions for reproducibility.
# Hashes captured 2026-04-25 by hitting the live Replicate API. To refresh:
#   curl -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
#     https://api.replicate.com/v1/models/<owner>/<name>
# and copy `latest_version.id`.
#
# Note: replicate.async_run REQUIRES the `:hash` suffix — bare slugs 404.
REPLICATE_MODELS: dict[tuple[Strategy, Backbone], str] = {
    (Strategy.PROMPT_ONLY, Backbone.SDXL): "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
    (Strategy.PROMPT_ONLY, Backbone.FLUX_DEV): "black-forest-labs/flux-dev:6e4a938f85952bda9b5797cd5f95eaee63d4bb2f0e3bd13a2e9ff5a1edcce80c",
    (Strategy.INSTANT_ID, Backbone.SDXL): "zsxkib/instant-id:2e4785a4d80dadf580077b2244c8d7c05d8e3faac04a04c02d8e099dd2876789",
    (Strategy.IP_ADAPTER_FACEID, Backbone.SDXL): "lucataco/ip-adapter-faceid:fb81ef963e74776af72e6f380949013533d46dd5c6228a9e586c57db6303d7cd",
    (Strategy.PHOTOMAKER, Backbone.SDXL): "tencentarc/photomaker:ddfc2b08d209f9fa8c1eca692712918bd449f695dabb4a958da31802a9570fe4",
    (Strategy.PULID, Backbone.FLUX_DEV): "zsxkib/flux-pulid:8baa7ef2255075b46f4d91cd238c21d31181b3e6a864463f967960bb0112525b",
    # SDXL LoRA inference: lucataco/sdxl accepts a LoRA URL via `lora_url` input.
    (Strategy.LORA, Backbone.SDXL): "lucataco/sdxl:c86579ac5193bf45422f1c8b92742135aa859b1850a8e4c531bff222fc75273d",
    (Strategy.LORA, Backbone.FLUX_DEV): "lucataco/flux-dev-lora:091495765fa5ef27a03cfd83eee0b73b87b8770f99ac1b32f1b8b56e2ee2a0c7",
}

REPLICATE_PRICES: dict[Backbone, float] = {
    Backbone.SDXL: 0.005,
    Backbone.FLUX_DEV: 0.030,
}

# LoRA trainer models. Replicate trainings target a (model, version) pair.
# Update the version pin once a known-good hash is validated; until then a bare
# slug resolves to the trainer's default version.
REPLICATE_TRAINERS: dict[TrainingMethod, tuple[str, str | None]] = {
    # SDXL LoRA trainer — Stability AI's official.
    TrainingMethod.LORA_SDXL: ("stability-ai/sdxl", None),
    # FLUX-dev LoRA trainer (community: ostris/flux-dev-lora-trainer).
    TrainingMethod.LORA_FLUX: ("ostris/flux-dev-lora-trainer", None),
}

# Rough $/training. Refine with measured data once jobs run.
REPLICATE_TRAINING_PRICES: dict[TrainingMethod, float] = {
    TrainingMethod.LORA_SDXL: 2.0,
    TrainingMethod.LORA_FLUX: 5.0,
}


def _build_input(request: GenerationRequest, selfie_bytes: bytes | None) -> dict[str, Any]:
    """Map a GenerationRequest to the input dict for Replicate.

    Per-strategy input keys differ slightly. Common fields are always set; identity
    strategies add the selfie under the key the model expects.
    """
    common: dict[str, Any] = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt or "",
        "width": request.width,
        "height": request.height,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "seed": request.seed,
    }

    if request.strategy is Strategy.PROMPT_ONLY:
        return common

    if selfie_bytes is None:
        raise BackendError(f"strategy={request.strategy.value} requires a selfie")

    selfie_file = io.BytesIO(selfie_bytes)

    # Per-strategy selfie key + identity-strength knob.
    # See model READMEs on Replicate for the exact field names.
    if request.strategy is Strategy.INSTANT_ID:
        return {**common, "image": selfie_file, "ip_adapter_scale": request.identity_strength}
    if request.strategy is Strategy.IP_ADAPTER_FACEID:
        return {**common, "image": selfie_file, "ip_adapter_scale": request.identity_strength}
    if request.strategy is Strategy.PHOTOMAKER:
        return {**common, "input_image": selfie_file, "style_strength_ratio": int(request.identity_strength * 50)}
    if request.strategy is Strategy.PULID:
        return {**common, "main_face_image": selfie_file, "id_weight": request.identity_strength}
    if request.strategy is Strategy.LORA:
        if not request.lora_name:
            raise BackendError("strategy=lora requires lora_name")
        return {**common, "lora_url": request.lora_name, "lora_scale": request.identity_strength}

    raise BackendError(f"Unhandled strategy {request.strategy}")


async def _read_output_bytes(output: Any) -> bytes:
    """Coerce a replicate output into bytes.

    `async_run(use_file_output=True)` returns FileOutput | list[FileOutput]; older
    or non-image models may return a URL string or list of strings.
    """
    if isinstance(output, list):
        if not output:
            raise BackendError("Replicate returned empty output list")
        output = output[0]
    if hasattr(output, "aread"):
        return await output.aread()  # type: ignore[no-any-return]
    if hasattr(output, "read"):
        return output.read()  # type: ignore[no-any-return]
    if isinstance(output, str):
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(output)
            resp.raise_for_status()
            return resp.content
    raise BackendError(f"Unrecognized replicate output type: {type(output).__name__}")


class ReplicateBackend(ComputeBackend):
    name = BackendName.REPLICATE

    def __init__(self, api_token: str, storage: Storage | None = None) -> None:
        if not api_token:
            raise BackendError("REPLICATE_API_TOKEN is required")
        self._api_token = api_token
        self._storage: Storage = storage or LocalStorage()

    def _new_client(self) -> Any:  # replicate.Client; not exported in stubs
        # Construct per-call. The internal httpx.AsyncClient binds its connection
        # pool to whatever event loop is current at construction. Streamlit calls
        # our async methods via asyncio.run() which builds a fresh loop each time;
        # a long-lived client would die with "Event loop is closed" on the second
        # request. Per-call construction is cheap and side-steps the issue.
        return replicate.Client(api_token=self._api_token)  # type: ignore[attr-defined]

    def supports(self, strategy: Strategy, backbone: Backbone) -> bool:
        return (strategy, backbone) in REPLICATE_MODELS

    def quote(self, request: GenerationRequest) -> Quote:
        if not self.supports(request.strategy, request.backbone):
            raise BackendError(f"Unsupported: {request.strategy}/{request.backbone}")
        return Quote(
            backend=self.name,
            estimated_cost_usd=REPLICATE_PRICES[request.backbone],
            estimated_seconds=20.0 if request.backbone is Backbone.SDXL else 45.0,
            model_version=REPLICATE_MODELS[(request.strategy, request.backbone)],
        )

    async def generate(self, request: GenerationRequest, selfie_bytes: bytes | None) -> GenerationResult:
        if not self.supports(request.strategy, request.backbone):
            raise BackendError(f"Unsupported: {request.strategy}/{request.backbone}")

        model_ref = REPLICATE_MODELS[(request.strategy, request.backbone)]
        payload = _build_input(request, selfie_bytes)
        run_id = uuid4()
        started = datetime.now(UTC)

        log.info(
            "replicate.generate.start",
            strategy=request.strategy.value,
            backbone=request.backbone.value,
            model=model_ref,
            run_id=str(run_id),
        )

        client = self._new_client()
        try:
            output = await client.async_run(model_ref, input=payload, use_file_output=True)
        except Exception as e:
            raise BackendError(f"Replicate call failed for {model_ref}: {e}") from e

        image_bytes = await _read_output_bytes(output)
        image_path = self._storage.put_output(image_bytes, str(run_id))
        completed = datetime.now(UTC)

        log.info(
            "replicate.generate.done",
            run_id=str(run_id),
            duration_s=(completed - started).total_seconds(),
        )

        return GenerationResult(
            run_id=run_id,
            request=request,
            image_path=image_path,
            backend=self.name,
            model_version=model_ref,
            started_at=started,
            completed_at=completed,
            cost_usd=REPLICATE_PRICES[request.backbone],
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
        """Kick off a LoRA training job on Replicate.

        `archive_url` must be a publicly readable URL (e.g. one returned by
        `replicate.files.async_create`) pointing to a zip of selfie images.

        `destination` is the user's pre-created Replicate model that the trained
        LoRA will be pushed to, e.g. "alice/my-loras".
        """
        if method not in REPLICATE_TRAINERS:
            raise BackendError(f"Unsupported training method: {method.value}")

        trainer_model, trainer_version = REPLICATE_TRAINERS[method]
        # Per-trainer input field names. Both major trainers use input_images +
        # standard hyperparam names, but FLUX trainers expect 'trigger_word'
        # while SDXL ones use 'token_string'. Build accordingly.
        trainer_input: dict[str, Any] = {
            "input_images": archive_url,
            "lora_rank": rank,
            "max_train_steps": steps,
            "learning_rate": learning_rate,
        }
        if method is TrainingMethod.LORA_FLUX:
            trainer_input["trigger_word"] = name
        else:
            trainer_input["token_string"] = f"TOK_{name}"

        log.info(
            "replicate.training.start",
            method=method.value,
            destination=destination,
            trainer=trainer_model,
        )

        client = self._new_client()
        try:
            training = await client.trainings.async_create(
                model=trainer_model,
                version=trainer_version or "latest",
                input=trainer_input,
                destination=destination,
            )
        except Exception as e:
            raise BackendError(f"Replicate training create failed: {e}") from e

        return str(training.id)

    async def training_status(self, job_id: str) -> dict[str, object]:
        """Poll a training job. Returns a flat dict — same shape regardless of vendor.

        Keys: status (queued|running|succeeded|failed|cancelled), lora_url (when
        succeeded), error (when failed), progress (0.0–1.0 if available).
        """
        client = self._new_client()
        try:
            training = await client.trainings.async_get(job_id)
        except Exception as e:
            raise BackendError(f"Replicate training get failed: {e}") from e

        # Normalize Replicate's status terms to our flat vocabulary.
        replicate_status = training.status  # 'starting'|'processing'|'succeeded'|'failed'|'canceled'
        status_map = {
            "starting": "running",
            "processing": "running",
            "succeeded": "succeeded",
            "failed": "failed",
            "canceled": "cancelled",
        }
        normalized = status_map.get(replicate_status, replicate_status)

        result: dict[str, object] = {"status": normalized}
        # On success, output is typically the trained model version, e.g.
        # {"version": "alice/my-loras:abc123"} or just the URL string.
        if normalized == "succeeded" and training.output is not None:
            output = training.output
            if isinstance(output, dict):
                result["lora_url"] = output.get("version") or output.get("weights") or str(output)
            else:
                result["lora_url"] = str(output)
        if normalized == "failed":
            result["error"] = training.error or "training failed (no error message)"
        return result

    async def health(self) -> bool:
        try:
            await self._new_client().models.async_list()
        except Exception:
            return False
        return True
