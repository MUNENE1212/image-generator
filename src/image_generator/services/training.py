"""Per-subject LoRA training orchestration.

Pipeline:
  1. Bundle uploaded selfie bytes into an in-memory zip archive.
  2. Upload the zip to Replicate's file storage → get a public URL.
  3. Call backend.train_lora(archive_url=...) → returns a training job id.
  4. Insert a `trainings` row with status='running'.
  5. UI polls `poll_training()` periodically; that calls backend.training_status
     and updates the DB row when the job moves to a terminal state.

Why zip + upload (instead of passing N file URLs)? Replicate trainers expect
exactly one input_images URL — a single archive containing the dataset.
"""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import replicate

from image_generator.backends.base import BackendError
from image_generator.config import settings
from image_generator.db.repository import TrainingsRepository
from image_generator.logging import get_logger
from image_generator.models.enums import BackendName, TrainingMethod

if TYPE_CHECKING:
    from image_generator.backends.registry import BackendRegistry
    from image_generator.db.connection import Database

log = get_logger(__name__)


def build_selfie_archive(selfies: list[tuple[str, bytes]]) -> bytes:
    """Pack named selfie bytes into a zip. Filenames inside the archive matter
    for some trainers (they read them as captions), so callers should use
    descriptive names like 'alex_01.png'."""
    if not selfies:
        raise ValueError("Cannot build archive from an empty selfie list")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in selfies:
            zf.writestr(name, data)
    return buf.getvalue()


async def _upload_archive(api_token: str, archive_bytes: bytes, filename: str) -> str:
    """Upload an in-memory zip to Replicate's file API. Returns a public URL.

    `replicate.files.async_create` accepts a file-like object; we use BytesIO.
    Anything Replicate's trainings API can fetch will work as input_images.
    """
    client = replicate.Client(api_token=api_token)  # type: ignore[attr-defined]
    file_obj = io.BytesIO(archive_bytes)
    file_obj.name = filename  # BytesIO accepts dynamic .name attr at runtime
    try:
        uploaded = await client.files.async_create(file=file_obj)
    except Exception as e:
        raise BackendError(f"Replicate file upload failed: {e}") from e
    # The created file's public URL.
    return str(uploaded.urls["get"])


async def start_training(
    *,
    method: TrainingMethod,
    selfies: list[tuple[str, bytes]],
    lora_name: str,
    rank: int,
    steps: int,
    learning_rate: float,
    registry: BackendRegistry,
    db: Database,
) -> str:
    """Bundle selfies → upload → kick off training → log to DuckDB.

    Returns the training_id (also the Replicate job id) so the UI can poll.
    Currently Replicate-only — Fal training is not wired (BackendError raised).
    """
    if not selfies:
        raise ValueError("At least one selfie is required to train a LoRA")
    if settings.replicate_api_token is None:
        raise BackendError("REPLICATE_API_TOKEN required for training")
    if settings.replicate_destination is None:
        raise BackendError(
            "REPLICATE_DESTINATION required: set it to a Replicate model you own "
            "(e.g. 'alice/my-loras'). Create one at https://replicate.com/create."
        )

    backend = registry.get(BackendName.REPLICATE)
    archive_bytes = build_selfie_archive(selfies)
    archive_url = await _upload_archive(
        api_token=settings.replicate_api_token.get_secret_value(),
        archive_bytes=archive_bytes,
        filename=f"{lora_name}-selfies.zip",
    )

    log.info(
        "training.start",
        method=method.value,
        lora_name=lora_name,
        archive_size=len(archive_bytes),
        n_selfies=len(selfies),
    )

    job_id = await backend.train_lora(
        method=method,
        archive_url=archive_url,
        name=lora_name,
        destination=settings.replicate_destination.get_secret_value(),
        rank=rank,
        steps=steps,
        learning_rate=learning_rate,
    )

    selfie_shas = [name.split(".")[0] for name, _ in selfies]
    repo = TrainingsRepository(db)
    repo.insert(
        training_id=job_id,
        method=method.value,
        selfie_shas=selfie_shas,
        backend=BackendName.REPLICATE.value,
        status="running",
        started_at=datetime.now(UTC),
    )
    return job_id


async def poll_training(*, training_id: str, registry: BackendRegistry, db: Database) -> dict[str, object]:
    """Refresh the status of one training job and persist any state change.

    Returns the latest status dict for the caller to display. Idempotent —
    safe to call repeatedly even after the job is in a terminal state.
    """
    repo = TrainingsRepository(db)
    current = repo.get(training_id)
    if current is None:
        raise ValueError(f"Unknown training_id: {training_id}")
    if current["status"] in {"succeeded", "failed", "cancelled"}:
        return {"status": current["status"], "lora_url": current.get("lora_path")}

    backend_name = BackendName(str(current["backend"]))
    backend = registry.get(backend_name)
    status = await backend.training_status(training_id)

    if status["status"] in {"succeeded", "failed", "cancelled"}:
        repo.update_status(
            training_id,
            status=str(status["status"]),
            lora_path=str(status.get("lora_url")) if status.get("lora_url") else None,
            lora_name=str(current.get("lora_name") or ""),
            error=str(status.get("error")) if status.get("error") else None,
            completed_at=datetime.now(UTC),
        )
    return status
