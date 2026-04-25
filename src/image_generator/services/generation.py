"""Generation orchestration service.

`generate_and_log()` is the single entry point used by every UI surface that
produces one image (Home, Strategy Lab cells, Experiments cells). It:

  1. Resolves a backend for (strategy, backbone)
  2. Calls `backend.generate()`
  3. Persists the resulting `GenerationResult` into DuckDB as a `RunRecord`

Eval scoring is deliberately *not* part of this — the harness runs separately so
the UI can show the image immediately and update metrics asynchronously.

`persist_selfie()` is the matching service for selfie ingestion: hash → storage
→ DB row. Idempotent on the SHA-256.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from image_generator.db.repository import RunsRepository, SelfiesRepository
from image_generator.logging import get_logger
from image_generator.models.results import RunRecord
from image_generator.models.selfie import Selfie

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from image_generator.backends.registry import BackendRegistry
    from image_generator.db.connection import Database
    from image_generator.models.enums import BackendName
    from image_generator.models.requests import GenerationRequest
    from image_generator.models.results import GenerationResult
    from image_generator.storage.base import Storage

log = get_logger(__name__)


def persist_selfie(
    *,
    data: bytes,
    pil_image: PILImage,
    storage: Storage,
    db: Database,
) -> Selfie:
    """Hash bytes, write to storage, upsert into selfies table. Idempotent.

    Returns the canonical Selfie. Hashing happens on the *original* bytes — never
    on a re-encoded copy — so the same image uploaded twice yields the same SHA.
    """
    sha = Selfie.hash_bytes(data)
    path = storage.put_selfie(data, sha)
    selfie = Selfie(
        sha256=sha,
        path=path,
        uploaded_at=datetime.now(UTC),
        width=pil_image.width,
        height=pil_image.height,
    )
    SelfiesRepository(db).upsert(selfie)
    log.info("selfie.persisted", sha=sha[:8], path=str(path))
    return selfie


async def generate_and_log(
    *,
    request: GenerationRequest,
    selfie_bytes: bytes | None,
    registry: BackendRegistry,
    db: Database,
    preferred_backend: BackendName | None = None,
) -> GenerationResult:
    """Run a generation against the resolved backend and log it to DuckDB.

    Validation is enforced here (not at UI boundaries) so every entry point
    benefits — including the Strategy Lab where requests are built programmatically.
    """
    request.validate_request_consistency()
    backend = registry.resolve(request.strategy, request.backbone, preferred=preferred_backend)
    result = await backend.generate(request, selfie_bytes)
    RunsRepository(db).insert(RunRecord.from_result(result))
    log.info(
        "generation.logged",
        run_id=str(result.run_id),
        strategy=request.strategy.value,
        backbone=request.backbone.value,
        cost_usd=result.cost_usd,
    )
    return result
