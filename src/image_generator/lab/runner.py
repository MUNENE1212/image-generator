"""Async runner for the Strategy Lab.

Given one selfie + one prompt + N cells, build N GenerationRequests sharing a seed,
dispatch them in parallel to the appropriate backends, and yield results as they
complete so the UI can stream cells into the grid.

Seed discipline: we pass the same integer seed to every cell, but note in README
that the initial-noise mapping differs per backbone — cross-backbone "same seed"
is not the same noise. Use `seed_salt` to derive a deterministic per-cell seed
that varies across cells but is reproducible across runs.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from image_generator.backends.base import BackendError, ComputeBackend
from image_generator.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from image_generator.backends.registry import BackendRegistry
    from image_generator.models.enums import Backbone, Strategy
    from image_generator.models.requests import GenerationRequest
    from image_generator.models.results import GenerationResult
    from image_generator.strategies.catalog import StrategyCell

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CellOutcome:
    """Result of one Lab cell. Exactly one of `result` / `error` is set."""

    cell: StrategyCell
    request: GenerationRequest
    result: GenerationResult | None
    error: str | None
    elapsed_seconds: float


def _derive_seed(base_seed: int, strategy: Strategy, backbone: Backbone) -> int:
    """Deterministic per-cell seed that's reproducible but not identical across cells.

    Same selfie + prompt + base_seed → same per-cell seeds on every rerun.
    """
    key = f"{base_seed}|{strategy.value}|{backbone.value}".encode()
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


class LabRunner:
    """Runs a list of cells in parallel against a BackendRegistry."""

    def __init__(self, registry: BackendRegistry, *, max_concurrent: int = 6) -> None:
        self._registry = registry
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def run(
        self,
        *,
        cells: list[StrategyCell],
        base_request: GenerationRequest,
        selfie_bytes: bytes | None,
    ) -> AsyncIterator[CellOutcome]:
        """Dispatch `cells` in parallel; yield each outcome as it completes.

        `base_request` provides the prompt + sampling defaults; strategy/backbone
        are overridden per cell. Seeds are derived deterministically from
        `base_request.seed` so results are reproducible.
        """
        tasks = [
            asyncio.create_task(self._run_cell(cell, base_request, selfie_bytes))
            for cell in cells
        ]
        for coro in asyncio.as_completed(tasks):
            yield await coro

    async def _run_cell(
        self,
        cell: StrategyCell,
        base: GenerationRequest,
        selfie_bytes: bytes | None,
    ) -> CellOutcome:
        request = base.model_copy(
            update={
                "strategy": cell.strategy,
                "backbone": cell.backbone,
                "seed": _derive_seed(base.seed, cell.strategy, cell.backbone),
            }
        )
        start = perf_counter()
        try:
            request.validate_request_consistency()
            backend: ComputeBackend = self._registry.resolve(cell.strategy, cell.backbone)
            async with self._semaphore:
                result = await backend.generate(request, selfie_bytes)
            return CellOutcome(
                cell=cell,
                request=request,
                result=result,
                error=None,
                elapsed_seconds=perf_counter() - start,
            )
        except (BackendError, ValueError, NotImplementedError) as e:
            log.warning(
                "lab.cell_failed",
                strategy=cell.strategy.value,
                backbone=cell.backbone.value,
                error=str(e),
            )
            return CellOutcome(
                cell=cell,
                request=request,
                result=None,
                error=str(e),
                elapsed_seconds=perf_counter() - start,
            )
