"""Parameter sweep orchestrator.

Given a `SweepConfig` (one base `GenerationRequest` + N `SweepAxis`), expand it
into the cartesian product of cells, then execute them concurrently — same
streaming pattern as `LabRunner`, but the axis being varied is hyperparameters
rather than (strategy, backbone).

Persistence is done by the caller after the iterator yields each outcome
(matches `LabRunner`'s contract — keeps the runner pure).
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from image_generator.backends.base import BackendError
from image_generator.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from image_generator.backends.registry import BackendRegistry
    from image_generator.models.requests import GenerationRequest, SweepConfig
    from image_generator.models.results import GenerationResult

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SweepCell:
    """One cell of a sweep — request + which axis values produced it.

    `overrides` is preserved separately from the request so plotting and CSV
    export can group/colour by the swept axes without reverse-engineering them
    from the (now-merged) request fields.
    """

    request: GenerationRequest
    overrides: tuple[tuple[str, Any], ...]  # ((axis_field, value), ...) — frozen for hashability


@dataclass(frozen=True, slots=True)
class SweepCellOutcome:
    cell: SweepCell
    result: GenerationResult | None
    error: str | None
    elapsed_seconds: float


def _coerce(field: str, value: Any) -> Any:
    """Coerce sweep values into the right Python type for GenerationRequest.

    SweepAxis values are typed as `float | int | str` for serialization
    convenience, but specific request fields need ints (steps, seed) or
    floats (guidance_scale, identity_strength). We coerce here so axes
    can be specified loosely in the UI.
    """
    int_fields = {"num_inference_steps", "seed", "width", "height"}
    if field in int_fields:
        return int(value)
    return value


def _expand(config: SweepConfig) -> list[SweepCell]:
    """Cartesian product of axes → list of SweepCells.

    Each axis contributes one (field, value) tuple per cell. The request is
    built by `model_copy(update=...)` over the base, so all unswept fields
    inherit from the base unchanged.
    """
    field_names = [axis.field for axis in config.axes]
    value_grids = [axis.values for axis in config.axes]

    cells: list[SweepCell] = []
    for combo in itertools.product(*value_grids):
        update = {field: _coerce(field, value) for field, value in zip(field_names, combo, strict=True)}
        # If 'seed' isn't in the swept axes, derive a deterministic per-cell seed
        # from the override key — keeps cells reproducible without forcing the
        # user to vary seed explicitly.
        if "seed" not in update:
            cell_key = "|".join(f"{f}={v}" for f, v in sorted(update.items()))
            digest_bytes = hashlib.blake2b(cell_key.encode(), digest_size=4).digest()
            update["seed"] = int.from_bytes(digest_bytes, "big") & 0x7FFFFFFF

        request = config.base.model_copy(update=update)
        # `overrides` records ONLY what was actually swept (excludes the derived seed).
        overrides = tuple(
            (f, _coerce(f, v)) for f, v in zip(field_names, combo, strict=True)
        )
        cells.append(SweepCell(request=request, overrides=overrides))
    return cells


class SweepRunner:
    """Runs the cartesian-product cells of a sweep concurrently."""

    def __init__(self, registry: BackendRegistry, *, max_concurrent: int = 4) -> None:
        self._registry = registry
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @staticmethod
    def expand(config: SweepConfig) -> list[SweepCell]:
        """Public entry point. Pure function — no I/O."""
        return _expand(config)

    async def run(
        self,
        *,
        cells: list[SweepCell],
        selfie_bytes: bytes | None,
    ) -> AsyncIterator[SweepCellOutcome]:
        """Dispatch all cells in parallel; yield outcomes as they complete."""
        tasks = [asyncio.create_task(self._run_cell(cell, selfie_bytes)) for cell in cells]
        for coro in asyncio.as_completed(tasks):
            yield await coro

    async def _run_cell(self, cell: SweepCell, selfie_bytes: bytes | None) -> SweepCellOutcome:
        start = perf_counter()
        try:
            cell.request.validate_request_consistency()
            backend = self._registry.resolve(cell.request.strategy, cell.request.backbone)
            async with self._semaphore:
                result = await backend.generate(cell.request, selfie_bytes)
            return SweepCellOutcome(
                cell=cell, result=result, error=None, elapsed_seconds=perf_counter() - start
            )
        except (BackendError, ValueError, NotImplementedError) as e:
            log.warning("sweep.cell_failed", overrides=dict(cell.overrides), error=str(e))
            return SweepCellOutcome(
                cell=cell, result=None, error=str(e), elapsed_seconds=perf_counter() - start
            )
