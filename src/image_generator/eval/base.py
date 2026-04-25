"""Metric protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class MetricContext:
    """Inputs available to every metric."""

    generated_image: Path
    prompt: str
    selfie_path: Path | None
    selfie_sha256: str | None
    # Sibling images from the same strategy, different seeds — for diversity metrics.
    siblings: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Output of one metric. `score` is None if the metric was not applicable."""

    name: str
    score: float | None


class Metric(Protocol):
    """Pure function: `(context) -> MetricResult`. Implementations may be stateful
    (model cached in memory) but must be thread-safe after `load()`."""

    name: str

    def load(self) -> None:
        """Expensive init (download model weights, etc). Called once."""
        ...

    def applicable(self, ctx: MetricContext) -> bool: ...

    def compute(self, ctx: MetricContext) -> MetricResult: ...
