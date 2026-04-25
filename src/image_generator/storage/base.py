"""Storage protocol.

Implementations are interchangeable — the rest of the app only sees this interface.
Swap to S3/R2 by writing `S3Storage(Storage)` later; no callers change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path


class Storage(Protocol):
    """Content-addressed binary storage."""

    def put_selfie(self, data: bytes, sha256: str) -> Path:
        """Persist a selfie and return its canonical path."""
        ...

    def put_output(self, data: bytes, run_id: str, ext: str = "png") -> Path:
        """Persist a generated image for a run."""
        ...

    def put_lora(self, data: bytes, name: str) -> Path:
        """Persist a trained LoRA artifact by name."""
        ...

    def read(self, path: Path) -> bytes:
        """Read bytes at a path previously returned by this Storage."""
        ...

    def exists(self, path: Path) -> bool: ...
