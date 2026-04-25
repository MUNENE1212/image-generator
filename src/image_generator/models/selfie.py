"""Selfie input + its cached face embedding."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Selfie(BaseModel):
    """A user-uploaded selfie, content-addressed by SHA-256.

    Caching identity-similarity scoring depends on this hash being stable:
    compute it from the raw bytes *before* any re-encoding.
    """

    model_config = ConfigDict(frozen=True)

    sha256: str = Field(..., min_length=64, max_length=64)
    path: Path
    uploaded_at: datetime
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()


class FaceEmbedding(BaseModel):
    """Cached face embedding for a selfie. One row per (selfie, model) pair."""

    selfie_sha256: str = Field(..., min_length=64, max_length=64)
    model_name: str  # e.g. "arcface_r100", "adaface_ir101"
    vector: list[float]
    computed_at: datetime
