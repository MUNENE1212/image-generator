"""Pydantic data contract. All cross-module types live here."""

from image_generator.models.enums import Backbone, BackendName, Strategy, TrainingMethod
from image_generator.models.requests import GenerationRequest, SweepAxis, SweepConfig
from image_generator.models.results import EvalMetrics, GenerationResult, RunRecord
from image_generator.models.selfie import FaceEmbedding, Selfie

__all__ = [
    "Backbone",
    "BackendName",
    "EvalMetrics",
    "FaceEmbedding",
    "GenerationRequest",
    "GenerationResult",
    "RunRecord",
    "Selfie",
    "Strategy",
    "SweepAxis",
    "SweepConfig",
    "TrainingMethod",
]
