"""Enumerations for strategies, backbones, and backends.

These are deliberately string-valued so they round-trip through DuckDB and JSON
without custom adapters.
"""

from __future__ import annotations

from enum import StrEnum


class Strategy(StrEnum):
    """Identity-preservation strategy applied on top of a diffusion backbone."""

    PROMPT_ONLY = "prompt_only"
    IP_ADAPTER_FACEID = "ip_adapter_faceid"
    INSTANT_ID = "instant_id"
    PHOTOMAKER = "photomaker"
    PULID = "pulid"
    LORA = "lora"

    @property
    def requires_training(self) -> bool:
        return self is Strategy.LORA

    @property
    def display_name(self) -> str:
        return {
            Strategy.PROMPT_ONLY: "Prompt only",
            Strategy.IP_ADAPTER_FACEID: "IP-Adapter FaceID",
            Strategy.INSTANT_ID: "InstantID",
            Strategy.PHOTOMAKER: "PhotoMaker v2",
            Strategy.PULID: "PuLID",
            Strategy.LORA: "LoRA (per-subject)",
        }[self]


class Backbone(StrEnum):
    """Diffusion backbone. Primary research axis alongside Strategy."""

    SDXL = "sdxl"
    FLUX_DEV = "flux_dev"

    @property
    def display_name(self) -> str:
        return {Backbone.SDXL: "SDXL", Backbone.FLUX_DEV: "FLUX.1 [dev]"}[self]


class BackendName(StrEnum):
    """Compute backend that executes a generation."""

    REPLICATE = "replicate"
    FAL = "fal"


class TrainingMethod(StrEnum):
    """Subject-adaptation training method offered in the Training Studio."""

    LORA_SDXL = "lora_sdxl"
    LORA_FLUX = "lora_flux"
