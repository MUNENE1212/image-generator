"""Static catalog of (strategy, backbone) cells.

This is the source of truth for what the Strategy Lab and Experiments page show.
A cell is `supported=True` only if at least one backend can serve it.
"""

from __future__ import annotations

from dataclasses import dataclass

from image_generator.models.enums import Backbone, Strategy


@dataclass(frozen=True, slots=True)
class StrategyCell:
    """One cell of the strategy × backbone matrix."""

    strategy: Strategy
    backbone: Backbone
    # Short marketing label shown on Lab cards.
    tagline: str
    # 1–2 sentence tradeoff summary — powers the About page.
    tradeoff: str
    # Is subject-specific training required before this cell can be used?
    requires_training: bool = False


CATALOG: tuple[StrategyCell, ...] = (
    StrategyCell(
        Strategy.PROMPT_ONLY, Backbone.SDXL,
        "Baseline",
        "No identity conditioning — the model generates from the prompt alone. Useful as the control cell to see how much the selfie is actually contributing.",
    ),
    StrategyCell(
        Strategy.PROMPT_ONLY, Backbone.FLUX_DEV,
        "Baseline (FLUX)",
        "FLUX has stronger prompt adherence than SDXL, so this cell measures how much identity methods must 'fight' the backbone.",
    ),
    StrategyCell(
        Strategy.IP_ADAPTER_FACEID, Backbone.SDXL,
        "Cheap & fast",
        "Image-prompt adapter conditioned on face embedding. Low identity fidelity but fast and composable with other adapters.",
    ),
    StrategyCell(
        Strategy.INSTANT_ID, Backbone.SDXL,
        "Strong ID, pose-locked",
        "Combines face embedding with a facial landmark map. High identity fidelity but inherits the selfie's pose — limits prompt flexibility.",
    ),
    StrategyCell(
        Strategy.PHOTOMAKER, Backbone.SDXL,
        "Prompt-flexible ID",
        "Stacks multiple reference images into a single class token. Best prompt adherence of SDXL-era methods; medium identity fidelity.",
    ),
    StrategyCell(
        Strategy.PULID, Backbone.FLUX_DEV,
        "SOTA zero-shot",
        "Trains an ID encoder aligned with the backbone via contrastive + diffusion losses. Currently the strongest zero-shot identity on FLUX.",
    ),
    StrategyCell(
        Strategy.LORA, Backbone.SDXL,
        "Per-subject LoRA (SDXL)",
        "Fine-tune a rank-16 adapter on 10–20 selfies. Highest flexibility + fidelity, but costs ~$2 and 10 min of training.",
        requires_training=True,
    ),
    StrategyCell(
        Strategy.LORA, Backbone.FLUX_DEV,
        "Per-subject LoRA (FLUX)",
        "Same idea on FLUX. More expensive to train but sharper output; expect ~$5 and 20 min.",
        requires_training=True,
    ),
)


def supported_cells(
    *,
    include_training: bool = True,
) -> list[StrategyCell]:
    """Return catalog cells, optionally excluding training-required ones."""
    if include_training:
        return list(CATALOG)
    return [c for c in CATALOG if not c.requires_training]
