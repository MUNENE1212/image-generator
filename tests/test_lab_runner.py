"""Unit tests for Lab seed-derivation. Full integration tests land in Phase 2."""

from __future__ import annotations

from image_generator.lab.runner import _derive_seed
from image_generator.models.enums import Backbone, Strategy


def test_derive_seed_deterministic() -> None:
    a = _derive_seed(42, Strategy.INSTANT_ID, Backbone.SDXL)
    b = _derive_seed(42, Strategy.INSTANT_ID, Backbone.SDXL)
    assert a == b


def test_derive_seed_varies_across_cells() -> None:
    seeds = {
        _derive_seed(42, Strategy.INSTANT_ID, Backbone.SDXL),
        _derive_seed(42, Strategy.PULID, Backbone.FLUX_DEV),
        _derive_seed(42, Strategy.PHOTOMAKER, Backbone.SDXL),
    }
    assert len(seeds) == 3


def test_derive_seed_in_valid_range() -> None:
    seed = _derive_seed(0, Strategy.PROMPT_ONLY, Backbone.SDXL)
    assert 0 <= seed <= 2**31 - 1
