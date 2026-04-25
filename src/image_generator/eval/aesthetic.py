"""Aesthetic metrics.

LAION aesthetic predictor is a tiny MLP on top of CLIP ViT-L/14 features. It's
trained on user-rated images and outputs a 1–10 score. Has known biases (favours
2022-era LAION aesthetics: oversaturated, painterly) — keep that in mind when
interpreting; not a ground truth, just a signal.

Q-Align (a stronger LMM-based aesthetic scorer) is left as a stub — the v1
metric stack ships LAION only to keep model downloads under ~1GB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from image_generator.eval.base import Metric, MetricContext, MetricResult
from image_generator.logging import get_logger

log = get_logger(__name__)

# LAION's official aesthetic predictor weights (~30MB MLP).
# Mirror: https://github.com/LAION-AI/aesthetic-predictor
_LAION_WEIGHTS_URL = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth"
_LAION_WEIGHTS_FILENAME = "laion_aesthetic_v1.pth"


class LaionAesthetic(Metric):
    name = "aesthetic_laion"

    def __init__(self) -> None:
        self._mlp: Any | None = None
        self._clip_model: Any | None = None
        self._clip_preprocess: Any | None = None

    def load(self) -> None:
        if self._mlp is not None:
            return
        try:
            import open_clip
            import torch
            from torch import nn
        except ImportError as e:
            raise ImportError(
                "LAION aesthetic requires the eval extra: `make install-eval`"
            ) from e

        # CLIP ViT-L/14 backbone (~900MB) — outputs 768-d features.
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        clip_model.eval()
        if torch.cuda.is_available():
            clip_model = clip_model.cuda()
        self._clip_model = clip_model
        self._clip_preprocess = clip_preprocess

        # Tiny MLP head: 768 → 1 scalar score.
        # Architecture matches LAION's published predictor.
        mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

        weights_path = self._download_weights()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        mlp.load_state_dict(state)
        mlp.eval()
        if torch.cuda.is_available():
            mlp = mlp.cuda()
        self._mlp = mlp
        log.info("eval.laion.loaded")

    @staticmethod
    def _download_weights() -> Path:
        """Cache the predictor weights under ~/.cache/image-generator/."""
        import urllib.request

        cache_dir = Path.home() / ".cache" / "image-generator"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / _LAION_WEIGHTS_FILENAME
        if not path.exists():
            log.info("eval.laion.downloading", url=_LAION_WEIGHTS_URL)
            urllib.request.urlretrieve(_LAION_WEIGHTS_URL, path)
        return path

    def applicable(self, ctx: MetricContext) -> bool:
        return True

    def compute(self, ctx: MetricContext) -> MetricResult:
        if self._mlp is None:
            self.load()
        assert (
            self._mlp is not None
            and self._clip_model is not None
            and self._clip_preprocess is not None
        )

        import torch
        from PIL import Image

        device = next(self._clip_model.parameters()).device
        image = Image.open(str(ctx.generated_image)).convert("RGB")
        image_tensor = self._clip_preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = self._clip_model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            score = self._mlp(features.float()).item()

        return MetricResult(name=self.name, score=float(score))


class QAlignAesthetic(Metric):
    """Q-Align aesthetic — deferred. Heavier alternative to LAION."""

    name = "aesthetic_qalign"

    def load(self) -> None: ...

    def applicable(self, ctx: MetricContext) -> bool:
        return True

    def compute(self, ctx: MetricContext) -> MetricResult:
        raise NotImplementedError(
            "Q-Align deferred — LAION is the v1 aesthetic metric. "
            "Add this when you need a stronger cross-check."
        )
