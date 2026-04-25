"""LPIPS perceptual diversity — guards against mode collapse on identity strategies.

Mean LPIPS distance between every pair of sibling images. Higher = more visual
variety; values near 0 mean the strategy is producing the same face every time
regardless of prompt/seed. Combine with identity_arcface to spot the failure
mode "high identity but no diversity = the strategy memorized one face."
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from image_generator.eval.base import Metric, MetricContext, MetricResult
from image_generator.logging import get_logger

log = get_logger(__name__)


class LpipsDiversity(Metric):
    name = "diversity_lpips"

    def __init__(self) -> None:
        self._model: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import lpips
            import torch
        except ImportError as e:
            raise ImportError(
                "LPIPS requires the eval extra: `make install-eval`"
            ) from e

        # AlexNet backbone — fastest LPIPS variant; VGG is slightly more accurate
        # but ~3× slower. AlexNet is the published default.
        model = lpips.LPIPS(net="alex", verbose=False)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self._model = model
        log.info("eval.lpips.loaded")

    def applicable(self, ctx: MetricContext) -> bool:
        # Need at least 2 images to compute pairwise distances.
        return len(ctx.siblings) >= 2

    def compute(self, ctx: MetricContext) -> MetricResult:
        if self._model is None:
            self.load()
        assert self._model is not None

        import torch
        from PIL import Image
        from torchvision import transforms

        device = next(self._model.parameters()).device

        # LPIPS expects [-1, 1] range, 64×64 minimum, RGB.
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        tensors = []
        for path in ctx.siblings:
            img = Image.open(str(path)).convert("RGB")
            tensors.append(preprocess(img).unsqueeze(0).to(device))

        with torch.no_grad():
            distances = [
                self._model(a, b).item() for a, b in combinations(tensors, 2)
            ]

        if not distances:
            return MetricResult(name=self.name, score=None)
        return MetricResult(name=self.name, score=float(sum(distances) / len(distances)))
