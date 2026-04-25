"""Prompt adherence via SigLIP.

SigLIP-SO400M is a stronger image-text matcher than CLIP — pre-trained with a
sigmoid loss that handles compositional prompts better than CLIP's contrastive
objective. We return raw cosine similarity; the value isn't normalized to [0,1]
out of the box, but it's directly comparable across cells of the same prompt.
"""

from __future__ import annotations

from typing import Any

from image_generator.eval.base import Metric, MetricContext, MetricResult
from image_generator.logging import get_logger

log = get_logger(__name__)


class SiglipPromptAdherence(Metric):
    name = "prompt_siglip"

    # Pinned model. Smaller alternatives (`ViT-B-16-SigLIP-256`) cut size by 5×
    # at the cost of measurable accuracy on long prompts.
    _MODEL_NAME = "ViT-SO400M-14-SigLIP-384"
    _PRETRAINED = "webli"

    def __init__(self) -> None:
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._tokenizer: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as e:
            raise ImportError(
                "SigLIP requires the eval extra: `make install-eval`"
            ) from e

        model, _, preprocess = open_clip.create_model_and_transforms(
            self._MODEL_NAME, pretrained=self._PRETRAINED
        )
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self._MODEL_NAME)
        log.info("eval.siglip.loaded", model=self._MODEL_NAME)

    def applicable(self, ctx: MetricContext) -> bool:
        return bool(ctx.prompt)

    def compute(self, ctx: MetricContext) -> MetricResult:
        if self._model is None:
            self.load()
        assert self._model is not None and self._preprocess is not None and self._tokenizer is not None

        import torch
        from PIL import Image

        device = next(self._model.parameters()).device

        image = Image.open(str(ctx.generated_image)).convert("RGB")
        image_tensor = self._preprocess(image).unsqueeze(0).to(device)
        text_tokens = self._tokenizer([ctx.prompt]).to(device)

        with torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            text_features = self._model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            score = (image_features @ text_features.T).item()

        return MetricResult(name=self.name, score=float(score))
