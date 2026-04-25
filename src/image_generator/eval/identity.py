"""Identity similarity metrics.

ArcFace is the v1 implementation. AdaFace is left as a stub — the harness
treats it as inapplicable when its `compute()` raises NotImplementedError, so
it appears as NULL in DuckDB rather than failing the run.

Selfie face embeddings are cached in DuckDB (`face_embeddings` table) keyed by
selfie SHA-256, so re-evaluating runs from the same selfie costs one face
detection on the output, not two.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from image_generator.eval.base import Metric, MetricContext, MetricResult
from image_generator.logging import get_logger

if TYPE_CHECKING:
    from image_generator.db.repository import SelfiesRepository

log = get_logger(__name__)


def _cosine(a: list[float] | Any, b: list[float] | Any) -> float:
    """Cosine similarity between two equal-length numeric vectors. Pure Python
    fallback — works without numpy."""
    import math

    a_list = list(a)
    b_list = list(b)
    if len(a_list) != len(b_list):
        raise ValueError(f"Vector length mismatch: {len(a_list)} vs {len(b_list)}")
    dot = sum(x * y for x, y in zip(a_list, b_list, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a_list))
    norm_b = math.sqrt(sum(y * y for y in b_list))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class ArcFaceIdentity(Metric):
    """Cosine similarity between selfie face embedding and output face embedding.

    Uses InsightFace's `buffalo_l` pack (RetinaFace + ArcFace ResNet-100) on CPU.
    First call downloads ~280MB of model weights; subsequent calls reuse the
    in-process cache.

    Output range: [-1, 1] in theory; in practice ArcFace embeddings are normalized
    so values cluster in [0.2, 0.9]. A typical "same person" threshold is 0.4.
    """

    name = "identity_arcface"

    def __init__(self, selfies_repo: SelfiesRepository | None = None) -> None:
        self._selfies_repo = selfies_repo
        self._app: Any | None = None

    def load(self) -> None:
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "ArcFace requires the eval extra: `make install-eval`"
            ) from e
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        # ctx_id=-1 forces CPU; det_size 640 is the default and enough for selfies.
        app.prepare(ctx_id=-1, det_size=(640, 640))
        self._app = app
        log.info("eval.arcface.loaded")

    def applicable(self, ctx: MetricContext) -> bool:
        return ctx.selfie_path is not None

    def compute(self, ctx: MetricContext) -> MetricResult:
        if self._app is None:
            self.load()
        assert self._app is not None  # for type narrowing
        if ctx.selfie_path is None:
            return MetricResult(name=self.name, score=None)

        # 1. Selfie embedding (cached in DB if we have a repo + sha)
        selfie_emb = self._get_or_compute_selfie_embedding(ctx)
        if selfie_emb is None:
            log.warning("eval.arcface.no_selfie_face", selfie=str(ctx.selfie_path))
            return MetricResult(name=self.name, score=None)

        # 2. Output image embedding (always fresh)
        output_emb = self._embed_image(str(ctx.generated_image))
        if output_emb is None:
            log.warning("eval.arcface.no_output_face", image=str(ctx.generated_image))
            return MetricResult(name=self.name, score=None)

        return MetricResult(name=self.name, score=_cosine(selfie_emb, output_emb))

    def _embed_image(self, path: str) -> list[float] | None:
        """Return the largest detected face's embedding, or None if no face."""
        import cv2  # bundled with insightface

        img = cv2.imread(path)
        if img is None:
            return None
        faces = self._app.get(img)  # type: ignore[union-attr]
        if not faces:
            return None
        # Pick the largest-area detection.
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces[0].normed_embedding.tolist()  # type: ignore[no-any-return]

    def _get_or_compute_selfie_embedding(self, ctx: MetricContext) -> list[float] | None:
        # If we have both a repo and a sha, try the cache first.
        if self._selfies_repo is not None and ctx.selfie_sha256 is not None:
            cached = self._selfies_repo.get_embedding(ctx.selfie_sha256, "arcface_buffalo_l")
            if cached is not None:
                return cached.vector

        emb = self._embed_image(str(ctx.selfie_path))
        if emb is None:
            return None

        # Cache for next time.
        if self._selfies_repo is not None and ctx.selfie_sha256 is not None:
            from image_generator.models.selfie import FaceEmbedding

            self._selfies_repo.put_embedding(
                FaceEmbedding(
                    selfie_sha256=ctx.selfie_sha256,
                    model_name="arcface_buffalo_l",
                    vector=emb,
                    computed_at=datetime.now(UTC),
                )
            )
        return emb


class AdaFaceIdentity(Metric):
    """AdaFace cross-check for ArcFace. Deferred — see eval/__init__ for status."""

    name = "identity_adaface"

    def load(self) -> None: ...

    def applicable(self, ctx: MetricContext) -> bool:
        return ctx.selfie_path is not None

    def compute(self, ctx: MetricContext) -> MetricResult:
        raise NotImplementedError(
            "AdaFace deferred — ArcFace alone is the v1 identity metric. "
            "Add this when you need a robustness cross-check."
        )
