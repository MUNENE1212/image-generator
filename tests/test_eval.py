"""Eval harness + service tests.

These run without the eval extra installed — real metric implementations are
mocked. An integration class at the bottom exercises the live metrics if
the eval extra is present (skipped otherwise).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from image_generator.eval.base import MetricContext, MetricResult
from image_generator.eval.harness import EvalHarness
from image_generator.eval.identity import _cosine

if TYPE_CHECKING:
    from image_generator.db.connection import Database
    from image_generator.models.results import GenerationResult


# ----------------------------- _cosine -----------------------------


class TestCosine:
    def test_identical(self) -> None:
        assert _cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self) -> None:
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            _cosine([1.0], [1.0, 2.0])


# ----------------------------- Harness orchestration -----------------------------


class _FakeMetric:
    """Configurable Metric for testing harness behaviour."""

    def __init__(
        self,
        name: str,
        *,
        score: float | None = 0.5,
        applicable_result: bool = True,
        raise_on_compute: type[Exception] | None = None,
    ) -> None:
        self.name = name
        self._score = score
        self._applicable = applicable_result
        self._raise = raise_on_compute
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def applicable(self, ctx: MetricContext) -> bool:
        return self._applicable

    def compute(self, ctx: MetricContext) -> MetricResult:
        if self._raise is not None:
            raise self._raise("simulated metric failure")
        return MetricResult(name=self.name, score=self._score)


@pytest.fixture
def fake_ctx(tmp_path: Path) -> MetricContext:
    img = tmp_path / "out.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    return MetricContext(
        generated_image=img,
        prompt="a test image",
        selfie_path=None,
        selfie_sha256=None,
    )


class TestEvalHarness:
    def test_runs_all_applicable(self, fake_ctx: MetricContext) -> None:
        m1 = _FakeMetric("m1", score=0.7)
        m2 = _FakeMetric("m2", score=0.3)
        results = EvalHarness([m1, m2]).evaluate(fake_ctx)
        assert {r.name: r.score for r in results} == {"m1": 0.7, "m2": 0.3}

    def test_skips_inapplicable(self, fake_ctx: MetricContext) -> None:
        m1 = _FakeMetric("m1", score=0.7, applicable_result=False)
        m2 = _FakeMetric("m2", score=0.3)
        results = EvalHarness([m1, m2]).evaluate(fake_ctx)
        assert {r.name: r.score for r in results} == {"m1": None, "m2": 0.3}

    def test_swallows_metric_exceptions(self, fake_ctx: MetricContext) -> None:
        # Metrics that crash must not fail the run — they return None and the
        # corresponding DB column stays NULL.
        m1 = _FakeMetric("m1", raise_on_compute=RuntimeError)
        m2 = _FakeMetric("m2", score=0.9)
        results = EvalHarness([m1, m2]).evaluate(fake_ctx)
        assert {r.name: r.score for r in results} == {"m1": None, "m2": 0.9}

    def test_swallows_not_implemented(self, fake_ctx: MetricContext) -> None:
        m1 = _FakeMetric("m1", raise_on_compute=NotImplementedError)
        m2 = _FakeMetric("m2", score=0.5)
        results = EvalHarness([m1, m2]).evaluate(fake_ctx)
        assert {r.name: r.score for r in results} == {"m1": None, "m2": 0.5}

    def test_load_called_once_per_metric(self, fake_ctx: MetricContext) -> None:
        # Two evaluate() calls should only load() each metric once (lazy + cached).
        m = _FakeMetric("m1", score=0.8)
        harness = EvalHarness([m])
        harness.evaluate(fake_ctx)
        harness.evaluate(fake_ctx)
        assert m.load_calls == 1


# ----------------------------- Service: evaluate_run -----------------------------


class TestEvaluateRun:
    def test_persists_metrics_to_db(
        self, db: Database, sample_result: GenerationResult, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from image_generator.db.repository import RunsRepository
        from image_generator.models.results import RunRecord

        # Seed the DB with a run + create an actual image file the service will load.
        image_path = Path(str(sample_result.image_path))
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        runs_repo = RunsRepository(db)
        runs_repo.insert(RunRecord.from_result(sample_result))

        # Stub the harness so evaluate_run doesn't try to load real ML models.
        from image_generator.services import eval as eval_svc

        fake_metrics = [
            MetricResult(name="identity_arcface", score=0.72),
            MetricResult(name="prompt_siglip", score=0.31),
            MetricResult(name="aesthetic_laion", score=6.4),
            MetricResult(name="diversity_lpips", score=None),
        ]

        class FakeHarness:
            def evaluate(self, ctx: MetricContext) -> list[MetricResult]:
                return fake_metrics

        monkeypatch.setattr(eval_svc, "get_harness", lambda _db: FakeHarness())

        scores = eval_svc.evaluate_run(run_id=str(sample_result.run_id), db=db)
        assert scores["identity_arcface"] == 0.72
        assert scores["prompt_siglip"] == 0.31

        # Verify persistence
        rows = runs_repo.recent(limit=1)
        assert rows[0]["identity_arcface"] == 0.72
        assert rows[0]["aesthetic_laion"] == 6.4

    def test_unknown_run_id_raises(
        self, db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from image_generator.services import eval as eval_svc

        with pytest.raises(ValueError, match="run_id not found"):
            eval_svc.evaluate_run(run_id=str(uuid4()), db=db)


# ----------------------------- Live integration (skipped without eval extra) -----------------------------

EVAL_AVAILABLE = all(
    importlib.util.find_spec(mod) is not None
    for mod in ["torch", "insightface", "open_clip", "lpips"]
)


@pytest.mark.integration
@pytest.mark.skipif(not EVAL_AVAILABLE, reason="eval extra not installed")
@pytest.mark.slow
class TestLiveEvalImports:
    """Verify the heavy ML deps actually import + the metric classes can be loaded.

    Doesn't run inference (slow + needs real images) — that's covered by
    end-to-end tests in scripts/.
    """

    def test_arcface_load(self) -> None:
        from image_generator.eval.identity import ArcFaceIdentity

        m = ArcFaceIdentity()
        m.load()  # downloads ~280MB on first call

    def test_siglip_load(self) -> None:
        from image_generator.eval.prompt import SiglipPromptAdherence

        m = SiglipPromptAdherence()
        m.load()  # downloads ~700MB on first call

    def test_lpips_load(self) -> None:
        from image_generator.eval.diversity import LpipsDiversity

        m = LpipsDiversity()
        m.load()
