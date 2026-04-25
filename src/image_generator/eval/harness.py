"""Orchestrates all metrics for one generation result.

Design notes:
- Metrics are lazy-loaded: the harness calls `load()` on first use and caches.
- Inapplicable metrics (no selfie for identity; <2 siblings for diversity) are
  skipped silently and the corresponding column stays NULL in DuckDB.
- This class is sync because metric libraries are sync. Run it in a thread via
  `anyio.to_thread.run_sync` from async callers.
"""

from __future__ import annotations

from image_generator.eval.base import Metric, MetricContext, MetricResult
from image_generator.logging import get_logger

log = get_logger(__name__)


class EvalHarness:
    """Owns a list of metrics; runs the applicable ones for each result."""

    def __init__(self, metrics: list[Metric]) -> None:
        self._metrics = metrics
        self._loaded: set[str] = set()

    def _ensure_loaded(self, metric: Metric) -> None:
        if metric.name not in self._loaded:
            metric.load()
            self._loaded.add(metric.name)

    def evaluate(self, ctx: MetricContext) -> list[MetricResult]:
        results: list[MetricResult] = []
        for metric in self._metrics:
            if not metric.applicable(ctx):
                results.append(MetricResult(name=metric.name, score=None))
                continue
            try:
                self._ensure_loaded(metric)
                results.append(metric.compute(ctx))
            except NotImplementedError:
                log.debug("eval.metric_not_implemented", metric=metric.name)
                results.append(MetricResult(name=metric.name, score=None))
            except Exception as e:
                log.warning("eval.metric_failed", metric=metric.name, error=str(e))
                results.append(MetricResult(name=metric.name, score=None))
        return results
