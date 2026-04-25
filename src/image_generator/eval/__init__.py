"""Automated metrics for generated images.

Implementations live behind the `Metric` protocol so the orchestrator doesn't
care whether identity is ArcFace or AdaFace, or whether aesthetic is LAION or
Q-Align. Real ML dependencies are installed via the `eval` extra.
"""

from image_generator.eval.base import Metric, MetricContext, MetricResult
from image_generator.eval.harness import EvalHarness

__all__ = ["EvalHarness", "Metric", "MetricContext", "MetricResult"]
