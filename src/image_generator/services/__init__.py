"""Application services. Orchestrate adapters + storage + DB + eval into one call.

Streamlit pages should call services, not adapters directly. This keeps UI thin
and the orchestration testable without spinning up Streamlit.
"""

from image_generator.services.eval import evaluate_run, evaluate_runs_batch, get_harness
from image_generator.services.generation import generate_and_log, persist_selfie
from image_generator.services.sweep import SweepCell, SweepCellOutcome, SweepRunner
from image_generator.services.training import build_selfie_archive, poll_training, start_training

__all__ = [
    "SweepCell",
    "SweepCellOutcome",
    "SweepRunner",
    "build_selfie_archive",
    "evaluate_run",
    "evaluate_runs_batch",
    "generate_and_log",
    "get_harness",
    "persist_selfie",
    "poll_training",
    "start_training",
]
