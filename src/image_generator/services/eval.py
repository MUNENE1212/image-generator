"""Eval orchestration service.

`evaluate_run(run_id, ...)` is the single entry point: load the run + selfie
from DuckDB, build a MetricContext, run the harness, persist the scores.
Idempotent — calling it twice on the same run just recomputes (and overwrites
NULL→value, keeping populated values via the COALESCE in update_metrics).

Process-global `EvalHarness` singleton via `get_harness()` so model weights
load exactly once per Streamlit process (the 700MB SigLIP weights would be
brutal to reload per call).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from image_generator.eval.aesthetic import LaionAesthetic
from image_generator.eval.base import MetricContext
from image_generator.eval.diversity import LpipsDiversity
from image_generator.eval.harness import EvalHarness
from image_generator.eval.identity import ArcFaceIdentity
from image_generator.eval.prompt import SiglipPromptAdherence
from image_generator.logging import get_logger

if TYPE_CHECKING:
    from image_generator.db.connection import Database


log = get_logger(__name__)

_harness_singleton: EvalHarness | None = None


def get_harness(db: Database) -> EvalHarness:
    """Build (once) the v1 metric stack: ArcFace + SigLIP + LAION + LPIPS.

    AdaFace and Q-Align are deliberately excluded for v1 — they exist as
    classes for future activation but their compute() raises NotImplementedError.
    """
    global _harness_singleton
    if _harness_singleton is not None:
        return _harness_singleton

    from image_generator.db.repository import SelfiesRepository

    selfies_repo = SelfiesRepository(db)
    _harness_singleton = EvalHarness(
        metrics=[
            ArcFaceIdentity(selfies_repo=selfies_repo),
            SiglipPromptAdherence(),
            LaionAesthetic(),
            LpipsDiversity(),
        ]
    )
    return _harness_singleton


def evaluate_run(
    *,
    run_id: str,
    db: Database,
    siblings: list[Path] | None = None,
) -> dict[str, float | None]:
    """Run all applicable metrics for one run and persist to DuckDB.

    `siblings` is optional — pass other generated images from the same strategy
    to enable LPIPS diversity scoring (no-op otherwise).

    Returns the metric dict for immediate UI display so the caller doesn't have
    to re-query DuckDB.
    """
    from image_generator.db.repository import RunsRepository, SelfiesRepository

    runs_repo = RunsRepository(db)
    rows = [r for r in runs_repo.recent(limit=1000) if r["run_id"] == run_id]
    if not rows:
        raise ValueError(f"run_id not found: {run_id}")
    row = rows[0]

    image_path = Path(str(row["image_path"]))
    if not image_path.exists():
        log.warning("eval.image_missing", run_id=run_id, path=str(image_path))
        return {}

    # Selfie path (optional — only for identity metrics).
    selfie_path: Path | None = None
    selfie_sha = row.get("selfie_sha256")
    if selfie_sha is not None:
        selfie = SelfiesRepository(db).get(str(selfie_sha))
        if selfie is not None:
            selfie_path = selfie.path

    ctx = MetricContext(
        generated_image=image_path,
        prompt=str(row["prompt"]),
        selfie_path=selfie_path,
        selfie_sha256=str(selfie_sha) if selfie_sha else None,
        siblings=tuple(siblings or ()),
    )

    harness = get_harness(db)
    results = harness.evaluate(ctx)
    scores = {r.name: r.score for r in results}

    runs_repo.update_metrics(
        run_id=run_id,
        identity_arcface=scores.get("identity_arcface"),
        identity_adaface=scores.get("identity_adaface"),
        prompt_siglip=scores.get("prompt_siglip"),
        aesthetic_laion=scores.get("aesthetic_laion"),
        aesthetic_qalign=scores.get("aesthetic_qalign"),
        diversity_lpips=scores.get("diversity_lpips"),
    )

    log.info("eval.run.done", run_id=run_id, scores={k: v for k, v in scores.items() if v is not None})
    return scores


def evaluate_runs_batch(
    *,
    run_ids: list[str],
    db: Database,
) -> dict[str, dict[str, float | None]]:
    """Sequential batch eval. Diversity is computed across all batch siblings.

    Use case: after a Strategy Lab run completes, evaluate every cell with
    diversity scored against the rest of the batch.
    """
    from image_generator.db.repository import RunsRepository

    runs_repo = RunsRepository(db)
    all_runs = {r["run_id"]: r for r in runs_repo.recent(limit=1000)}

    # Build the siblings list once.
    sibling_paths = [
        Path(str(all_runs[rid]["image_path"]))
        for rid in run_ids
        if rid in all_runs
    ]

    results: dict[str, dict[str, float | None]] = {}
    for run_id in run_ids:
        # Each run's siblings = the others in the batch
        others = [p for p in sibling_paths if str(p) != str(all_runs[run_id]["image_path"])]
        results[run_id] = evaluate_run(run_id=run_id, db=db, siblings=others)
    return results
