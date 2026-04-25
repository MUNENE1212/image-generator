"""Experiments — parameter sweeps with filterable result grid + CSV export.

Sweep over (CFG, steps, identity_strength, seed) for one fixed (strategy, backbone).
Streams results as they complete. Persists to runs + sweeps + sweep_runs tables
so any sweep can be revisited or exported later.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st
from PIL import Image

from image_generator.backends.registry import get_registry
from image_generator.db.connection import get_database
from image_generator.db.repository import RunsRepository, SweepsRepository
from image_generator.models.enums import Backbone, Strategy
from image_generator.models.requests import GenerationRequest, SweepAxis, SweepConfig
from image_generator.models.results import RunRecord
from image_generator.services.eval import evaluate_runs_batch
from image_generator.services.generation import persist_selfie
from image_generator.services.sweep import SweepCellOutcome, SweepRunner
from image_generator.storage.local import LocalStorage
from image_generator.ui.state import bootstrap_session

st.set_page_config(page_title="Experiments", page_icon="🔬", layout="wide")
bootstrap_session()

st.title("Experiments")
st.caption("Sweep hyperparameter axes for one (strategy, backbone). Export results as CSV.")

col_axes, col_preview = st.columns([1, 1])

# ------------------------------ Base + axes ------------------------------
with col_axes:
    st.markdown("### Base")
    strategy = st.selectbox(
        "Strategy", options=list(Strategy), format_func=lambda s: s.display_name
    )
    backbone = st.selectbox(
        "Backbone", options=list(Backbone), format_func=lambda b: b.display_name
    )
    prompt = st.text_area(
        "Prompt", value="portrait of a person, cinematic lighting", height=60
    )
    needs_selfie = strategy is not Strategy.PROMPT_ONLY
    uploaded = st.file_uploader("Selfie", type=["png", "jpg", "jpeg"]) if needs_selfie else None

    sweep_name = st.text_input(
        "Sweep name",
        value=f"{strategy.value}-{backbone.value}-{datetime.now(UTC).strftime('%Y%m%d-%H%M')}",
    )

    st.markdown("### Axes")
    cfg_values = st.multiselect(
        "Guidance scale", [3.0, 5.0, 7.0, 9.0, 11.0], default=[3.0, 5.0, 7.0]
    )
    step_values = st.multiselect("Steps", [20, 30, 40, 50], default=[30])
    strength_values = (
        st.multiselect("Identity strength", [0.4, 0.6, 0.8, 1.0, 1.2], default=[0.6, 0.8, 1.0])
        if needs_selfie
        else []
    )
    seed_values = st.multiselect("Seeds", [0, 1, 2, 3, 42], default=[0, 1])

axes: list[SweepAxis] = []
if cfg_values:
    axes.append(SweepAxis(field="guidance_scale", values=list(cfg_values)))
if step_values:
    axes.append(SweepAxis(field="num_inference_steps", values=list(step_values)))
if strength_values:
    axes.append(SweepAxis(field="identity_strength", values=list(strength_values)))
if seed_values:
    axes.append(SweepAxis(field="seed", values=list(seed_values)))

total = 1
for a in axes:
    total *= max(1, len(a.values))

# Cost estimate uses the same per-backbone numbers as the LabRunner price tables.
per_cell = 0.030 if backbone is Backbone.FLUX_DEV else 0.005

with col_preview:
    st.markdown("### Preview")
    cols_metrics = st.columns(3)
    cols_metrics[0].metric("Total cells", total if axes else 0)
    cols_metrics[1].metric("Estimated cost", f"${total * per_cell:.3f}" if axes else "—")
    cols_metrics[2].metric("Axes", len(axes))

    if axes:
        st.caption("Axes being swept:")
        for a in axes:
            st.write(f"- **{a.field}** ({len(a.values)} values): {a.values}")
    else:
        st.warning("Pick at least one axis with at least one value.")

run_clicked = st.button(
    "Run Sweep",
    type="primary",
    disabled=not axes or (needs_selfie and uploaded is None),
)


# ------------------------------ Streaming runner ------------------------------


async def _run_sweep(
    *,
    runner: SweepRunner,
    cells: list[Any],
    selfie_bytes: bytes | None,
    progress: Any,
    status_label: Any,
    sweep_id: str,
) -> list[SweepCellOutcome]:
    db = get_database()
    runs_repo = RunsRepository(db)
    sweeps_repo = SweepsRepository(db)

    outcomes: list[SweepCellOutcome] = []
    completed = 0
    total = len(cells)

    async for outcome in runner.run(cells=cells, selfie_bytes=selfie_bytes):
        completed += 1
        progress.progress(completed / total, text=f"{completed}/{total} cells complete")
        if outcome.result is not None:
            runs_repo.insert(RunRecord.from_result(outcome.result))
            sweeps_repo.link_run(sweep_id, str(outcome.result.run_id))
        else:
            status_label.warning(f"Cell failed: {outcome.error}", icon="⚠️")
        outcomes.append(outcome)

    return outcomes


# ------------------------------ Run + render ------------------------------

if run_clicked:
    registry = get_registry()
    if not registry.available:
        st.error("No backend credentials configured.")
        st.stop()

    db = get_database()
    storage = LocalStorage()

    selfie_bytes: bytes | None = None
    selfie_sha: str | None = None
    if uploaded is not None:
        selfie_bytes = uploaded.getvalue()
        pil = Image.open(io.BytesIO(selfie_bytes))
        selfie = persist_selfie(data=selfie_bytes, pil_image=pil, storage=storage, db=db)
        selfie_sha = selfie.sha256

    base_request = GenerationRequest(
        strategy=strategy,
        backbone=backbone,
        prompt=prompt,
        selfie_sha256=selfie_sha,
    )
    config = SweepConfig(base=base_request, axes=axes)
    cells = SweepRunner.expand(config)

    sweep_id = str(uuid4())
    SweepsRepository(db).insert(
        sweep_id=sweep_id,
        name=sweep_name,
        config_json=json.dumps(
            {
                "base": base_request.model_dump(mode="json"),
                "axes": [a.model_dump() for a in axes],
            },
            default=str,
        ),
        created_at=datetime.now(UTC),
    )

    st.markdown(f"## Sweep: `{sweep_name}`")
    progress = st.progress(0.0, text=f"0/{len(cells)} cells complete")
    status = st.empty()

    runner = SweepRunner(registry, max_concurrent=4)
    with st.spinner(f"Running {len(cells)} cells…"):
        outcomes = asyncio.run(
            _run_sweep(
                runner=runner,
                cells=cells,
                selfie_bytes=selfie_bytes,
                progress=progress,
                status_label=status,
                sweep_id=sweep_id,
            )
        )

    successful = [o for o in outcomes if o.result is not None]
    total_cost = sum(o.result.cost_usd for o in successful if o.result is not None)
    wall_time = max((o.elapsed_seconds for o in outcomes), default=0.0)

    st.markdown("### Summary")
    cols = st.columns(4)
    cols[0].metric("Cells", f"{len(successful)}/{len(outcomes)}")
    cols[1].metric("Total cost", f"${total_cost:.3f}")
    cols[2].metric("Wall time", f"{wall_time:.1f}s")
    cols[3].metric(
        "Avg cell time",
        f"{(sum(o.elapsed_seconds for o in outcomes) / len(outcomes)):.1f}s" if outcomes else "—",
    )

    # ------------------------------ Results table + CSV ------------------------------
    if successful:
        # Batch eval — diversity_lpips works because all sibling images share the sweep.
        with st.spinner("Computing metrics for the sweep…"):
            try:
                evaluate_runs_batch(
                    run_ids=[str(o.result.run_id) for o in successful if o.result is not None],
                    db=db,
                )
            except ImportError as e:
                st.info(f"Eval extra not installed — metrics skipped. ({e})")
            except Exception as e:
                st.warning(f"Eval failed: {e}")

        st.markdown("### Results")
        rows = SweepsRepository(db).get_runs(sweep_id)

        # Streamlit dataframe view (incl. metrics now that eval has populated them)
        st.dataframe(
            [
                {
                    "guidance_scale": r["guidance_scale"],
                    "num_inference_steps": r["num_inference_steps"],
                    "identity_strength": r["identity_strength"],
                    "seed": r["seed"],
                    "cost_usd": r["cost_usd"],
                    "duration_s": r["duration_seconds"],
                    "identity": r.get("identity_arcface"),
                    "prompt": r.get("prompt_siglip"),
                    "aesthetic": r.get("aesthetic_laion"),
                    "diversity": r.get("diversity_lpips"),
                    "image": Path(str(r["image_path"])).name,
                }
                for r in rows
            ],
            hide_index=True,
            use_container_width=True,
        )

        # CSV export
        csv_buf = io.StringIO()
        if rows:
            writer = csv.DictWriter(csv_buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow({k: ("" if v is None else v) for k, v in r.items()})
        st.download_button(
            "Download CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"{sweep_name}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Image grid (4 per row)
        st.markdown("### Images")
        cols_per_row = 4
        for i in range(0, len(rows), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, r in enumerate(rows[i : i + cols_per_row]):
                p = Path(str(r["image_path"]))
                with row_cols[j], st.container(border=True):
                    if p.exists():
                        st.image(str(p), use_container_width=True)
                    label_parts = []
                    if "guidance_scale" in (a.field for a in axes):
                        label_parts.append(f"cfg={r['guidance_scale']}")
                    if "num_inference_steps" in (a.field for a in axes):
                        label_parts.append(f"steps={r['num_inference_steps']}")
                    if "identity_strength" in (a.field for a in axes):
                        label_parts.append(f"id={r['identity_strength']}")
                    if "seed" in (a.field for a in axes):
                        label_parts.append(f"seed={r['seed']}")
                    st.caption(" · ".join(label_parts))
