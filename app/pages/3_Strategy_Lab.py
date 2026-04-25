"""Strategy Lab — the centerpiece. One selfie + one prompt × N strategies in parallel.

Streaming UX: pre-render placeholder cards for every selected cell, then stream
LabRunner outcomes into them via asyncio.as_completed. Each card replaces its
contents three times: queued → spinner → final (image or error).
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

import streamlit as st
from PIL import Image

from image_generator.backends.registry import get_registry
from image_generator.db.connection import get_database
from image_generator.db.repository import RunsRepository
from image_generator.lab.runner import CellOutcome, LabRunner
from image_generator.models.enums import Backbone, Strategy
from image_generator.models.requests import GenerationRequest
from image_generator.models.results import RunRecord
from image_generator.services.eval import evaluate_run
from image_generator.services.generation import persist_selfie
from image_generator.storage.local import LocalStorage
from image_generator.strategies.catalog import CATALOG, StrategyCell
from image_generator.ui.state import bootstrap_session

st.set_page_config(page_title="Strategy Lab", page_icon="🧪", layout="wide")
bootstrap_session()

st.title("Strategy Lab")
st.caption("Compare identity-preservation strategies side-by-side on the same subject and prompt.")

# ------------------------------ Inputs ------------------------------
uploaded = st.file_uploader("Selfie", type=["png", "jpg", "jpeg"])
prompt = st.text_area(
    "Prompt", value="portrait of a person, cinematic lighting, 85mm lens", height=80
)

with st.expander("Advanced", expanded=False):
    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=2**31 - 1,
        value=42,
        help="Each cell derives a per-cell seed from this base, so cells differ but the whole Lab run is reproducible.",
    )
    steps = st.slider("Steps", 10, 80, 30)
    guidance = st.slider("Guidance", 1.0, 15.0, 5.0)
    identity_strength = st.slider(
        "Identity strength",
        0.0,
        1.5,
        0.8,
        help="Strategy-specific knob. For InstantID/IP-Adapter it's the IP scale; for PuLID it's the ID weight.",
    )
    max_concurrent = st.slider(
        "Max concurrent cells",
        1,
        8,
        4,
        help="How many cells run in parallel. Higher = faster wall time but bigger spike on the backend's rate limit.",
    )

st.markdown("### Cells to run")

col_sdxl, col_flux = st.columns(2)
selected: list[StrategyCell] = []

with col_sdxl:
    st.markdown("**SDXL**")
    for cell in CATALOG:
        if cell.backbone is Backbone.SDXL and st.checkbox(
            f"{cell.strategy.display_name} — {cell.tagline}",
            value=cell.strategy in {Strategy.PROMPT_ONLY, Strategy.INSTANT_ID, Strategy.PHOTOMAKER},
            key=f"sdxl_{cell.strategy.value}",
            disabled=cell.requires_training,
            help="Requires a trained LoRA — coming in the Training Studio." if cell.requires_training else None,
        ):
            selected.append(cell)

with col_flux:
    st.markdown("**FLUX.1 [dev]**")
    for cell in CATALOG:
        if cell.backbone is Backbone.FLUX_DEV and st.checkbox(
            f"{cell.strategy.display_name} — {cell.tagline}",
            value=cell.strategy is Strategy.PULID,
            key=f"flux_{cell.strategy.value}",
            disabled=cell.requires_training,
            help="Requires a trained LoRA — coming in the Training Studio." if cell.requires_training else None,
        ):
            selected.append(cell)

st.markdown("---")
cost_estimate = sum(0.030 if c.backbone is Backbone.FLUX_DEV else 0.005 for c in selected)
needs_selfie = any(c.strategy is not Strategy.PROMPT_ONLY for c in selected)

st.caption(
    f"{len(selected)} cells selected · estimated cost ≈ ${cost_estimate:.3f}"
    f"{' · selfie required' if needs_selfie else ''}"
)

run_clicked = st.button(
    "Run Lab",
    type="primary",
    disabled=not selected or (needs_selfie and uploaded is None),
)


# ------------------------------ Streaming runner ------------------------------


async def _stream_into_placeholders(
    *,
    runner: LabRunner,
    cells: list[StrategyCell],
    base_request: GenerationRequest,
    selfie_bytes: bytes | None,
    placeholders: dict[StrategyCell, Any],
) -> list[CellOutcome]:
    """Consume the LabRunner's async iterator and update each cell's placeholder
    as outcomes arrive. Successful outcomes are persisted to DuckDB so they
    appear in Gallery and Experiments. Returns the full list of outcomes for
    summary rendering after the loop ends.
    """
    db = get_database()
    repo = RunsRepository(db)
    outcomes: list[CellOutcome] = []

    async for outcome in runner.run(
        cells=cells, base_request=base_request, selfie_bytes=selfie_bytes
    ):
        ph = placeholders[outcome.cell]
        with ph.container(border=True):
            st.caption(
                f"**{outcome.cell.strategy.display_name}** × {outcome.cell.backbone.display_name}"
            )
            if outcome.result is not None:
                st.image(str(outcome.result.image_path), use_container_width=True)
                st.caption(
                    f"{outcome.elapsed_seconds:.1f}s · ${outcome.result.cost_usd:.3f} · "
                    f"{outcome.result.backend.value}"
                )
                repo.insert(RunRecord.from_result(outcome.result))
            else:
                st.error(f"Failed: {outcome.error}", icon="⚠️")
                st.caption(f"{outcome.elapsed_seconds:.1f}s")
        outcomes.append(outcome)

    return outcomes


# ------------------------------ Main run ------------------------------

if run_clicked:
    registry = get_registry()
    if not registry.available:
        st.error("No backend credentials configured. Add REPLICATE_API_TOKEN or FAL_KEY to .env.")
        st.stop()

    db = get_database()
    storage = LocalStorage()

    # Persist selfie (if needed) and capture its hash for the request.
    selfie_bytes: bytes | None = None
    selfie_sha: str | None = None
    if uploaded is not None:
        selfie_bytes = uploaded.getvalue()
        pil = Image.open(io.BytesIO(selfie_bytes))
        selfie = persist_selfie(data=selfie_bytes, pil_image=pil, storage=storage, db=db)
        selfie_sha = selfie.sha256

    # Base request — strategy/backbone get overridden per cell by LabRunner.
    base_request = GenerationRequest(
        strategy=Strategy.PROMPT_ONLY,
        backbone=Backbone.SDXL,
        prompt=prompt,
        selfie_sha256=selfie_sha,
        seed=int(seed),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        identity_strength=float(identity_strength),
    )

    st.markdown("## Results")

    # Pre-render placeholder grid: 3 cells per row.
    cols_per_row = 3
    placeholders: dict[StrategyCell, Any] = {}
    row: list = []
    for i, cell in enumerate(selected):
        if i % cols_per_row == 0:
            row = list(st.columns(cols_per_row))
        col = row[i % cols_per_row]
        with col:
            ph = st.empty()
            with ph.container(border=True):
                st.caption(
                    f"**{cell.strategy.display_name}** × {cell.backbone.display_name}"
                )
                st.info("Queued…")
            placeholders[cell] = ph

    runner = LabRunner(registry, max_concurrent=int(max_concurrent))

    with st.spinner(f"Running {len(selected)} cells in parallel…"):
        outcomes = asyncio.run(
            _stream_into_placeholders(
                runner=runner,
                cells=selected,
                base_request=base_request,
                selfie_bytes=selfie_bytes,
                placeholders=placeholders,
            )
        )

    # Summary row
    successful = [o for o in outcomes if o.result is not None]
    total_cost = sum(o.result.cost_usd for o in successful if o.result is not None)
    wall_time = max((o.elapsed_seconds for o in outcomes), default=0.0)

    st.markdown("---")
    cols_summary = st.columns(4)
    cols_summary[0].metric("Cells", f"{len(successful)}/{len(outcomes)}")
    cols_summary[1].metric("Total cost", f"${total_cost:.3f}")
    cols_summary[2].metric("Wall time", f"{wall_time:.1f}s")
    cols_summary[3].metric(
        "Avg cell time",
        f"{(sum(o.elapsed_seconds for o in outcomes) / len(outcomes)):.1f}s" if outcomes else "—",
    )

    # ------------------------------ Post-batch eval ------------------------------
    if successful:
        st.markdown("### Metrics")
        eval_progress = st.progress(0.0, text="Computing identity / prompt / aesthetic scores…")
        scores_by_cell: dict[str, dict[str, float | None]] = {}

        for i, outcome in enumerate(successful):
            assert outcome.result is not None  # for type narrowing
            try:
                scores = evaluate_run(run_id=str(outcome.result.run_id), db=db)
            except ImportError as e:
                st.warning(
                    f"Eval models unavailable: {e}. Run `make install-eval` for metrics.",
                    icon="ℹ️",
                )
                scores_by_cell.clear()
                break
            except Exception as e:
                st.caption(f"Eval failed for {outcome.cell.strategy.display_name}: {e}")
                scores = {}
            scores_by_cell[outcome.cell.strategy.display_name + " × " + outcome.cell.backbone.display_name] = scores
            eval_progress.progress((i + 1) / len(successful))

        eval_progress.empty()

        if scores_by_cell:
            # Identity bar chart — the headline number per cell.
            chart_data = {
                cell_name: scores.get("identity_arcface") or 0.0
                for cell_name, scores in scores_by_cell.items()
            }
            if any(v > 0 for v in chart_data.values()):
                st.bar_chart(chart_data, x_label="Cell", y_label="Identity (ArcFace cosine)")

            # Per-cell scores table
            st.dataframe(
                [
                    {
                        "cell": name,
                        "identity": s.get("identity_arcface"),
                        "prompt": s.get("prompt_siglip"),
                        "aesthetic": s.get("aesthetic_laion"),
                    }
                    for name, s in scores_by_cell.items()
                ],
                hide_index=True,
                use_container_width=True,
            )
