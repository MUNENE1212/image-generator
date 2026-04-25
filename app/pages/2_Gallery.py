"""Gallery — every generation this user has produced, in a thumbnail grid.

Click a card → expander with full GenerationRequest params, metrics, cost,
download button. Filters at the top narrow by strategy/backbone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from image_generator.db.connection import get_database
from image_generator.db.repository import RunsRepository
from image_generator.models.enums import Backbone, Strategy
from image_generator.services.eval import evaluate_run
from image_generator.ui.state import bootstrap_session

st.set_page_config(page_title="Gallery", page_icon="🖼️", layout="wide")
bootstrap_session()

st.title("Gallery")

repo = RunsRepository(get_database())
total = repo.count()

if total == 0:
    st.info("No generations yet. Visit **Home** or **Strategy Lab** to make some.")
    st.stop()

# ------------------------------ Filters ------------------------------
with st.container(border=True):
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        strategy_filter = st.multiselect(
            "Strategy",
            options=[s.value for s in Strategy],
            default=[],
            help="Empty = all strategies",
        )
    with f2:
        backbone_filter = st.multiselect(
            "Backbone",
            options=[b.value for b in Backbone],
            default=[],
            help="Empty = all backbones",
        )
    with f3:
        limit = st.number_input("Limit", min_value=4, max_value=200, value=24, step=4)

# Load + filter in memory. Phase 2c if/when this gets slow: push filters to SQL.
runs = repo.recent(limit=int(limit))
if strategy_filter:
    runs = [r for r in runs if r["strategy"] in strategy_filter]
if backbone_filter:
    runs = [r for r in runs if r["backbone"] in backbone_filter]

st.caption(f"Showing {len(runs)} of {total} generations.")

if not runs:
    st.warning("No runs match the current filters.")
    st.stop()


def _render_card(run: dict[str, Any], col: Any) -> None:
    image_path = Path(str(run["image_path"]))
    has_metrics = run.get("identity_arcface") is not None or run.get("aesthetic_laion") is not None
    with col, st.container(border=True):
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning(f"Missing: {image_path.name}", icon="⚠️")
        st.caption(
            f"**{run['strategy']}** × {run['backbone']} · "
            f"{run['duration_seconds']:.1f}s · ${run['cost_usd']:.3f}"
        )

        # Inline metric badges when present
        if has_metrics:
            badges = []
            if run.get("identity_arcface") is not None:
                badges.append(f"id={run['identity_arcface']:.2f}")
            if run.get("prompt_siglip") is not None:
                badges.append(f"prompt={run['prompt_siglip']:.2f}")
            if run.get("aesthetic_laion") is not None:
                badges.append(f"aesthetic={run['aesthetic_laion']:.1f}")
            st.caption(" · ".join(badges))
        elif image_path.exists():
            # Lazy compute: only offered if image exists and metrics are missing.
            if st.button(
                "Compute metrics",
                key=f"eval_{run['run_id']}",
                use_container_width=True,
            ):
                with st.spinner("Running eval harness…"):
                    try:
                        evaluate_run(run_id=str(run["run_id"]), db=get_database())
                    except ImportError as e:
                        st.error(f"Eval extra not installed: {e}")
                    except Exception as e:
                        st.error(f"Eval failed: {e}")
                st.rerun()

        with st.expander("Details"):
            payload = {k: v for k, v in run.items() if k != "image_path"}
            st.code(json.dumps(payload, indent=2, default=str), language="json")
            if image_path.exists():
                st.download_button(
                    "Download image",
                    data=image_path.read_bytes(),
                    file_name=image_path.name,
                    mime="image/png",
                    key=f"dl_{run['run_id']}",
                    use_container_width=True,
                )


# ------------------------------ Grid ------------------------------
COLS_PER_ROW = 3
for i in range(0, len(runs), COLS_PER_ROW):
    row = st.columns(COLS_PER_ROW)
    for j, run in enumerate(runs[i : i + COLS_PER_ROW]):
        _render_card(run, row[j])
