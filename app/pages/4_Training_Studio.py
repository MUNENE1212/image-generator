"""Training Studio — train a per-subject LoRA from 10–20 selfies.

Currently Replicate-only. Requires REPLICATE_API_TOKEN + REPLICATE_DESTINATION
(a Replicate model you own where the trained LoRA gets pushed).

Status of in-flight trainings is polled when the page reruns. A 'Refresh status'
button kicks one explicit poll; users can also auto-poll via the auto-refresh
checkbox (which uses st.fragment to re-render the table every 10s).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib

import streamlit as st

from image_generator.backends.registry import get_registry
from image_generator.config import settings
from image_generator.db.connection import get_database
from image_generator.db.repository import TrainingsRepository
from image_generator.models.enums import TrainingMethod
from image_generator.services.training import poll_training, start_training
from image_generator.ui.state import bootstrap_session

st.set_page_config(page_title="Training Studio", page_icon="🎓", layout="wide")
bootstrap_session()

st.title("Training Studio")
st.caption("Upload 10–20 selfies, train a personal LoRA, then use it from any page.")

# ------------------------------ Pre-flight checks ------------------------------
registry = get_registry()
db = get_database()
repo = TrainingsRepository(db)

config_issues: list[str] = []
if settings.replicate_api_token is None:
    config_issues.append("`REPLICATE_API_TOKEN` not set — training requires Replicate.")
if settings.replicate_destination is None:
    config_issues.append(
        "`REPLICATE_DESTINATION` not set — create a model at "
        "https://replicate.com/create then set this env to `username/model-name`."
    )

if config_issues:
    for msg in config_issues:
        st.warning(msg)

# ------------------------------ Inputs ------------------------------
files = st.file_uploader(
    "Selfies (10–20 recommended)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)
n_files = len(files) if files else 0
if files and (n_files < 5 or n_files > 30):
    st.info(f"You uploaded {n_files} selfies. Sweet spot is 10–20 for LoRA training.")

lora_name = st.text_input(
    "LoRA name",
    placeholder="alex_2026",
    help="Short identifier; the trained model will be pushed to {destination}:{name}.",
)

method = st.radio(
    "Method",
    options=[TrainingMethod.LORA_SDXL, TrainingMethod.LORA_FLUX],
    format_func=lambda m: {
        "lora_sdxl": "LoRA on SDXL (~10 min, ~$2)",
        "lora_flux": "LoRA on FLUX (~20 min, ~$5)",
    }[m.value],
    horizontal=True,
)

with st.expander("Advanced", expanded=False):
    rank = st.select_slider(
        "LoRA rank",
        options=[4, 8, 16, 32],
        value=16,
        help="Adapter capacity. Higher = more expressive (and bigger file) but slower to train. 16 is the standard sweet spot.",
    )
    steps = st.slider(
        "Training steps",
        min_value=500,
        max_value=3000,
        value=1500 if method is TrainingMethod.LORA_SDXL else 1000,
        step=100,
        help="More steps = better fit but risk of overfitting. SDXL: 1000–2000; FLUX: 800–1200.",
    )
    learning_rate = st.select_slider(
        "Learning rate",
        options=[1e-5, 5e-5, 1e-4, 5e-4],
        value=1e-4,
        help="1e-4 is the standard. Lower if you're getting overfit (output looks identical to your selfies).",
    )

st.markdown("---")

train_disabled = (
    not files
    or not lora_name
    or settings.replicate_api_token is None
    or settings.replicate_destination is None
)

if st.button("Train", type="primary", disabled=train_disabled):
    selfies: list[tuple[str, bytes]] = []
    for i, f in enumerate(files or []):
        data = f.getvalue()
        sha = hashlib.sha256(data).hexdigest()[:12]
        # Keep extension so Replicate's content-type detection works.
        ext = f.name.split(".")[-1].lower() if "." in f.name else "png"
        selfies.append((f"{lora_name}_{i:02d}_{sha}.{ext}", data))

    with st.spinner("Bundling selfies and submitting to Replicate…"):
        try:
            training_id = asyncio.run(
                start_training(
                    method=method,
                    selfies=selfies,
                    lora_name=lora_name,
                    rank=int(rank),
                    steps=int(steps),
                    learning_rate=float(learning_rate),
                    registry=registry,
                    db=db,
                )
            )
        except Exception as e:
            st.error(f"Training submission failed: {e}")
            st.stop()

    st.success(f"Training submitted: `{training_id}`. Track progress below.")
    st.session_state["last_training_id"] = training_id


# ------------------------------ Status table ------------------------------
st.markdown("## Recent trainings")

auto_refresh = st.toggle(
    "Auto-refresh every 10s",
    value=False,
    help="Polls every running training. Counts toward your Replicate API rate limit.",
)


def _render_table() -> None:
    rows = repo.recent(limit=20)
    if not rows:
        st.caption("No trainings yet.")
        return

    # Active trainings get a manual refresh button + are auto-polled if toggle is on.
    running_ids = [str(r["training_id"]) for r in rows if r["status"] == "running"]

    if running_ids and st.button("Refresh status now", use_container_width=False):
        with st.spinner(f"Polling {len(running_ids)} running training(s)…"):
            for tid in running_ids:
                try:
                    asyncio.run(poll_training(training_id=tid, registry=registry, db=db))
                except Exception as e:
                    st.warning(f"Poll failed for {tid[:8]}…: {e}")
        st.rerun()

    # Render table with status badge.
    st.dataframe(
        [
            {
                "started": r["started_at"],
                "status": r["status"],
                "method": r["method"],
                "lora_name": r.get("lora_name") or "—",
                "lora_path": r.get("lora_path") or "—",
                "error": (str(r.get("error"))[:80] + "…") if r.get("error") else "",
                "training_id": str(r["training_id"])[:12] + "…",
            }
            for r in rows
        ],
        hide_index=True,
        use_container_width=True,
    )


if auto_refresh:
    @st.fragment(run_every=10)  # type: ignore[misc]
    def _auto_refresh_fragment() -> None:
        # Poll any running trainings, then re-render the table.
        rows = repo.recent(limit=20)
        for r in rows:
            if r["status"] == "running":
                # Auto-poll must never break the page render — broker errors,
                # rate limits, transient failures all swallowed silently.
                with contextlib.suppress(Exception):
                    asyncio.run(
                        poll_training(
                            training_id=str(r["training_id"]), registry=registry, db=db
                        )
                    )
        _render_table()

    _auto_refresh_fragment()
else:
    _render_table()


# ------------------------------ Available LoRAs ------------------------------
st.markdown("## Available LoRAs")
succeeded = repo.succeeded()
if succeeded:
    st.caption(
        f"{len(succeeded)} trained LoRA(s) available. Select one in **Home** or "
        "**Strategy Lab** under strategy=`lora`."
    )
    st.dataframe(
        [
            {
                "lora_name": r.get("lora_name") or "—",
                "lora_path": r["lora_path"],
                "method": r["method"],
                "completed": r["completed_at"],
            }
            for r in succeeded
        ],
        hide_index=True,
        use_container_width=True,
    )
else:
    st.caption("No completed LoRAs yet. Train one above.")
