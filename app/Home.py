"""Home — 'Make an Image'. Three-click flow for casual visitors.

Progressive disclosure: Advanced expander shows sliders, strategy/backbone pickers.
All orchestration happens in `services.generation`; this page only renders UI.
"""

from __future__ import annotations

import asyncio
import io

import streamlit as st
from PIL import Image

from image_generator.backends.base import BackendError
from image_generator.backends.registry import get_registry
from image_generator.db.connection import get_database
from image_generator.models.enums import Backbone, Strategy
from image_generator.models.requests import GenerationRequest
from image_generator.services.eval import evaluate_run
from image_generator.services.generation import generate_and_log, persist_selfie
from image_generator.storage.local import LocalStorage
from image_generator.strategies.catalog import CATALOG
from image_generator.ui.state import bootstrap_session

st.set_page_config(page_title="Image Generator", page_icon="🖼️", layout="wide")
bootstrap_session()

PRESETS: dict[str, str] = {
    "Anime Coder": "portrait of a person as an anime character, sitting at a computer, vibrant colors, studio ghibli style",
    "Graduation": "portrait of a person in graduation robes and cap, proud expression, campus background, natural light",
    "Headshot": "professional headshot of a person, neutral background, soft studio lighting, 85mm lens, sharp focus",
    "Fantasy Warrior": "epic portrait of a person as a fantasy warrior, ornate armor, dramatic lighting, cinematic, concept art",
    "Street Photography": "candid street photograph of a person, 35mm film, golden hour, shallow depth of field",
}

st.title("Make an Image")
st.caption("Upload a selfie, pick a preset, click Generate.")

col_input, col_output = st.columns([1, 1])

# ------------------------------ Input column ------------------------------
with col_input:
    uploaded = st.file_uploader("Selfie", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    if uploaded is not None:
        st.image(uploaded, caption="Selfie preview", width=240)

    preset_name = st.selectbox("Preset", list(PRESETS.keys()))
    prompt = st.text_area("Prompt", value=PRESETS[preset_name], height=100)

    with st.expander("Advanced", expanded=False):
        strategy = st.selectbox(
            "Strategy",
            options=list(Strategy),
            format_func=lambda s: s.display_name,
            index=list(Strategy).index(Strategy.INSTANT_ID),
        )
        backbone = st.selectbox(
            "Backbone",
            options=list(Backbone),
            format_func=lambda b: b.display_name,
        )
        c1, c2 = st.columns(2)
        with c1:
            steps = st.slider(
                "Steps",
                10,
                80,
                30,
                help="More steps = sharper detail but slower. 25–35 is the sweet spot for SDXL/FLUX.",
            )
            guidance = st.slider(
                "Guidance",
                1.0,
                15.0,
                5.0,
                help="How strongly the model adheres to the prompt. Too high (>10) → over-saturated; too low (<2) → loose.",
            )
        with c2:
            identity_strength = st.slider(
                "Identity strength",
                0.0,
                1.5,
                0.8,
                help="How much the selfie influences the output. 0 = ignore selfie; 1 = strong match; >1 = forced (often weird).",
            )
            seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=2**31 - 1,
                value=0,
                help="Same selfie + prompt + seed always produces the same image. Useful for replication.",
            )

    generate_clicked = st.button(
        "Generate",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None and strategy is not Strategy.PROMPT_ONLY,
    )

# ------------------------------ Output column ------------------------------
with col_output:
    st.subheader("Result")

    if generate_clicked:
        registry = get_registry()
        if not registry.available:
            st.error("No backend credentials configured. Add REPLICATE_API_TOKEN or FAL_KEY to .env.")
        else:
            db = get_database()
            storage = LocalStorage()

            # Persist selfie (if provided) and get the SHA-256 the request needs.
            selfie_sha: str | None = None
            selfie_bytes: bytes | None = None
            if uploaded is not None:
                selfie_bytes = uploaded.getvalue()
                pil = Image.open(io.BytesIO(selfie_bytes))
                selfie = persist_selfie(data=selfie_bytes, pil_image=pil, storage=storage, db=db)
                selfie_sha = selfie.sha256
                st.session_state["selfie_sha256"] = selfie_sha

            try:
                request = GenerationRequest(
                    strategy=strategy,
                    backbone=backbone,
                    prompt=prompt,
                    selfie_sha256=selfie_sha,
                    seed=int(seed),
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    identity_strength=float(identity_strength),
                )
            except ValueError as e:
                st.error(f"Invalid request: {e}")
                st.stop()

            with st.spinner(f"Generating with {strategy.display_name} on {backbone.display_name}…"):
                try:
                    result = asyncio.run(
                        generate_and_log(
                            request=request, selfie_bytes=selfie_bytes, registry=registry, db=db
                        )
                    )
                except BackendError as e:
                    st.error(f"Generation failed: {e}")
                    st.stop()

            st.session_state["last_run_id"] = str(result.run_id)
            st.image(str(result.image_path), caption=prompt, use_container_width=True)
            st.caption(
                f"Strategy: **{request.strategy.display_name}** · "
                f"Backbone: **{request.backbone.display_name}** · "
                f"Backend: **{result.backend.value}** · "
                f"{result.duration_seconds:.1f}s · ${result.cost_usd:.3f}"
            )

            # Eval — show metrics when available; silently degrades when not.
            with st.spinner("Computing metrics…"):
                try:
                    scores = evaluate_run(run_id=str(result.run_id), db=db)
                except ImportError:
                    scores = {}  # eval extra not installed — skip silently
                except Exception:
                    scores = {}

            if scores:
                cols_metrics = st.columns(3)
                if scores.get("identity_arcface") is not None:
                    cols_metrics[0].metric("Identity (ArcFace)", f"{scores['identity_arcface']:.2f}")
                if scores.get("prompt_siglip") is not None:
                    cols_metrics[1].metric("Prompt (SigLIP)", f"{scores['prompt_siglip']:.2f}")
                if scores.get("aesthetic_laion") is not None:
                    cols_metrics[2].metric("Aesthetic (LAION)", f"{scores['aesthetic_laion']:.1f}")
    else:
        st.caption("Click Generate to create an image.")

st.markdown("---")
st.caption(f"{len(CATALOG)} strategy × backbone cells available. Visit **Strategy Lab** to compare.")
