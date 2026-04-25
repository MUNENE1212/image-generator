"""About — explains the project for visitors, researchers, and engineers.

Stakeholder-tailored sections so the same page reads naturally to a casual user
("how do I make an image?"), a researcher ("what's measurable here?"), and an
engineer/recruiter ("how is it built?").
"""

from __future__ import annotations

import streamlit as st

from image_generator.strategies.catalog import CATALOG
from image_generator.ui.state import bootstrap_session

st.set_page_config(page_title="About", page_icon="📖", layout="wide")
bootstrap_session()

st.title("About this project")
st.caption(
    "An identity-preserving portrait generator that doubles as a research surface "
    "for comparing fine-tuning strategies on the same subject."
)

# ------------------------------ Audience picker ------------------------------
audience = st.radio(
    "Who are you?",
    ["Visitor — I just want to make an image", "Researcher — I want to compare approaches", "Engineer — I want to see how it's built"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("---")

# ------------------------------ Visitor ------------------------------
if audience.startswith("Visitor"):
    st.markdown(
        """
        ## Make an image in three clicks

        1. Go to **Home** in the sidebar.
        2. Upload a selfie, pick a preset (Anime Coder, Graduation, Headshot, …).
        3. Click **Generate**.

        That's it. The default settings use **InstantID on SDXL**, which is the
        sweet spot for "looks like you, in a different scene." If you want a
        higher-quality result and don't mind waiting a bit longer (~30s vs ~10s),
        open the **Advanced** expander on Home and pick **PuLID** with the
        **FLUX.1 [dev]** backbone.

        ### What's a "strategy"?

        It's a method of teaching the image model what *you* look like. Some
        strategies look at your selfie once at generation time (fast, no setup);
        others train a small adapter on 10–20 of your selfies first (slower
        setup, better results). The **Strategy Lab** lets you compare them
        side-by-side on the same prompt.

        ### Where do my images go?

        Locally on this machine, in `data/outputs/`. Every generation is also
        logged in **Gallery** — you can scroll through your history, re-download
        any image, or open the "Details" expander to see exactly which
        parameters made it.
        """
    )
    st.info("**Tip:** the same selfie + same prompt + same seed produces the same image every time. Keep your favourite seeds noted.", icon="💡")

# ------------------------------ Researcher ------------------------------
elif audience.startswith("Researcher"):
    st.markdown(
        """
        ## What's measurable here

        Every generation is automatically scored on four metrics, persisted to a
        DuckDB table (`runs.duckdb`), and exportable as CSV.

        | Metric | Model | What it tells you |
        |---|---|---|
        | **Identity** | ArcFace (InsightFace `buffalo_l`) | Cosine similarity between selfie and output face. ≥ 0.4 typically reads as "same person". |
        | **Prompt adherence** | SigLIP-SO400M-14 (webli) | Image-text alignment. Comparable across cells of the same prompt. |
        | **Aesthetic** | LAION predictor (CLIP-L/14 + MLP) | 1–10 score, biased toward 2022-era LAION aesthetics — read as a signal, not ground truth. |
        | **Diversity** | LPIPS (AlexNet) | Mean perceptual distance across sibling images. Catches mode collapse on identity-preserving strategies. |

        AdaFace and Q-Align are scaffolded as `Metric` stubs that return NULL —
        activate them by replacing the `compute()` method when you need a
        cross-check.

        ## The research axis

        The interesting comparison is **`(Strategy × Backbone)`**, not strategy
        alone. Two backbones × six strategies = an honest matrix where you can
        see, e.g., that PuLID-on-FLUX beats InstantID-on-SDXL at identity but
        costs 6× more per image. The **Strategy Lab** runs N cells of this
        matrix in parallel on the same selfie + prompt; the **Experiments**
        page sweeps hyperparameters within one fixed cell.

        ## Reproducibility

        Every cell uses a deterministic per-cell seed (BLAKE2b of
        `base_seed | strategy | backbone`) so cross-backbone comparisons aren't
        accidentally seeded with the same noise. The `runs` table stores
        every parameter that went in, and the CSV export reproduces any
        sweep with one SQL query.

        ## To extend

        - **Add a strategy or backbone:** add a `StrategyCell` to
          `src/image_generator/strategies/catalog.py`, then add the
          `(strategy, backbone) -> model_id` row to either backend's `_MODELS`
          dict (`backends/replicate.py` or `backends/fal.py`).
        - **Add a metric:** subclass `Metric` in `src/image_generator/eval/`,
          add it to the harness in `services/eval.py::get_harness`. The harness
          handles applicability, lazy-load, and never lets a metric crash the run.
        """
    )

# ------------------------------ Engineer ------------------------------
else:
    st.markdown(
        """
        ## Architecture

        Streamlit UI on top of an async Python backend that talks to Replicate
        and Fal.ai. **Six pages**, **one data contract** (Pydantic frozen
        models), **one DB** (DuckDB), **two interchangeable compute backends**
        behind a `ComputeBackend` Protocol.

        ```
        app/                            Streamlit pages (thin — UI only)
        ├── Home.py                     "Make an Image" — three-click flow
        └── pages/
            ├── 2_Gallery.py            Thumbnail grid + per-image expander
            ├── 3_Strategy_Lab.py       N cells × 1 prompt, parallel + streaming
            ├── 4_Training_Studio.py    Per-subject LoRA training (Replicate)
            ├── 5_Experiments.py        Cartesian-product sweeps + CSV export
            └── 6_About.py              You are here

        src/image_generator/
        ├── models/                     Pydantic data contract
        ├── db/                         DuckDB schema + repositories
        ├── backends/                   Replicate + Fal adapters (async Protocol)
        ├── strategies/                 Strategy × backbone catalog (data)
        ├── storage/                    Content-addressed local FS
        ├── eval/                       Metric Protocol + 4 implementations
        ├── lab/                        Strategy Lab async runner
        ├── services/                   Orchestration: generate / sweep / train / eval
        └── ui/                         Streamlit shared helpers
        ```

        ## Key engineering decisions

        - **Async-native backends.** `ComputeBackend` is an async Protocol
          (`replicate.async_run`, `fal_client.subscribe_async`); we never wrap
          sync calls in `run_in_executor`. This is what makes the Strategy Lab's
          `asyncio.as_completed` streaming actually work.
        - **DuckDB, not SQLite.** Same single-file ergonomics; 10–100× faster
          analytical queries on the Experiments page.
        - **Content-addressed storage.** Selfies are sharded by SHA-256 prefix.
          Re-uploading the same selfie is a no-op; re-evaluating a generation
          uses the cached face embedding.
        - **Lazy heavy imports.** Eval metrics import `torch`, `insightface`,
          `open_clip`, `lpips` inside `compute()`, not at module top. This
          means the eval modules can be loaded without the heavy `eval` extra
          installed; only running scoring requires it. The `make install-eval`
          extra is ~1 GB; without it, the metric badges just don't show up.
        - **Per-cell seed derivation.** BLAKE2b of `(base_seed, strategy,
          backbone)` so "shared seed" across backbones doesn't mean shared
          noise (different backbones consume initial noise differently).

        ## Tech stack

        Python 3.12 · Streamlit · Pydantic v2 · DuckDB · `replicate` + `fal-client`
        async SDKs · `insightface` + `open-clip-torch` + `lpips` for eval ·
        `structlog` · `uv` · ruff + mypy strict · pytest.

        ## Tests

        ```
        make check        # ruff + mypy + fast tests
        make test         # full pytest (54 tests)
        ```

        Live integration tests for the eval models are marked
        `@pytest.mark.integration @pytest.mark.slow` and skip without the eval
        extra installed.
        """
    )

# ------------------------------ Strategy reference (always shown) ------------------------------
st.markdown("---")
st.markdown("## Strategy reference")
st.caption("Each `(strategy × backbone)` cell available in the app, with the tradeoff that defines it.")

for cell in CATALOG:
    with st.container(border=True):
        cols = st.columns([2, 5])
        with cols[0]:
            st.markdown(f"**{cell.strategy.display_name}**")
            st.caption(cell.backbone.display_name)
            if cell.requires_training:
                st.caption("⚠️ requires training")
        with cols[1]:
            st.write(f"_{cell.tagline}_")
            st.write(cell.tradeoff)

st.markdown("---")
st.caption(
    "Built by Munene — code at github.com/<your-handle>/image-generator. "
    "Issues + pull requests welcome."
)
