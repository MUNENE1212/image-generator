"""Session-state bootstrap + shared sidebar.

`bootstrap_session()` is called at the top of every page. It is idempotent.
"""

from __future__ import annotations

import streamlit as st

from image_generator.backends.registry import get_registry
from image_generator.config import settings
from image_generator.db.connection import get_database
from image_generator.logging import configure_logging


def bootstrap_session() -> None:
    """Initialize logging, DB, backend registry, and shared session_state keys."""
    if st.session_state.get("_bootstrapped"):
        _render_sidebar()
        return

    configure_logging()
    settings.ensure_dirs()
    get_database()
    get_registry()

    st.session_state.setdefault("selfie_bytes", None)
    st.session_state.setdefault("selfie_sha256", None)
    st.session_state.setdefault("last_run_id", None)
    st.session_state["_bootstrapped"] = True

    _render_sidebar()


def _render_sidebar() -> None:
    """Global sidebar: quickstart + backend status + spend display."""
    registry = get_registry()
    with st.sidebar:
        with st.expander("First time? Start here", expanded=False):
            st.markdown(
                """
                1. **Home** — upload a selfie, pick a preset, click Generate.
                2. **Strategy Lab** — same selfie + prompt, but compare 6 strategies side-by-side.
                3. **Gallery** — every image you've made, with full params + download.

                **Researchers:** the **Experiments** page sweeps hyperparameters
                and exports CSV. **About** has a researcher-tailored explainer.
                """
            )

        st.markdown("### Backends")
        if not registry.available:
            st.warning("No backend credentials. Add REPLICATE_API_TOKEN or FAL_KEY to .env.")
        else:
            for name in registry.available:
                st.caption(f"✓ {name.value}")

        st.markdown("---")
        st.toggle("Advanced mode", key="advanced_mode", value=False)
        st.caption(f"Daily spend cap: ${settings.imggen_daily_spend_cap_usd:.2f}")
