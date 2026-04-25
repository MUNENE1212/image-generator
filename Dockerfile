# Multi-stage build:
#   stage 1 — uv-based dep install (large, throwaway)
#   stage 2 — slim runtime that only carries the venv + app source

ARG PYTHON_VERSION=3.12

# ---------- stage 1: build ----------
FROM python:${PYTHON_VERSION}-slim AS build

# uv: fast Python installer. Pinned for reproducible builds.
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

# Build deps for any wheels that need to compile (insightface, lpips do).
# Slim's apt is short — install what we need, then drop the lists.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the dep-defining files first so we get a layer cache hit when
# only source changes.
COPY pyproject.toml ./
COPY README.md ./
# uv needs the package source present to build the editable install,
# but for the install layer cache we want this BEFORE COPY-ing the rest.
COPY src/ ./src/

# Install core deps. To bake in the eval extra at build time, change to
# `--extra eval`. Default is core-only — eval install on a slim base is ~1.5GB,
# slow to build, and most VPS deploys won't actually run eval (compute is on
# Replicate/Fal anyway).
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen --no-dev --no-editable || uv sync --no-dev --no-editable


# ---------- stage 2: runtime ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

# Runtime libs that the wheels pulled in expect. libgl1 is for cv2 (used by
# insightface if eval extra is installed).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

# Bring in the prebuilt venv and source.
COPY --from=build --chown=app:app /opt/venv /opt/venv
COPY --from=build --chown=app:app /app /app

# Make the venv first on PATH so `streamlit` and `python` resolve to it.
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

USER app
WORKDIR /app

# Streamlit's own healthcheck endpoint — used by docker compose + Caddy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "app/Home.py"]
