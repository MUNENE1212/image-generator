# image-generator

Streamlit app that turns one selfie into identity-preserving portraits, with a
research surface for comparing fine-tuning strategies side-by-side on the same
subject — automatically scored on identity, prompt adherence, aesthetic, and
diversity.

> One line: a multi-page Streamlit app for identity-preserving image generation
> that systematically compares six strategies on a `(strategy × backbone)` matrix
> with automated metrics, deployable to HuggingFace Spaces.

## What it does

| Page | For whom | What it does |
|---|---|---|
| **Home** | First-time visitors | Upload selfie + preset → image in 3 clicks. Advanced expander reveals strategy/backbone/sliders. |
| **Gallery** | Anyone | Thumbnail grid of every generation, with full params, metrics, and per-image download. |
| **Strategy Lab** | Researchers | One selfie + one prompt × N strategies, run in parallel, auto-scored, ranked by identity. |
| **Training Studio** | Power users | Train a per-subject LoRA from 10–20 selfies via Replicate (~10 min, ~$2). |
| **Experiments** | Researchers | Sweep hyperparameters (CFG × steps × seed × identity_strength) → CSV export. |
| **About** | Stakeholders | Stakeholder-aware explainers (visitor / researcher / engineer). |

## Quick start

```bash
cp .env.example .env       # add REPLICATE_API_TOKEN and/or FAL_KEY
make install               # core + dev deps (fast, ~250MB)
make dev                   # http://localhost:8501
```

Heavy ML deps for the eval harness (ArcFace, SigLIP, LAION, LPIPS, ~1 GB of
weights) live behind an extra:

```bash
make install-eval          # only needed when you want metric badges
```

The app works without the eval extra — metrics just don't render. See
[CLAUDE.md](CLAUDE.md) for the architecture overview.

## Dev

```bash
make check       # ruff + mypy strict + fast tests (54 tests)
make test        # full pytest including any integration tests
make format      # ruff format + autofix
```

## Architecture (one minute)

- **Pydantic v2 frozen models** are the data contract — `GenerationRequest` /
  `GenerationResult` / `Selfie` / `EvalMetrics`.
- **DuckDB** persists everything (`runs.duckdb`); chosen over SQLite for fast
  analytical queries on the Experiments page.
- **`ComputeBackend`** is an async Protocol with two implementations
  (Replicate, Fal.ai) — same generate/train/health surface, different vendor SDKs.
- **`(Strategy × Backbone)` matrix** is the research axis. Catalog lives in
  `src/image_generator/strategies/catalog.py` as data; UI pages read from it.
- **`LabRunner`** runs N cells via `asyncio.as_completed`, with a
  BLAKE2b-derived per-cell seed so cross-backbone "shared seed" doesn't mean
  shared noise.
- **Eval harness** ships ArcFace + SigLIP + LAION + LPIPS as the v1 metric
  stack. AdaFace + Q-Align are scaffolded as stubs ready to activate.

Full breakdown in [CLAUDE.md](CLAUDE.md).

## Deploy

**VPS (recommended)** — `docker compose up -d` with auto-TLS via Caddy and a
named volume for persistent `runs.duckdb` + image cache. Full recipe with
provisioning, updates, backups, and a bare-metal/systemd alternative is in
[`deploy/README.md`](deploy/README.md).

Other options:

- **Streamlit Community Cloud** — works, but its filesystem is ephemeral, so
  every restart wipes your generation history. Fine for a demo, not for
  research data.
- **HuggingFace Spaces** — same ephemeral-disk caveat.

For any public deployment: set `IMGGEN_DAILY_SPEND_CAP_USD` to bound abuse,
and either gate access with HTTP basic auth (Caddy one-liner in the deploy
guide) or replace the API-token env with a sidebar BYO-key textbox so
visitors pay their own Replicate/Fal bills.

## Tech stack

Python 3.12 · Streamlit · Pydantic v2 · DuckDB · `replicate` + `fal-client`
async SDKs · `insightface` + `open-clip-torch` + `lpips` (eval extra) ·
`structlog` · uv · ruff + mypy strict · pytest.

## Status

10-day plan: complete through Day 9 (about polish). Day 10 is deploy + blog post.
The full design history and architectural decisions are in
[CLAUDE.md](CLAUDE.md).

## Further reading

- [`docs/article.md`](docs/article.md) — *From novelty to infrastructure: where
  image generation actually is in 2026* — a stakeholder-oriented field guide
  covering the state of the art, trends, challenges, and business / investment
  angles. Uses this repo as the lens.
