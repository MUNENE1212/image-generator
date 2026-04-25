# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Streamlit app for identity-preserving portrait generation, built around a **research-grade comparison surface** (the Strategy Lab) where one selfie + one prompt is run across N (strategy × backbone) cells in parallel and scored on automated metrics. Six pages: Home · Gallery · Strategy Lab · Training Studio · Experiments · About.

The project is in **Phase 1**: the data contract, DB layer, async backend protocol, Lab runner, eval harness, and full Streamlit page skeleton are written and parse-clean. Backend `generate()` / `train_lora()` methods and all `eval/*` metric `compute()` methods raise `NotImplementedError` — wiring them is Phase 2. Don't be surprised by stubs; they are deliberate.

## Common commands

All targets live in the Makefile; they wrap `uv`.

```
make install         # core + dev deps (fast — no torch/insightface/lpips)
make install-eval    # adds the heavy ML deps for the real eval harness
make dev             # streamlit run app/Home.py
make test            # full pytest suite
make test-fast       # skip @pytest.mark.slow and @pytest.mark.integration
make check           # ruff + mypy + test-fast
make lint / format / typecheck
```

Single test: `uv run pytest tests/test_models.py::TestGenerationRequest::test_lora_requires_name -v`

## Architecture — the big picture

### The research axis is `(Strategy, Backbone)`, not strategy alone

`Strategy` (PROMPT_ONLY, IP_ADAPTER_FACEID, INSTANT_ID, PHOTOMAKER, PULID, LORA) and `Backbone` (SDXL, FLUX_DEV) are independent. The `strategies.catalog.CATALOG` is the source of truth for which combinations exist; pages read from it rather than hardcoding lists. When adding a new strategy or backbone, **add cells to `CATALOG` first**, then ensure at least one backend's `_MODELS` table maps the new pair.

### Data contract is the spine

Everything routes through `models/`:
- `GenerationRequest` (frozen Pydantic) is the UI ↔ backend contract and serializes 1-to-1 into the `runs` table via `RunRecord.from_result()`. Cross-field validity (e.g. INSTANT_ID requires a selfie, LORA requires `lora_name`, dims must be /8) is checked by **`request.validate_request_consistency()`**, called explicitly — not by a Pydantic `field_validator`, which can't see sibling fields cleanly in v2.
- `EvalMetrics` fields are all `float | None`; the harness fills them in async, so a fresh row has metric columns NULL until evaluated.
- `RunRecord` flattens request + result + metrics into one row so DuckDB analytics queries (`GROUP BY strategy`, `WHERE guidance_scale > 5`) need no joins.

### `ComputeBackend` is an async Protocol with a registry

Two adapters: `backends/replicate.py` and `backends/fal.py`. Both carry a `_MODELS` dict pinning `(strategy, backbone) -> model_version`; **update the version string here, not at call sites**.

`BackendRegistry.resolve(strategy, backbone, preferred=None)` picks a backend with priority: `preferred` > `settings.imggen_default_backend` > first-supporting. Backends without credentials are silently dropped from the registry — pages should check `registry.available` to decide what UI to show.

### Lab runner: per-cell deterministic seeds

`LabRunner.run()` uses `asyncio.as_completed` to stream cell outcomes back to the UI as they finish. Critically, **`_derive_seed(base_seed, strategy, backbone)`** uses BLAKE2b to compute a per-cell seed that is reproducible across runs but *different per cell*. Naively reusing the same integer seed across backbones is a footgun — each backbone consumes initial noise differently, so "shared seed" doesn't mean "shared noise." Don't strip this and pass `base.seed` directly.

### Eval harness: protocol + applicability + lazy load

`Metric` (in `eval/base.py`) implementations are sync — orchestrate from async via `anyio.to_thread.run_sync`. `EvalHarness.evaluate()` skips `applicable=False` metrics (returns score=None), lazy-`load()`s on first use, and never lets a metric exception fail the run — it logs and returns `score=None` instead. Identity metrics need a selfie; LPIPS diversity needs ≥2 sibling images. The selfie face-embedding cache lives in DuckDB (`face_embeddings` table) keyed by selfie SHA-256 — recomputing it per call is wasteful, always go through `SelfiesRepository.get_embedding()` first.

**v1 metric stack (shipped):** ArcFace (identity, InsightFace `buffalo_l`), SigLIP (prompt adherence, `ViT-SO400M-14-SigLIP-384` via open_clip), LAION (aesthetic, CLIP ViT-L/14 + small MLP), LPIPS (diversity, AlexNet backbone). **Deliberately deferred:** AdaFace and Q-Align — both have stub classes that raise `NotImplementedError`, which the harness catches and returns NULL for. Activate by replacing the `compute()` body when needed.

**Imports are deliberately lazy.** Each metric's `compute()` does its `import torch / insightface / open_clip / lpips` at call time, not module top. This is so importing `image_generator.eval.identity` works even without the `eval` extra installed — the eval modules can be referenced from anywhere (registries, tests, harness construction) and only the actual scoring requires the heavy install. If you add a new metric, follow the same pattern.

**JIT eval, not pre-compute.** The orchestration service `services/eval.py::evaluate_run` is invoked by UI pages *after* the image is rendered (Home, Strategy Lab, Experiments) or *on demand* (Gallery's "Compute metrics" button), never inside `services/generation.py::generate_and_log` — generation must show the image immediately and not wait on ~5s of CPU eval.

### Storage is content-addressed

`LocalStorage.put_selfie()` shards under `data/selfies/<aa>/<sha256>.png` (first 2 hex chars) to avoid giant directories. Outputs use `data/outputs/<run_id>.png`. The `Storage` Protocol exists so an `S3Storage` swap is a one-file change later — don't bypass it with raw `Path.write_bytes` calls in feature code.

### DuckDB, not SQLite

Chosen for the Experiments page's analytical workload. `Database` is a process-global singleton via `get_database()`; **tests should construct `Database(":memory:")` directly** (see `tests/conftest.py`). Schema lives in `db/schema.sql` and is applied idempotently on every connect via `IF NOT EXISTS`. All writes take `db.lock` (an RLock) — DuckDB is single-writer per file.

### Streamlit pages: bootstrap pattern

Every page (`app/Home.py`, `app/pages/*.py`) calls `bootstrap_session()` at the top. It is idempotent: configures logging, ensures dirs, instantiates DB + registry singletons, and renders the global sidebar (backend status, advanced toggle, spend cap). Page-specific state goes in `st.session_state` with the page name as a key prefix.

## Conventions worth knowing

- **Heavy ML deps are gated.** `torch`, `insightface`, `onnxruntime`, `open-clip-torch`, `lpips`, `transformers` live in the `eval` extra. Core install (`make install`) stays fast; CI and UI-only work don't pull a CUDA-ready torch. When implementing real metric `compute()` methods, import inside the method or behind `if TYPE_CHECKING` so `make test-fast` works without the extra.

- **Frozen Pydantic models everywhere.** Mutate via `request.model_copy(update={...})`. Tests rely on this immutability.

- **No `field_validator` for cross-field rules.** Use a separate method (`validate_request_consistency`) called explicitly. This keeps `model_validate` cheap and side-effect-free.

- **Strategy/Backbone/BackendName are `StrEnum`.** They round-trip through DuckDB and JSON without custom adapters; `enum.value` is the canonical string everywhere.

- **Per-file ignores in ruff.toml.** Streamlit pages use numeric prefixes (`2_Gallery.py`) so `N999` is silenced under `app/`. Don't generalize that ignore.

## Phase 2 wiring checklist (what's stubbed → what to fill)

| File | Replace `NotImplementedError` with |
|---|---|
| `backends/replicate.py::generate` | `await replicate.async_run(model_version, input=...)`; download URL → `LocalStorage.put_output`; build `GenerationResult` |
| `backends/replicate.py::train_lora` | `replicate.trainings.create(...)`; persist job id |
| `backends/fal.py::generate` | `await fal_client.submit_async(model, arguments=...)`; mirror Replicate path |
| `eval/identity.py` | `insightface.app.FaceAnalysis("buffalo_l")` for ArcFace; cosine vs cached selfie embedding from `SelfiesRepository` |
| `eval/prompt.py` | `open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384", pretrained="webli")` |
| `eval/aesthetic.py` | LAION aesthetic predictor (small MLP on CLIP features); Q-Align via transformers pipeline |
| `eval/diversity.py` | `lpips.LPIPS(net="alex")` mean over `ctx.siblings` pairs |
| Each Streamlit page | The `# TODO(Phase 2):` markers identify exact wiring points |
