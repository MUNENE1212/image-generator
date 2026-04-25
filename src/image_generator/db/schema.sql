-- DuckDB schema for image-generator.
-- Executed idempotently on startup by Database._apply_schema().

CREATE TABLE IF NOT EXISTS selfies (
    sha256          VARCHAR PRIMARY KEY,
    path            VARCHAR NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    uploaded_at     TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    selfie_sha256   VARCHAR NOT NULL,
    model_name      VARCHAR NOT NULL,
    vector          FLOAT[],
    computed_at     TIMESTAMP NOT NULL,
    PRIMARY KEY (selfie_sha256, model_name)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id              VARCHAR PRIMARY KEY,
    -- Request
    strategy            VARCHAR NOT NULL,
    backbone            VARCHAR NOT NULL,
    prompt              VARCHAR NOT NULL,
    negative_prompt     VARCHAR,
    selfie_sha256       VARCHAR,
    seed                INTEGER NOT NULL,
    num_inference_steps INTEGER NOT NULL,
    guidance_scale      DOUBLE  NOT NULL,
    width               INTEGER NOT NULL,
    height              INTEGER NOT NULL,
    identity_strength   DOUBLE  NOT NULL,
    lora_name           VARCHAR,
    -- Execution
    backend             VARCHAR NOT NULL,
    model_version       VARCHAR NOT NULL,
    image_path          VARCHAR NOT NULL,
    started_at          TIMESTAMP NOT NULL,
    completed_at        TIMESTAMP NOT NULL,
    duration_seconds    DOUBLE  NOT NULL,
    cost_usd            DOUBLE  NOT NULL,
    -- Metrics (NULL until eval harness fills them in)
    identity_arcface    DOUBLE,
    identity_adaface    DOUBLE,
    prompt_siglip       DOUBLE,
    aesthetic_laion     DOUBLE,
    aesthetic_qalign    DOUBLE,
    diversity_lpips     DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_runs_strategy     ON runs(strategy);
CREATE INDEX IF NOT EXISTS idx_runs_backbone     ON runs(backbone);
CREATE INDEX IF NOT EXISTS idx_runs_selfie       ON runs(selfie_sha256);
CREATE INDEX IF NOT EXISTS idx_runs_completed_at ON runs(completed_at);

-- Sweeps: groups runs that belong to one Experiments-page experiment.
CREATE TABLE IF NOT EXISTS sweeps (
    sweep_id        VARCHAR PRIMARY KEY,
    name            VARCHAR NOT NULL,
    config_json     VARCHAR NOT NULL,      -- serialized SweepConfig
    created_at      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS sweep_runs (
    sweep_id        VARCHAR NOT NULL,
    run_id          VARCHAR NOT NULL,
    PRIMARY KEY (sweep_id, run_id)
);

-- Training jobs produced by the Training Studio.
CREATE TABLE IF NOT EXISTS trainings (
    training_id     VARCHAR PRIMARY KEY,
    method          VARCHAR NOT NULL,       -- TrainingMethod enum
    selfie_shas     VARCHAR[],              -- array of sha256s used
    lora_name       VARCHAR,                -- artifact name, set on completion
    lora_path       VARCHAR,                -- local / remote URL once built
    status          VARCHAR NOT NULL,       -- queued | running | succeeded | failed
    backend         VARCHAR NOT NULL,
    cost_usd        DOUBLE  NOT NULL DEFAULT 0.0,
    started_at      TIMESTAMP NOT NULL,
    completed_at    TIMESTAMP,
    error           VARCHAR
);
