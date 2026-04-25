"""Application configuration. All values come from environment or a `.env` file.

Import the singleton `settings` — do not instantiate `Settings` directly at call sites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    # Compute backend credentials (optional at boot; required for generation)
    replicate_api_token: SecretStr | None = Field(default=None)
    fal_key: SecretStr | None = Field(default=None)

    # Replicate-only: destination model to push trained LoRAs to (e.g. "alice/my-loras").
    # You must create this model on replicate.com first.
    replicate_destination: SecretStr | None = Field(default=None)

    # Backend selection
    imggen_default_backend: Literal["replicate", "fal"] = Field(default="replicate")

    # Abuse / cost controls
    imggen_daily_spend_cap_usd: float = Field(default=5.00, ge=0.0)

    # Paths
    imggen_data_dir: Path = Field(default=Path("data"))
    imggen_db_path: Path = Field(default=Path("data/runs/runs.duckdb"))

    # Logging
    imggen_log_format: Literal["console", "json"] = Field(default="console")
    imggen_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    @property
    def selfies_dir(self) -> Path:
        return self.imggen_data_dir / "selfies"

    @property
    def outputs_dir(self) -> Path:
        return self.imggen_data_dir / "outputs"

    @property
    def loras_dir(self) -> Path:
        return self.imggen_data_dir / "loras"

    def ensure_dirs(self) -> None:
        for path in (self.selfies_dir, self.outputs_dir, self.loras_dir, self.imggen_db_path.parent):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
