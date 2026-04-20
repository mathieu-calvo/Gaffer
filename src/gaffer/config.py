"""Runtime configuration loaded from environment via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings. Override via environment variables prefixed `GAFFER_`."""

    model_config = SettingsConfigDict(
        env_prefix="GAFFER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_source: Literal["live", "csv"] = "live"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    bootstrap_ttl_hours: int = 6
    fixtures_ttl_hours: int = 24

    horizon: int = Field(default=3, ge=1, le=5)
    solver_time_limit: int = Field(default=60, ge=5)
    ewma_alpha: float = Field(default=0.95, gt=0.0, le=1.0)

    historical_csv: Path = (
        PROJECT_ROOT / "data" / "historical" / "clean_merged_gwdf_2016_to_2024.csv"
    )


settings = Settings()
