"""
Centralised config & secrets loader.

Reads YAML from `config/*.yaml` and merges with env vars (`.env`).
All other modules import `settings` and `providers` from here rather than
touching the filesystem directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# dotenv is optional at runtime — it only loads a packaged .env file. Env vars
# set directly (CI, systemd, etc.) work without it.
try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except ImportError:  # pragma: no cover — graceful degradation
    load_dotenv = None
    _HAS_DOTENV = False

# Project root = parent of the `src` package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow override via env (e.g. for tests); otherwise use packaged .env.
_ENV_PATH = Path(os.environ.get("PROJECT_ROOT", "")) / ".env"
if not _ENV_PATH.exists():
    _ENV_PATH = PROJECT_ROOT / ".env"
if _HAS_DOTENV and _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / os.environ.get("DATA_DIR", "data")


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    data_dir: Path
    budget_cap_usd: float
    whisper_model: str

    # Convenience nested accessors
    video: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    publishing: dict[str, Any] = field(default_factory=dict)
    schedule: dict[str, Any] = field(default_factory=dict)

    def env(self, key: str, default: str = "") -> str:
        return os.environ.get(key, default)

    def env_required(self, key: str) -> str:
        val = os.environ.get(key)
        if not val:
            raise RuntimeError(
                f"Required env var {key!r} is missing. "
                f"Copy .env.example to .env and fill it in."
            )
        return val


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    raw = _load_yaml("settings.yaml")
    return Settings(
        raw=raw,
        data_dir=Path(os.environ.get("DATA_DIR", raw.get("project", {}).get("data_dir", "data"))),
        budget_cap_usd=float(os.environ.get("BUDGET_CAP", raw.get("budget", {}).get("monthly_cap_usd", 5.0))),
        whisper_model=os.environ.get("WHISPER_MODEL", raw.get("whisper_model", "base")),
        video=raw.get("video", {}),
        llm=raw.get("llm", {}),
        publishing=raw.get("publishing", {}),
        schedule=raw.get("schedule", {}),
    )


@lru_cache(maxsize=1)
def get_providers_config() -> dict[str, Any]:
    """Return the parsed providers.yaml (order + per-provider metadata)."""
    return _load_yaml("providers.yaml")


def db_path() -> Path:
    """Resolve the SQLite database path, creating the parent dir if needed."""
    settings = get_settings()
    # Absolute path if PROJECT_ROOT set; otherwise relative to this package.
    base = Path(os.environ.get("PROJECT_ROOT", "")) or PROJECT_ROOT
    path = base / settings.data_dir / "state.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
