"""Pytest fixtures shared across the test suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `src` importable when running from the project root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Automatically strip potentially sensitive or polluting host environment variables."""
    vars_to_clear = [
        "RUNPOD_API_KEY",
        "RUNPOD_POD_ID",
        "RUNPOD_NETWORK_VOLUME_ID",
        "GEMINI_API_KEY",
        "JAMENDO_CLIENT_ID",
        "FREESOUND_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "WAN_MODEL_ID"
    ]
    for var in vars_to_clear:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    """Point PROJECT_ROOT + DATA_DIR at a temp dir so tests never touch real state."""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    # Clear the lru_cache so config picks up the new env.
    from src import config
    config.get_settings.cache_clear()
    config.get_providers_config.cache_clear()
    yield tmp_path
