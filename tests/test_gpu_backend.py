"""GPU backend switch: runpod (local.py, pristine) vs colab (colab.py, separate).

Contract under test:
  * local.py RunPod flow is byte-identical logic to v1.1.0 (no shared code
    with the colab path).
  * ColabGenerator resolves its tunnel URL (explicit → env → settings),
    validates it, and drives ready → submit → poll → download on its own.
  * The pool factory routes "local"→RunPod and "colab"→Colab, and the
    unselected family always reports unconfigured.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.config import get_settings
from src.dashboard.accounts import AccountStore
from src.generators.colab import (
    ColabGenerator,
    resolve_colab_url,
    selected_backend,
)
from src.generators.local import LocalGenerator
from src.store import Store


def _set_gpu(tmp_env, **overrides):
    store = Store(tmp_env / "data" / "state.db")
    for k, v in overrides.items():
        store.set_override(k, json.dumps(v))
    store.close()
    from src import config
    config.get_settings.cache_clear()
    return get_settings()


class TestBackendSelection:
    def test_default_is_runpod(self, tmp_env):
        assert selected_backend() == "runpod"

    def test_settings_switch(self, tmp_env):
        _set_gpu(tmp_env, **{"gpu.backend": "colab"})
        assert selected_backend() == "colab"

    def test_local_steps_out_in_colab_mode(self, tmp_env, monkeypatch):
        # Even with RunPod keys present, the local provider must not claim
        # the chain when colab is selected.
        monkeypatch.setenv("RUNPOD_API_KEY", "k")
        monkeypatch.setenv("RUNPOD_POD_ID", "pod")
        _set_gpu(tmp_env, **{"gpu.backend": "colab"})
        assert LocalGenerator(server_url="http://x").is_configured is False

    def test_local_unchanged_in_runpod_mode(self, tmp_env, monkeypatch):
        monkeypatch.setenv("RUNPOD_API_KEY", "k")
        monkeypatch.setenv("RUNPOD_POD_ID", "pod")
        assert LocalGenerator(server_url="").is_configured is True


class TestColabUrl:
    def test_explicit_env_settings_precedence(self, tmp_env, monkeypatch):
        _set_gpu(tmp_env, **{"gpu.colab_url": "https://file.trycloudflare.com"})
        monkeypatch.setenv("COLAB_TUNNEL_URL", "https://env.trycloudflare.com/")
        assert resolve_colab_url() == "https://env.trycloudflare.com"
        assert resolve_colab_url("https://arg.trycloudflare.com") == "https://arg.trycloudflare.com"
        monkeypatch.delenv("COLAB_TUNNEL_URL")
        assert resolve_colab_url() == "https://file.trycloudflare.com"

    def test_generator_uses_explicit_url(self, tmp_env):
        gen = ColabGenerator(tunnel_url="https://abc.trycloudflare.com/")
        assert gen.tunnel_url == "https://abc.trycloudflare.com"
        assert gen.is_configured is True
        assert ColabGenerator().is_configured is False

    def test_run_rejects_missing_and_bad_url(self, tmp_env):
        _set_gpu(tmp_env, **{"gpu.backend": "colab", "gpu.colab_url": ""})
        import pytest
        with pytest.raises(ValueError, match="no tunnel URL"):
            ColabGenerator()._run("prompt here", 5, Path("/tmp/x.mp4"))
        _set_gpu(tmp_env, **{"gpu.backend": "colab", "gpu.colab_url": "not-a-url"})
        with pytest.raises(ValueError, match="looks invalid"):
            ColabGenerator()._run("prompt here", 5, Path("/tmp/x.mp4"))

    def test_colab_full_flow_against_mock(self, tmp_env):
        """Drive the separate colab implementation over real HTTP."""
        import subprocess
        import sys
        import time
        import httpx
        _set_gpu(tmp_env, **{
            "gpu.backend": "colab",
            "gpu.colab_url": "http://127.0.0.1:8098",
        })
        mock = subprocess.Popen(
            [sys.executable, "-c",
             "import uvicorn, sys; sys.path.insert(0, '.'); "
             "from scripts.mock_gpu_server import app; "
             "uvicorn.run(app, host='127.0.0.1', port=8098, log_level='error')"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(30):
                try:
                    if httpx.get("http://127.0.0.1:8098/ready", timeout=2).status_code == 200:
                        break
                except Exception:
                    time.sleep(1)
            out = Path("/tmp/opencode/colab_sep.mp4")
            ColabGenerator()._run("a rainy neon street chase at night", 5, out, scene_index=1)
            assert out.exists() and out.stat().st_size > 0
        finally:
            mock.terminate()


class TestFactoryRouting:
    def _pool(self, tmp_env, cfg):
        from src.generators.pool import GeneratorPool
        store = Store(tmp_env / "data" / "state.db")
        return GeneratorPool(store=store, settings=get_settings(),
                             providers_config=cfg,
                             account_store=AccountStore(store))

    def test_runpod_mode_builds_local(self, tmp_env, monkeypatch):
        monkeypatch.setenv("LOCAL_GENERATOR_URL", "http://gpu:8000")
        cfg = {"order": ["local", "colab"],
               "providers": {"local": {"enabled": True, "env_key": "LOCAL_GENERATOR_URL"},
                             "colab": {"enabled": True}}}
        pool = self._pool(tmp_env, cfg)
        names = [type(g.gen).__name__ for g in pool.chain]
        assert names == ["LocalGenerator"]  # colab unconfigured → skipped

    def test_colab_mode_builds_colab(self, tmp_env):
        _set_gpu(tmp_env, **{"gpu.backend": "colab",
                             "gpu.colab_url": "https://abc.trycloudflare.com"})
        cfg = {"order": ["local", "colab"],
               "providers": {"local": {"enabled": True},
                             "colab": {"enabled": True}}}
        pool = self._pool(tmp_env, cfg)
        names = [type(g.gen).__name__ for g in pool.chain]
        assert names == ["ColabGenerator"]  # local stepped out → skipped

    def test_colab_account_carries_tunnel_url(self, tmp_env):
        _set_gpu(tmp_env, **{"gpu.backend": "colab"})
        store = Store(tmp_env / "data" / "state.db")
        AccountStore(store).add("colab", "main", "https://acc.trycloudflare.com")
        from src.generators.pool import GeneratorPool
        pool = GeneratorPool(store=store, settings=get_settings(),
                             providers_config={
                                 "order": ["colab"],
                                 "providers": {"colab": {"enabled": True}}},
                             account_store=AccountStore(store))
        assert len(pool.chain) == 1
        assert pool.chain[0].gen.tunnel_url == "https://acc.trycloudflare.com"
        store.close()


class TestRunpodUntouchedInColabMode:
    def test_stop_skipped_for_colab(self, tmp_env, monkeypatch):
        monkeypatch.setenv("RUNPOD_API_KEY", "k")
        monkeypatch.setenv("RUNPOD_POD_ID", "pod")
        _set_gpu(tmp_env, **{"gpu.backend": "colab"})
        from src.pipeline import Pipeline
        pipe = Pipeline(dry_run=True, theme="heist_getaway")
        with patch("src.generators.runpod_manager.RunPodManager") as mgr:
            pipe._stop_runpod_if_configured()
            mgr.assert_not_called()

    def test_colab_notebook_is_valid(self):
        nb = json.loads(Path("colab/ActionClipBot_Colab.ipynb").read_text())
        assert nb["nbformat"] == 4
        kinds = [c["cell_type"] for c in nb["cells"]]
        assert "code" in kinds and "markdown" in kinds
        src = "\n".join("".join(c["source"]) for c in nb["cells"])
        assert "trycloudflare" in src and "keep-alive" in src

    def test_spec_has_no_gpu_duplication(self, tmp_env):
        # GPU backend lives on the Default Gpu page; the generic Settings
        # spec must not duplicate those keys.
        from src.dashboard.settings_spec import SPEC
        keys = [f["key"] for g in SPEC for f in g["fields"]]
        assert not any(k.startswith("gpu.") for k in keys)
        from src.config import _load_yaml
        raw = _load_yaml("settings.yaml")
        assert raw["gpu"]["backend"] == "runpod"
        assert "colab_url" in raw["gpu"]
