"""LLM script generation: backends resolve from dashboard accounts + env,
theme-only pipeline planning, and the dropdown-only generate routes."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config import get_settings
from src.content.scriptwriter import GroqBackend, OfflineBackend
from src.dashboard.accounts import AccountStore
from src.store import Store


def _writer(tmp_env, monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from src.content.scriptwriter import Scriptwriter
    store = Store(tmp_env / "data" / "state.db")
    return Scriptwriter(get_settings(), store), store


class TestBackendResolution:
    def test_env_key_builds_backend(self, tmp_env, monkeypatch):
        w, store = _writer(tmp_env, monkeypatch, GEMINI_API_KEY="g-key")
        kinds = [type(b).__name__ for b in w._backends]
        assert kinds == ["GeminiBackend", "OfflineBackend"]
        assert w._backends[0].account_id is None
        store.close()

    def test_dashboard_accounts_rotate(self, tmp_env, monkeypatch):
        store = Store(tmp_env / "data" / "state.db")
        accts = AccountStore(store)
        accts.add("groq", "main", "k1", priority=10)
        accts.add("groq", "backup", "k2", priority=20)
        from src.content.scriptwriter import Scriptwriter
        w = Scriptwriter(get_settings(), store, accts)
        groq = [b for b in w._backends if isinstance(b, GroqBackend)]
        assert len(groq) == 2
        assert [b.account_id for b in groq] == sorted(b.account_id for b in groq)
        assert isinstance(w._backends[-1], OfflineBackend)
        store.close()

    def test_no_keys_offline_only(self, tmp_env, monkeypatch):
        w, store = _writer(tmp_env, monkeypatch)
        assert [type(b).__name__ for b in w._backends] == ["OfflineBackend"]
        plan = w.write(theme="neon_city_chase")
        assert plan.theme == "neon_city_chase"
        assert len(plan.scenes) > 0
        store.close()

    def test_temperature_from_settings(self, tmp_env, monkeypatch):
        w, store = _writer(tmp_env, monkeypatch, GROQ_API_KEY="q")
        assert isinstance(w._backends[0], GroqBackend)
        store.close()


class TestPipelinePlan:
    def test_manual_override_still_parses(self, tmp_env):
        import json
        from src.pipeline import Pipeline
        script = json.dumps({
            "title": "T", "theme": "neon_city_chase", "hook": "H",
            "scenes": [{"prompt": "a rainy neon street chase at night", "duration_sec": 5, "sound_query": "q"}],
            "narration": "N", "hashtags": ["#a"],
            "captions": {"youtube": "y", "facebook": "f", "instagram": "i",
                         "threads": "t", "tiktok": "k"},
        })
        pipe = Pipeline(dry_run=True, script_json=script)
        plan = pipe._plan_content()
        assert plan.title == "T" and len(plan.scenes) == 1

    def test_theme_only_uses_writer(self, tmp_env):
        from src.pipeline import Pipeline
        fake_plan = MagicMock()
        fake_plan.title = "AI"
        fake_plan.scenes = []
        with patch("src.content.scriptwriter.Scriptwriter.write",
                   return_value=fake_plan) as mock_write:
            pipe = Pipeline(dry_run=True, theme="heist_getaway")
            assert pipe._plan_content() is fake_plan
            mock_write.assert_called_once_with(theme="heist_getaway")


class TestGenerateRoutes:
    def test_get_shows_themes_and_llm_status(self, tmp_env, monkeypatch):
        from fastapi.testclient import TestClient
        import src.dashboard.app as appmod
        temp_store = Store(tmp_env / "data" / "state.db")
        monkeypatch.setattr(appmod, "store", temp_store)
        monkeypatch.setattr(appmod, "acct_store", AccountStore(temp_store))
        c = TestClient(appmod.app)
        r = c.get("/generate")
        assert r.status_code == 200
        assert "neon_city_chase" in r.text
        assert "gemini" in r.text and "groq" in r.text
        assert "script_json" not in r.text
        temp_store.close()

    def test_post_rejects_bad_theme(self, tmp_env, monkeypatch):
        from fastapi.testclient import TestClient
        import src.dashboard.app as appmod
        temp_store = Store(tmp_env / "data" / "state.db")
        monkeypatch.setattr(appmod, "store", temp_store)
        monkeypatch.setattr(appmod, "acct_store", AccountStore(temp_store))
        c = TestClient(appmod.app)
        r = c.post("/generate", data={"theme": "nope"})
        assert r.status_code == 200
        assert "valid theme" in r.text
        temp_store.close()

    def test_post_submits_theme_only(self, tmp_env, monkeypatch):
        from fastapi.testclient import TestClient
        import src.dashboard.app as appmod
        temp_store = Store(tmp_env / "data" / "state.db")
        monkeypatch.setattr(appmod, "store", temp_store)
        monkeypatch.setattr(appmod, "acct_store", AccountStore(temp_store))
        fake_state = MagicMock()
        fake_state.run_id = "run123"
        fake_runner = MagicMock()
        fake_runner.is_busy = False
        fake_runner.current_run_id = None
        fake_runner.submit.return_value = fake_state
        monkeypatch.setattr(appmod, "runner", fake_runner)
        c = TestClient(appmod.app, follow_redirects=False)
        r = c.post("/generate", data={"theme": "heist_getaway", "quick_test": "on"})
        assert r.status_code == 303
        assert r.headers["location"] == "/runs/run123"
        _, kwargs = fake_runner.submit.call_args
        assert kwargs["theme"] == "heist_getaway"
        assert kwargs["quick_test"] is True
        assert kwargs["script_json"] is None
        temp_store.close()
