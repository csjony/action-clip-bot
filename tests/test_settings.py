"""Tests for centralized tunables: dotted Settings.get, DB overrides,
credit-cap precedence, and the settings-page spec coverage."""
from __future__ import annotations

import json

from src.config import get_settings
from src.store import Store


class TestDottedGet:
    def test_get_nested(self, tmp_env):
        s = get_settings()
        assert s.get("editor.crf") == 18
        assert s.get("video.generation.fps") == 16
        assert s.get("nope.missing", "fallback") == "fallback"
        assert s.get("editor.crf.deeper", "fallback") == "fallback"
        s2 = Store(tmp_env / "data" / "state.db")
        s2.close()

    def test_get_list(self, tmp_env):
        s = get_settings()
        enh = s.get("generator_client.prompt_enhancers", [])
        assert isinstance(enh, list) and "photorealistic" in enh


class TestOverrides:
    def test_roundtrip(self, tmp_env):
        from src import config
        store = Store(tmp_env / "data" / "state.db")
        assert store.list_overrides() == []
        store.set_override("editor.crf", json.dumps(20))
        assert store.get_override("editor.crf") == "20"
        config.get_settings.cache_clear()
        try:
            assert get_settings().get("editor.crf") == 20
        finally:
            config.get_settings.cache_clear()
        assert store.delete_override("editor.crf") is True
        assert store.delete_override("editor.crf") is False
        store.close()

    def test_effective_merge(self, tmp_env):
        from src import config
        store = Store(tmp_env / "data" / "state.db")
        store.set_override("dashboard.toast_ms", json.dumps(9999))
        config.get_settings.cache_clear()
        try:
            s = get_settings()
            assert s.get("dashboard.toast_ms") == 9999
            # Untouched keys still come from the file.
            assert s.get("dashboard.logs_poll_ms") == 1500
        finally:
            config.get_settings.cache_clear()
        store.close()


class TestCreditPrecedence:
    def test_defaults(self, tmp_env):
        from src.generators.pool import _credit_spec
        kind, cap = _credit_spec({"order": [], "providers": {}}, "local")
        assert (kind, cap) == ("daily", 1_000_000)

    def test_providers_yaml(self, tmp_env):
        from src.generators.pool import _credit_spec
        # 'freesound' has no settings.yaml credits entry, so providers.yaml applies.
        cfg = {"order": ["freesound"],
               "providers": {"freesound": {"credits": {"kind": "monthly", "cap": 7}}}}
        assert _credit_spec(cfg, "freesound") == ("monthly", 7)

    def test_settings_layer_wins(self, tmp_env):
        from src.generators.pool import _credit_spec
        # settings.yaml credits.local (daily/1000000) beats providers.yaml.
        cfg = {"order": ["local"],
               "providers": {"local": {"credits": {"kind": "monthly", "cap": 7}}}}
        assert _credit_spec(cfg, "local") == ("daily", 1000000)

    def test_settings_override_wins(self, tmp_env):
        from src import config
        from src.generators.pool import _credit_spec
        store = Store(tmp_env / "data" / "state.db")
        store.set_override("credits.local", json.dumps({"kind": "daily", "cap": 42}))
        config.get_settings.cache_clear()
        try:
            cfg = {"order": ["local"],
                   "providers": {"local": {"credits": {"kind": "monthly", "cap": 7}}}}
            assert _credit_spec(cfg, "local") == ("daily", 42)
        finally:
            config.get_settings.cache_clear()
        store.close()


class TestSpecCoverage:
    def test_every_spec_key_resolves(self, tmp_env):
        from src.dashboard.settings_spec import SPEC
        from src.config import _load_yaml
        raw = _load_yaml("settings.yaml")

        def lookup(key):
            node = raw
            for part in key.split("."):
                assert isinstance(node, dict) and part in node, f"missing in settings.yaml: {key}"
                node = node[part]
            return node

        for g in SPEC:
            assert g["group"] and g["fields"]
            for f in g["fields"]:
                assert f["key"] and f["label"] and f["type"]
                lookup(f["key"])
                if f["type"] == "select":
                    assert f["options"], f["key"]
                    assert lookup(f["key"]) in f["options"], f["key"]
