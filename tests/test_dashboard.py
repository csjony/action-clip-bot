"""
Tests for the dashboard subsystem — accounts, events, runner, and the
account-aware generator pool.

Covers:
  * AccountStore CRUD + resolve (dashboard accounts vs env-var fallback)
  * EventBus emit + helpers (with run context)
  * NullEventBus is a no-op
  * Store events / runs methods
  * Account-aware chain expansion in GeneratorPool
  * PipelineRunner single-flight + run_id pass-through
  * Auth middleware (open when no creds, rejects bad creds, accepts good creds)
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dashboard.accounts import Account, AccountStore
from src.dashboard.events import (
    EventBus,
    NullEventBus,
    KIND_CLIP_STARTED,
    KIND_PHASE_STARTED,
    KIND_PROVIDER_FAIL,
    KIND_PROVIDER_OK,
    KIND_PROVIDER_SKIP,
    KIND_PROVIDER_TRY,
    KIND_RUN_FAILED,
    KIND_RUN_FINISHED,
    KIND_RUN_STARTED,
    set_run_context,
)
from src.dashboard.runner import PipelineRunner, RunState
from src.store import Store


# ======================================================================
# AccountStore tests
# ======================================================================

class TestAccountStore:
    """CRUD and resolve for the accounts table."""

    def test_add_returns_id(self, tmp_env):
        store = Store(tmp_env / "state.db")
        as_ = AccountStore(store)
        aid = as_.add("hailuo", "main", "key-123", email="user@test.com")
        assert isinstance(aid, int) and aid > 0
        store.close()

    def test_get_roundtrip(self, tmp_env):
        store = Store(tmp_env / "state.db")
        as_ = AccountStore(store)
        aid = as_.add("pixverse", "backup", "pk-abc")
        row = as_.get(aid)
        assert row["provider"] == "pixverse"
        assert row["label"] == "backup"
        assert row["api_key"] == "pk-abc"
        assert row["enabled"] == 1
        store.close()

    def test_list_all(self, tmp_env):
        store = Store(tmp_env / "state.db")
        as_ = AccountStore(store)
        as_.add("hailuo", "main", "k1")
        as_.add("hailuo", "backup", "k2")
        as_.add("pixverse", "default", "k3")
        all_accts = as_.list_all()
        assert len(all_accts) == 3
        hailuo = as_.list_all("hailuo")
        assert len(hailuo) == 2
        store.close()

    def test_update_toggle(self, tmp_env):
        store = Store(tmp_env / "state.db")
        as_ = AccountStore(store)
        aid = as_.add("kling", "acc1", "key")
        as_.update(aid, enabled=False)
        row = as_.get(aid)
        assert row["enabled"] == 0
        as_.update(aid, enabled=True)
        assert as_.get(aid)["enabled"] == 1
        store.close()

    def test_delete(self, tmp_env):
        store = Store(tmp_env / "state.db")
        as_ = AccountStore(store)
        aid = as_.add("hailuo", "del_me", "k")
        assert as_.delete(aid) is True
        assert as_.get(aid) is None
        # Deleting non-existent returns False
        assert as_.delete(99999) is False
        store.close()

    def test_resolve_dashboard_accounts(self, tmp_env):
        """When dashboard has enabled accounts, resolve returns those."""
        store = Store(tmp_env / "state.db")
        as_ = AccountStore(store)
        as_.add("hailuo", "main", "key1", priority=10)
        as_.add("hailuo", "backup", "key2", priority=20)
        accounts = as_.resolve("hailuo", env_value="env_key_ignored")
        assert len(accounts) == 2
        assert accounts[0].label == "main"   # lower priority first
        assert accounts[0].id is not None
        assert accounts[1].label == "backup"
        store.close()

    def test_resolve_env_fallback(self, tmp_env):
        """No dashboard accounts → falls back to env_value."""
        store = Store(tmp_env / "state.db")
        as_ = AccountStore(store)
        accounts = as_.resolve("pixverse", env_value="pix_key_123")
        assert len(accounts) == 1
        assert accounts[0].is_env_fallback is True
        assert accounts[0].api_key == "pix_key_123"
        assert accounts[0].label == "env"
        store.close()

    def test_resolve_empty_when_nothing(self, tmp_env):
        """No dashboard accounts + no env_value → empty list."""
        store = Store(tmp_env / "state.db")
        as_ = AccountStore(store)
        accounts = as_.resolve("kling", env_value="")
        assert accounts == []
        store.close()

    def test_resolve_skips_disabled_accounts(self, tmp_env):
        store = Store(tmp_env / "state.db")
        as_ = AccountStore(store)
        as_.add("hailuo", "main", "key1", enabled=False)
        as_.add("hailuo", "backup", "key2")
        accounts = as_.resolve("hailuo")
        assert len(accounts) == 1
        assert accounts[0].label == "backup"
        store.close()

    def test_account_display_name(self, tmp_env):
        """Account.display_name = 'provider:label'."""
        a = Account(id=1, provider="hailuo", label="main", api_key="k")
        assert a.display_name == "hailuo:main"
        # Env fallback
        b = Account(id=None, provider="pixverse", label="env", api_key="k")
        assert b.display_name == "pixverse:env"
        assert b.is_env_fallback is True


# ======================================================================
# EventBus tests
# ======================================================================

class TestEventBus:
    """EventBus writes to the events table; silent when no run context."""

    def test_emit_writes_row(self, tmp_env):
        store = Store(tmp_env / "state.db")
        bus = EventBus(store)
        run_id = uuid.uuid4().hex
        set_run_context(run_id, post_id=42)
        try:
            bus.emit(
                KIND_PHASE_STARTED,
                message="scriptwriter starting",
                detail={"phase": "scriptwriter"},
            )
            events = store.list_events(run_id)
            assert len(events) == 1
            assert events[0]["kind"] == "phase_started"
            assert events[0]["message"] == "scriptwriter starting"
            assert events[0]["post_id"] == 42
        finally:
            set_run_context(None)
        store.close()

    def test_emit_silent_without_context(self, tmp_env):
        """No run context → emit is a no-op (no rows written)."""
        store = Store(tmp_env / "state.db")
        bus = EventBus(store)
        set_run_context(None)
        bus.emit(KIND_RUN_STARTED, message="should not appear")
        # No run_id to query — but we can check no rows exist
        with store._lock:
            count = store._conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
        assert count == 0
        store.close()

    def test_provider_helpers(self, tmp_env):
        store = Store(tmp_env / "state.db")
        bus = EventBus(store)
        run_id = uuid.uuid4().hex
        set_run_context(run_id)
        try:
            token = bus.provider_try("hailuo", account_id=1,
                                      scene_index=0, total_scenes=3)
            assert isinstance(token, int)

            bus.provider_skip("pixverse", account_id=2, reason="no credits",
                              scene_index=0, total_scenes=3)

            elapsed = bus.provider_elapsed(token, scene_index=0, provider="hailuo")
            assert elapsed >= 0

            bus.provider_fail("hailuo", account_id=1,
                              exc=RuntimeError("boom"),
                              scene_index=0, total_scenes=3, elapsed_ms=elapsed)

            bus.provider_ok("pixverse", account_id=2,
                            scene_index=1, total_scenes=3, elapsed_ms=100,
                            cost_usd=0.0, clip_path="/tmp/clip.mp4")

            events = store.list_events(run_id)
            kinds = [e["kind"] for e in events]
            assert kinds == [
                "provider_try", "provider_skip",
                "provider_fail", "provider_ok",
            ]
        finally:
            set_run_context(None)
        store.close()

    def test_emit_accepts_explicit_post_id(self, tmp_env):
        """Regression: emit() must accept post_id kwarg (pipeline passes it)."""
        store = Store(tmp_env / "state.db")
        bus = EventBus(store)
        run_id = uuid.uuid4().hex
        set_run_context(run_id)  # NOTE: post_id NOT set in contextvar
        try:
            # This used to raise TypeError because emit() lacked the post_id param
            bus.emit(KIND_CLIP_STARTED, post_id=42, message="clip starting")
            events = store.list_events(run_id)
            assert len(events) == 1
            # Explicit post_id is recorded even though contextvar was unset
            assert events[0]["post_id"] == 42
        finally:
            set_run_context(None)
        store.close()

    def test_explicit_post_id_overrides_context(self, tmp_env):
        """When both context and kwarg set post_id, the kwarg wins."""
        store = Store(tmp_env / "state.db")
        bus = EventBus(store)
        run_id = uuid.uuid4().hex
        set_run_context(run_id, post_id=10)
        try:
            bus.emit(KIND_CLIP_STARTED, post_id=99, message="explicit wins")
            events = store.list_events(run_id)
            assert events[0]["post_id"] == 99
        finally:
            set_run_context(None)
        store.close()

    def test_list_events_incremental(self, tmp_env):
        """list_events(after_id=N) returns only events after that id."""
        store = Store(tmp_env / "state.db")
        bus = EventBus(store)
        run_id = uuid.uuid4().hex
        set_run_context(run_id)
        try:
            bus.emit(KIND_RUN_STARTED, message="first")
            bus.emit(KIND_PHASE_STARTED, message="second")
            bus.emit(KIND_RUN_FINISHED, message="third")
            events = store.list_events(run_id)
            assert len(events) == 3
            # Get events after first event's id
            after = events[0]["id"]
            incremental = store.list_events(run_id, after_id=after)
            assert len(incremental) == 2
            assert incremental[0]["message"] == "second"
        finally:
            set_run_context(None)
        store.close()


class TestNullEventBus:
    """NullEventBus — every method is a no-op."""

    def test_methods_dont_crash(self):
        bus = NullEventBus()
        bus.emit("anything")
        bus.provider_try("x", 1, scene_index=0, total_scenes=1)
        bus.provider_skip("x", 1, "reason", scene_index=0, total_scenes=1)
        bus.provider_fail("x", 1, RuntimeError("e"), scene_index=0, total_scenes=1, elapsed_ms=10)
        bus.provider_ok("x", 1, scene_index=0, total_scenes=1, elapsed_ms=10,
                        cost_usd=0.0, clip_path="/x")

    def test_provider_try_returns_token(self):
        bus = NullEventBus()
        token = bus.provider_try("hailuo", None, scene_index=0, total_scenes=1)
        assert isinstance(token, int)


# ======================================================================
# Store events/runs methods
# ======================================================================

class TestStoreEvents:
    """Store.log_event, list_events, list_runs, current_run_id."""

    def test_log_event(self, tmp_env):
        store = Store(tmp_env / "state.db")
        run_id = uuid.uuid4().hex
        eid = store.log_event(
            run_id=run_id, kind="run_started", message="starting",
            detail_json='{"dry_run": true}',
        )
        assert isinstance(eid, int) and eid > 0
        store.close()

    def test_list_runs_groups_correctly(self, tmp_env):
        store = Store(tmp_env / "state.db")
        rid1 = uuid.uuid4().hex
        rid2 = uuid.uuid4().hex
        store.log_event(run_id=rid1, kind="run_started", message="a")
        store.log_event(run_id=rid1, kind="run_finished", message="b")
        store.log_event(run_id=rid2, kind="run_started", message="c")
        runs = store.list_runs()
        assert len(runs) == 2
        # Both runs have 2 and 1 events respectively
        by_id = {r["run_id"]: r for r in runs}
        assert by_id[rid1]["event_count"] == 2
        assert by_id[rid2]["event_count"] == 1
        # latest_kind is the most recent event's kind
        assert by_id[rid1]["latest_kind"] == "run_finished"
        assert by_id[rid2]["latest_kind"] == "run_started"
        store.close()

    def test_current_run_id_returns_active(self, tmp_env):
        """current_run_id returns the run that started but hasn't finished."""
        store = Store(tmp_env / "state.db")
        rid_active = uuid.uuid4().hex
        rid_done = uuid.uuid4().hex
        # Active run: started but not finished
        store.log_event(run_id=rid_active, kind="run_started", message="a")
        # Done run: started and finished
        store.log_event(run_id=rid_done, kind="run_started", message="b")
        store.log_event(run_id=rid_done, kind="run_finished", message="c")
        current = store.current_run_id()
        assert current == rid_active
        store.close()


# ======================================================================
# Account-aware GeneratorPool tests
# ======================================================================

class TestAccountAwarePool:
    """Verify the pool expands providers into per-account entries."""

    def _make_fake(self, name, behaviour="ok", **kw):
        """Factory that returns _FakeGen-compatible objects for the custom path."""
        from tests.test_pool import _FakeGen
        return _FakeGen(name, behaviour, **kw)

    def test_custom_factory_chain_preserves_names(self, tmp_env):
        """Custom factory (test path) wraps with label='' so name == provider_key."""
        store = Store(tmp_env / "state.db")
        from src.config import get_settings
        from src.generators.pool import GeneratorPool
        cfg = {"order": ["hailuo", "pixverse"], "providers": {}}
        pool = GeneratorPool(
            store=store,
            settings=get_settings(),
            providers_config=cfg,
            factory=lambda name, spec, s: self._make_fake(name),
        )
        assert [g.name for g in pool.chain] == ["hailuo", "pixverse"]
        # All are env-var path (label="")
        assert all(g.label == "" for g in pool.chain)
        store.close()

    def test_account_store_expand_per_provider(self, tmp_env):
        """When AccountStore returns 2 accounts for local, chain has 2 entries."""
        store = Store(tmp_env / "state.db")
        as_ = AccountStore(store)
        as_.add("local", "main", "http://local1", priority=10)
        as_.add("local", "backup", "http://local2", priority=20)

        from src.config import get_settings
        from src.generators.pool import GeneratorPool

        # Do NOT pass factory — this exercises the production path where
        # account_store.resolve() is called. The _factory_for_account method
        # will create real LocalGenerator instances.
        cfg = {"order": ["local"], "providers": {"local": {"enabled": True}}}
        pool = GeneratorPool(
            store=store,
            settings=get_settings(),
            providers_config=cfg,
            account_store=as_,
        )
        # Two entries for local (one per account)
        assert len(pool.chain) == 2
        assert pool.chain[0].name == "local:main"
        assert pool.chain[1].name == "local:backup"
        assert pool.chain[0].account_id is not None
        assert pool.chain[1].account_id is not None
        store.close()

    def test_event_bus_receives_provider_events(self, tmp_env):
        """Pool.generate emits events to the event_bus for each attempt."""
        store = Store(tmp_env / "state.db")
        run_id = uuid.uuid4().hex
        set_run_context(run_id)
        try:
            bus = EventBus(store)
            from src.config import get_settings
            from src.generators.pool import GeneratorPool
            from tests.test_pool import _FakeGen
            cfg = {"order": ["hailuo", "pixverse"], "providers": {}}
            pool = GeneratorPool(
                store=store,
                settings=get_settings(),
                providers_config=cfg,
                factory=lambda name, spec, s: _FakeGen(name, "ok"),
                event_bus=bus,
            )
            result = pool.generate(
                "test prompt",
                out_dir=tmp_env / "clips",
                scene_index=1,
                total_scenes=3,
            )
            assert result.provider == "hailuo"
            # Should have a provider_try and provider_ok event
            events = store.list_events(run_id)
            kinds = [e["kind"] for e in events]
            assert "provider_try" in kinds
            assert "provider_ok" in kinds
        finally:
            set_run_context(None)
        store.close()


# ======================================================================
# PipelineRunner tests
# ======================================================================

class TestPipelineRunner:
    """Single-flight runner with run_id pass-through."""

    def test_submit_returns_state(self, tmp_env):
        """submit() returns a RunState with a generated run_id."""
        called_with = {}
        barrier = threading.Event()

        def fake_pipeline(**kwargs):
            called_with.update(kwargs)
            pipeline = MagicMock()
            def slow_run():
                barrier.wait(timeout=2)
                return 0
            pipeline.run = slow_run
            return pipeline

        runner = PipelineRunner(fake_pipeline)
        state = runner.submit(dry_run=True, theme="neon")
        # While the thread is blocked on the barrier, state should still be running
        assert state.status == "running"
        assert state.dry_run is True
        assert state.theme == "neon"
        assert state.run_id is not None

        # Release the barrier and wait for completion
        barrier.set()
        time.sleep(0.3)
        assert state.status == "finished"
        assert state.exit_code == 0
        # Verify run_id was passed to the factory
        assert called_with.get("run_id") == state.run_id
        assert called_with.get("dry_run") is True
        assert called_with.get("theme") == "neon"

    def test_single_flight_rejects_concurrent(self, tmp_env):
        """Submitting while a run is active raises RuntimeError."""
        def slow_pipeline(**kwargs):
            pipeline = MagicMock()
            pipeline.run.side_effect = lambda: time.sleep(1) or 0
            return pipeline

        runner = PipelineRunner(slow_pipeline)
        runner.submit(dry_run=False)
        with pytest.raises(RuntimeError, match="already in progress"):
            runner.submit(dry_run=True)
        # Wait for cleanup
        time.sleep(1.5)

    def test_pipeline_failure_captured(self, tmp_env):
        """If the pipeline raises, state.status = 'failed'."""
        def failing_pipeline(**kwargs):
            raise RuntimeError("database exploded")

        runner = PipelineRunner(failing_pipeline)
        state = runner.submit()
        time.sleep(0.2)
        assert state.status == "failed"
        assert "database exploded" in state.error

    def test_is_busy_and_current_run_id(self, tmp_env):
        def slow_pipeline(**kwargs):
            pipeline = MagicMock()
            pipeline.run.side_effect = lambda: time.sleep(0.5) or 0
            return pipeline

        runner = PipelineRunner(slow_pipeline)
        assert runner.is_busy is False
        assert runner.current_run_id is None

        state = runner.submit()
        assert runner.is_busy is True
        assert runner.current_run_id == state.run_id

        time.sleep(0.8)
        assert runner.is_busy is False
        assert runner.current_run_id is None

    def test_recent_runs(self, tmp_env):
        def fast_pipeline(**kwargs):
            p = MagicMock()
            p.run.return_value = 0
            return p

        runner = PipelineRunner(fast_pipeline)
        s1 = runner.submit()
        time.sleep(0.2)
        s2 = runner.submit()
        time.sleep(0.2)

        recent = runner.recent_runs(limit=10)
        assert len(recent) >= 2
        run_ids = {r.run_id for r in recent}
        assert s1.run_id in run_ids
        assert s2.run_id in run_ids

    def test_get_run(self, tmp_env):
        def fast_pipeline(**kwargs):
            p = MagicMock()
            p.run.return_value = 0
            return p

        runner = PipelineRunner(fast_pipeline)
        state = runner.submit()
        time.sleep(0.2)
        fetched = runner.get_run(state.run_id)
        assert fetched is not None
        assert fetched.status == "finished"
        assert fetched.run_id == state.run_id

    def test_make_pipeline_arguments(self, tmp_env):
        """Verify that _make_pipeline accepts all arguments passed by the runner."""
        from src.dashboard.app import _make_pipeline
        import threading
        stop_event = threading.Event()
        with patch("src.pipeline.Pipeline") as mock_pipeline_cls:
            _make_pipeline(
                dry_run=True,
                theme="nature",
                run_id="run-123",
                stop_event=stop_event,
                quick_test=True,
            )
            mock_pipeline_cls.assert_called_once_with(
                dry_run=True,
                theme="nature",
                run_id="run-123",
                stop_event=stop_event,
                quick_test=True,
                script_json=None,
            )


# ======================================================================
# Auth middleware tests
# ======================================================================

class TestAuthMiddleware:
    """Tests for the _check_auth helper in app.py."""

    def _check_auth(self, request):
        """Import the helper directly from app module."""
        from src.dashboard.app import _check_auth
        return _check_auth(request)

    def _make_request(self, headers=None):
        req = MagicMock()
        req.headers = headers or {}
        return req

    def test_open_when_no_env_creds(self, tmp_env, monkeypatch):
        """No DASHBOARD_USER/PASSWORD set → all requests allowed."""
        monkeypatch.setenv("DASHBOARD_USER", "")
        monkeypatch.setenv("DASHBOARD_PASSWORD", "")
        assert self._check_auth(self._make_request()) is True

    def test_rejects_no_auth_header(self, tmp_env, monkeypatch):
        """Creds set but no Authorization header → rejected."""
        monkeypatch.setenv("DASHBOARD_USER", "admin")
        monkeypatch.setenv("DASHBOARD_PASSWORD", "secret")
        assert self._check_auth(self._make_request()) is False

    def test_accepts_valid_credentials(self, tmp_env, monkeypatch):
        import base64
        monkeypatch.setenv("DASHBOARD_USER", "admin")
        monkeypatch.setenv("DASHBOARD_PASSWORD", "secret")
        token = base64.b64encode(b"admin:secret").decode()
        req = self._make_request({"Authorization": f"Basic {token}"})
        assert self._check_auth(req) is True

    def test_rejects_wrong_credentials(self, tmp_env, monkeypatch):
        import base64
        monkeypatch.setenv("DASHBOARD_USER", "admin")
        monkeypatch.setenv("DASHBOARD_PASSWORD", "secret")
        token = base64.b64encode(b"admin:wrongpass").decode()
        req = self._make_request({"Authorization": f"Basic {token}"})
        assert self._check_auth(req) is False

    def test_rejects_malformed_header(self, tmp_env, monkeypatch):
        monkeypatch.setenv("DASHBOARD_USER", "admin")
        monkeypatch.setenv("DASHBOARD_PASSWORD", "secret")
        req = self._make_request({"Authorization": "Bearer abc123"})
        assert self._check_auth(req) is False


# ======================================================================
# Integration: store accounts + events together
# ======================================================================

class TestStoreAccountsIntegration:
    """Accounts and events tables coexist in the same DB."""

    def test_accounts_and_events_independent(self, tmp_env):
        store = Store(tmp_env / "state.db")
        # Add an account
        aid = AccountStore(store).add("hailuo", "main", "key")
        # Log an event
        run_id = uuid.uuid4().hex
        store.log_event(run_id=run_id, kind="run_started")
        # Both queries work
        assert store.get_account(aid) is not None
        assert len(store.list_events(run_id)) == 1
        store.close()


@pytest.mark.anyio
async def test_api_usage_stats_route(tmp_env, monkeypatch):
    from src.dashboard.app import api_usage_stats, store, acct_store
    import src.dashboard.app

    temp_store = Store(tmp_env / "state.db")
    monkeypatch.setattr(src.dashboard.app, "store", temp_store)
    monkeypatch.setattr(src.dashboard.app, "acct_store", AccountStore(temp_store))

    # Add dummy account
    aid = temp_store.add_account("gemini", "Gemini Test Key", "key123")

    # Log some calls
    temp_store.log_api_call("gemini", "success", account_id=aid, prompt_tokens=100, completion_tokens=50, cost_usd=0.002)

    res = await api_usage_stats(range="24h")
    assert "accounts_stats" in res
    assert len(res["accounts_stats"]) == 1
    assert res["accounts_stats"][0]["account_id"] == aid
    assert res["accounts_stats"][0]["stats"]["total_requests"] == 1
    assert res["accounts_stats"][0]["stats"]["total_prompt_tokens"] == 100

    temp_store.close()

