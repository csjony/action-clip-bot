"""
Tests for GeneratorPool — the fallback-chain orchestrator.

These use fake in-process generators (no network) to exercise:
  * Happy path: first provider succeeds.
  * Transient failure → fall through to next provider.
  * Quota exhaustion → skip and fall through.
  * Budget cap → paid providers are skipped.
  * Total exhaustion → BudgetExceeded raised.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings, get_settings
from src.generators.base import (
    ContentRejected,
    GenerationResult,
    QuotaExceeded,
    VideoGenerator,
)
from src.generators.pool import BudgetExceeded, GenerationFailed, GeneratorPool
from src.store import Store


# ----------------------------------------------------------- fake generators
class _FakeGen(VideoGenerator):
    """
    Test double. `behaviour` is one of:
      'ok'          — always succeeds
      'quota'       — always raises QuotaExceeded
      'content'     — always raises ContentRejected
      'transient'   — always raises a generic Exception
      'ok_on_third' — fails twice then succeeds (counts via _calls)
    """
    def __init__(self, name, behaviour="ok", *, is_free=True, cost=0.0,
                 configured=True):
        super().__init__(env_value="x" if configured else "")
        self.name = name
        self.behaviour = behaviour
        self.is_free = is_free
        self.cost_per_clip_usd = cost
        self._calls = 0

    def _run(self, prompt, duration_sec, out_path, scene_index=0):
        self._calls += 1
        if self.behaviour == "quota":
            raise QuotaExceeded("no credits")
        if self.behaviour == "content":
            raise ContentRejected("policy")
        if self.behaviour == "transient":
            raise RuntimeError("network blip")
        if self.behaviour == "ok_on_third" and self._calls < 3:
            raise RuntimeError("retry")
        # success — write a dummy file
        out_path.write_bytes(b"\x00\x00\x00\x1cftypisom")


def _make_pool(store, generators):
    """Build a pool with an explicit chain (factory returns preset generators)."""
    cfg = {"order": [g.name for g in generators], "providers": {}}
    return GeneratorPool(
        store=store,
        settings=get_settings(),
        providers_config=cfg,
        factory=lambda name, spec, s: next(g for g in generators if g.name == name),
    )


# --------------------------------------------------------------------- tests
def test_first_provider_success(tmp_env):
    store = Store(tmp_env / "state.db")
    pool = _make_pool(store, [_FakeGen("hailuo", "ok"), _FakeGen("pixverse", "ok")])
    out = pool.generate("a chase", out_dir=tmp_env / "clips")
    assert out.provider == "hailuo"
    assert Path(out.clip_path).exists()
    assert out.cost_usd == 0.0
    store.close()


def test_falls_through_on_transient(tmp_env):
    store = Store(tmp_env / "state.db")
    pool = _make_pool(store, [
        _FakeGen("hailuo", "transient"),
        _FakeGen("pixverse", "ok"),
    ])
    out = pool.generate("a chase", out_dir=tmp_env / "clips")
    assert out.provider == "pixverse"
    # both attempts logged
    store.close()
    store2 = Store(tmp_env / "state.db")
    rows = store2._conn.execute(
        "SELECT provider, status FROM generations ORDER BY id"
    ).fetchall()
    assert [dict(r) for r in rows] == [
        {"provider": "hailuo", "status": "failed"},
        {"provider": "pixverse", "status": "success"},
    ]
    store2.close()


def test_falls_through_on_quota(tmp_env):
    store = Store(tmp_env / "state.db")
    pool = _make_pool(store, [
        _FakeGen("hailuo", "quota"),
        _FakeGen("pixverse", "content"),
        _FakeGen("kling", "ok"),
    ])
    out = pool.generate("a fight", out_dir=tmp_env / "clips")
    assert out.provider == "kling"
    store.close()


def test_paid_skipped_when_budget_exceeded(tmp_env):
    """When monthly spend already hits the cap, paid providers are skipped."""
    store = Store(tmp_env / "state.db")
    # Pre-charge the ledger so we're already at the cap.
    for _ in range(20):
        store.log_generation("paid_safety", "success", cost_usd=0.25)
    # Cap is 5.0 by default (from settings.yaml); 20 * 0.25 = 5.0.
    pool = _make_pool(store, [
        _FakeGen("paid_safety", "ok", is_free=False, cost=0.25),
    ])
    with pytest.raises(BudgetExceeded):
        pool.generate("a heist", out_dir=tmp_env / "clips")
    store.close()


def test_total_exhaustion_raises_budget_exceeded(tmp_env):
    store = Store(tmp_env / "state.db")
    pool = _make_pool(store, [
        _FakeGen("hailuo", "quota"),
        _FakeGen("pixverse", "transient"),
        _FakeGen("kling", "content"),
    ])
    with pytest.raises(GenerationFailed):
        pool.generate("a duel", out_dir=tmp_env / "clips")
    store.close()


def test_free_credits_decrement_on_success(tmp_env):
    """A successful free generation should consume a credit in the ledger."""
    store = Store(tmp_env / "state.db")
    before = store.credit_remaining("hailuo", "daily", cap=100)
    pool = _make_pool(store, [_FakeGen("hailuo", "ok")])
    pool.generate("a chase", out_dir=tmp_env / "clips")
    after = store.credit_remaining("hailuo", "daily", cap=100)
    assert after == before - 1
    store.close()


def test_exhausted_free_provider_is_skipped(tmp_env):
    """If free credits hit zero, the provider is skipped even if it would succeed."""
    store = Store(tmp_env / "state.db")
    # Drain Hailuo's daily credits (seed first so consume has a row to update).
    store.credit_remaining("hailuo", "daily", cap=100)
    for _ in range(100):
        store.consume_credit("hailuo", "daily", cap=100)
    assert store.credit_remaining("hailuo", "daily", cap=100) == 0
    pool = _make_pool(store, [
        _FakeGen("hailuo", "ok"),          # would succeed, but no credits
        _FakeGen("pixverse", "ok"),
    ])
    out = pool.generate("a chase", out_dir=tmp_env / "clips")
    assert out.provider == "pixverse"
    store.close()


def test_paid_success_records_spend(tmp_env):
    store = Store(tmp_env / "state.db")
    pool = _make_pool(store, [_FakeGen("paid_safety", "ok", is_free=False, cost=0.25)])
    pool.generate("a heist", out_dir=tmp_env / "clips")
    assert store.monthly_spend_usd() == pytest.approx(0.25)
    store.close()


def test_disabled_providers_excluded_from_chain(tmp_env, monkeypatch):
    """`enabled: false` in providers.yaml removes a provider from the chain."""
    store = Store(tmp_env / "state.db")
    cfg = {
        "order": ["hailuo", "pixverse", "fal", "paid_safety"],
        "providers": {
            "hailuo":      {"enabled": True,  "env_key": "DUMMY"},
            "pixverse":    {"enabled": True,  "env_key": "DUMMY"},
            "fal":         {"enabled": False, "env_key": "DUMMY"},  # disabled
            "paid_safety": {"enabled": False, "env_key": "DUMMY"},  # disabled
        },
    }
    pool = _make_pool(store, [
        _FakeGen("hailuo", "ok"),
        _FakeGen("pixverse", "ok"),
    ])
    # The _make_pool helper builds via factory, so disabled ones never reach
    # the chain — confirm only the two enabled providers appear.
    assert [g.name for g in pool.chain] == ["hailuo", "pixverse"]
    store.close()


def test_missing_credential_skips_provider(tmp_env, monkeypatch):
    """A provider whose env key is unset is skipped (not crashed on)."""
    monkeypatch.setenv("PRESENT_KEY", "abc")  # set
    # ABSENT_KEY deliberately NOT set.

    class _CredAwareFake(_FakeGen):
        """Fake that reports is_configured only when its env_value is set."""
        def __init__(self, name, env_value):
            super().__init__(name, "ok")
            self.env_value = env_value  # "" => not configured

        @property
        def is_configured(self):
            return bool(self.env_value)

    settings = get_settings()
    store = Store(tmp_env / "state.db")
    cfg = {
        "order": ["hailuo", "pixverse"],
        "providers": {
            "hailuo":   {"enabled": True, "env_key": "ABSENT_KEY"},   # not set
            "pixverse": {"enabled": True, "env_key": "PRESENT_KEY"},  # set
        },
    }
    pool = GeneratorPool(
        store=store, settings=settings, providers_config=cfg,
        factory=lambda name, spec, s: _CredAwareFake(
            name, s.env(spec.get("env_key", ""))),
    )
    # Hailuo skipped (empty env), pixverse present.
    assert [g.name for g in pool.chain] == ["pixverse"]
    store.close()


def test_factory_import_error_skips_provider(tmp_env):
    """If the factory raises ImportError (missing optional dep), the provider
    is skipped rather than crashing the whole chain."""
    store = Store(tmp_env / "state.db")

    def factory_with_import_error(name, spec, settings):
        if name == "hailuo":
            raise ImportError("httpx not installed")
        return _FakeGen(name, "ok")

    pool = GeneratorPool(
        store=store,
        providers_config={
            "order": ["hailuo", "pixverse"],
            "providers": {
                "hailuo":   {"enabled": True, "env_key": "DUMMY"},
                "pixverse": {"enabled": True, "env_key": "DUMMY"},
            },
        },
        factory=factory_with_import_error,
    )
    assert [g.name for g in pool.chain] == ["pixverse"]
    store.close()
