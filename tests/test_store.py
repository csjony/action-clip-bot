"""Tests for the SQLite Store — posts, credit ledger, monthly spend."""
from __future__ import annotations

import pytest

from src.store import Store


def test_create_post_returns_id(tmp_env):
    s = Store(tmp_env / "state.db")
    pid = s.create_post(title="Test", theme="neon_city_chase")
    assert isinstance(pid, int) and pid > 0
    s.close()


def test_credit_seeds_then_decrements(tmp_env):
    s = Store(tmp_env / "state.db")
    # First call seeds the row.
    assert s.credit_remaining("provider_a", "daily", cap=100) == 100
    s.consume_credit("provider_a", "daily", 1)
    assert s.credit_remaining("provider_a", "daily", cap=100) == 99
    # Drain and verify floor at 0.
    s.consume_credit("provider_a", "daily", 999)
    assert s.credit_remaining("provider_a", "daily", cap=100) == 0
    s.close()


def test_monthly_spend_only_counts_successful_paid(tmp_env):
    s = Store(tmp_env / "state.db")
    s.log_generation("paid_provider", "success", cost_usd=0.5)
    s.log_generation("paid_provider", "success", cost_usd=0.5)
    s.log_generation("paid_provider", "failed", cost_usd=0.5, error="boom")
    s.log_generation("free_provider", "success", cost_usd=0.0)  # free, doesn't count
    assert s.monthly_spend_usd() == pytest.approx(1.0)
    s.close()


def test_record_publish_url_writes_correct_column(tmp_env):
    s = Store(tmp_env / "state.db")
    pid = s.create_post(title="X")
    s.record_publish_url(pid, "youtube", "https://youtu.be/abc")
    s.record_publish_url(pid, "tiktok", "https://tiktok.com/x")
    row = s._conn.execute("SELECT * FROM posts WHERE id = ?", (pid,)).fetchone()
    assert dict(row)["youtube_url"] == "https://youtu.be/abc"
    assert dict(row)["tiktok_url"] == "https://tiktok.com/x"
    assert dict(row)["status"] == "published"
    s.close()


def test_unknown_platform_column_rejected(tmp_env):
    s = Store(tmp_env / "state.db")
    pid = s.create_post(title="X")
    with pytest.raises(ValueError):
        s.record_publish_url(pid, "myspace", "https://x")
    s.close()


def test_credit_period_isolation(tmp_env):
    """Daily and monthly counters are independent keys."""
    s = Store(tmp_env / "state.db")
    assert s.credit_remaining("provider_a", "daily", cap=100) == 100
    assert s.credit_remaining("provider_b", "monthly", cap=66) == 66
    s.consume_credit("provider_a", "daily", 1)
    # monthly counter untouched
    assert s.credit_remaining("provider_b", "monthly", cap=66) == 66
    s.close()


def test_log_api_call_and_usage_stats(tmp_env):
    s = Store(tmp_env / "state.db")
    aid = s.add_account("gemini", "Gemini Test Key", "key123")
    
    s.log_api_call("gemini", "success", account_id=aid, prompt_tokens=100, completion_tokens=50, cost_usd=0.002)
    s.log_api_call("gemini", "failed", account_id=aid, error="rate limited")
    s.log_api_call("gemini", "success", account_id=None, prompt_tokens=10, completion_tokens=5, cost_usd=0.0001)
    
    stats = s.get_api_usage_stats()
    assert len(stats) == 2
    
    aid_stat = next(x for x in stats if x["account_id"] == aid)
    assert aid_stat["total_requests"] == 2
    assert aid_stat["success_requests"] == 1
    assert aid_stat["total_prompt_tokens"] == 100
    assert aid_stat["total_completion_tokens"] == 50
    assert aid_stat["total_cost_usd"] == 0.002
    
    fallback_stat = next(x for x in stats if x["account_id"] is None)
    assert fallback_stat["total_requests"] == 1
    assert fallback_stat["success_requests"] == 1
    assert fallback_stat["total_prompt_tokens"] == 10
    assert fallback_stat["total_completion_tokens"] == 5
    assert fallback_stat["total_cost_usd"] == 0.0001
    
    s.close()

