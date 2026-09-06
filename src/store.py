"""
SQLite-backed state for action-clip-bot.

Three concerns:
  * `posts`         — every video we generate + where it was published
  * `credits`       — per-provider free-credit ledger (the fallback chain's brain)
  * `generations`   — every clip attempt (success or fail) → drives monthly spend

All access goes through the `Store` class so callers never write raw SQL.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    title           TEXT    NOT NULL,
    theme           TEXT,
    vertical_path   TEXT,
    horizontal_path TEXT,
    script_json     TEXT,
    youtube_url     TEXT,
    facebook_url    TEXT,
    instagram_url   TEXT,
    threads_url     TEXT,
    tiktok_url      TEXT,
    status          TEXT    NOT NULL DEFAULT 'drafted'   -- drafted|published|failed
);

CREATE TABLE IF NOT EXISTS credits (
    provider        TEXT    NOT NULL,
    period_key      TEXT    NOT NULL,                  -- 'YYYY-MM-DD' (daily) or 'YYYY-MM' (monthly)
    period_kind     TEXT    NOT NULL,                  -- 'daily' | 'monthly'
    remaining       INTEGER NOT NULL,
    cap             INTEGER NOT NULL,
    PRIMARY KEY (provider, period_key, period_kind)
);

CREATE TABLE IF NOT EXISTS generations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    post_id         INTEGER,
    provider        TEXT    NOT NULL,
    scene_index     INTEGER,
    status          TEXT    NOT NULL,                  -- success|failed
    cost_usd        REAL    NOT NULL DEFAULT 0,
    error           TEXT,
    clip_path       TEXT,
    FOREIGN KEY (post_id) REFERENCES posts(id)
);

-- Multiple API accounts per provider, managed from the dashboard.
-- Falls back to env-var credentials when no enabled accounts exist
-- for a provider (see AccountStore).
CREATE TABLE IF NOT EXISTS accounts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    provider        TEXT    NOT NULL,                  -- local|gemini|groq|...
    label           TEXT    NOT NULL,                  -- short user-facing name
    email           TEXT,                              -- the account email (optional)
    api_key         TEXT    NOT NULL,                  -- primary credential
    extra_json      TEXT,                              -- {group_id, secret_key, ...}
    enabled         INTEGER NOT NULL DEFAULT 1,
    priority        INTEGER NOT NULL DEFAULT 100,      -- lower = tried first
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_accounts_provider ON accounts(provider, enabled, priority);

-- Append-only progress log for the dashboard's live-run view.
-- Written by EventBus from pool.py + pipeline.py.
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,                  -- UUID per Pipeline.run()
    post_id         INTEGER,
    ts              TEXT    NOT NULL,
    kind            TEXT    NOT NULL,                  -- see EventBus.KIND_*
    scene_index     INTEGER,
    total_scenes    INTEGER,
    provider        TEXT,                              -- provider:label when account-aware
    account_id      INTEGER,                           -- accounts.id (nullable)
    message         TEXT,                              -- one-line human summary
    elapsed_ms      INTEGER,
    detail_json     TEXT                               -- arbitrary structured payload
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);

CREATE TABLE IF NOT EXISTS api_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT    NOT NULL,
    account_id          INTEGER,                           -- references accounts(id), NULL for env fallback
    provider            TEXT    NOT NULL,                  -- gemini|groq|local|jamendo
    status              TEXT    NOT NULL,                  -- success|failed
    prompt_tokens       INTEGER DEFAULT 0,
    completion_tokens   INTEGER DEFAULT 0,
    cost_usd            REAL    NOT NULL DEFAULT 0.0,
    error               TEXT,
    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_api_logs_created_at ON api_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_api_logs_account_id ON api_logs(account_id);
"""


@dataclass(frozen=True)
class Generation:
    provider: str
    cost_usd: float
    status: str


class Store:
    """Thread-safe wrapper around the SQLite database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # A single connection guarded by a lock — simpler than a pool and
        # sufficient for the low write volume of one cron job.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("PRAGMA journal_mode=WAL;")
        self._conn.executescript(SCHEMA)
        self._run_migrations()

    # -------------------------------------------------------------- migrations
    def _run_migrations(self) -> None:
        """One-time data migrations for legacy rows.

        These are idempotent — safe to run on every startup. They clean up
        artefacts left over from older versions of the bot (Colab references,
        Pixabay provider rows, etc.).
        """
        with self._lock:
            # 1. Rename 'Colab' labels on local provider accounts → 'GPU VPS'
            self._conn.execute(
                """UPDATE accounts SET label = 'GPU VPS', updated_at = ?
                   WHERE provider = 'local'
                     AND LOWER(label) LIKE '%colab%'""",
                (self._now(),),
            )
            # 2. Remove deprecated pixabay accounts entirely
            self._conn.execute(
                "DELETE FROM accounts WHERE LOWER(provider) = 'pixabay'"
            )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------ posts
    def create_post(
        self,
        title: str,
        theme: str | None = None,
        script_json: str | None = None,
    ) -> int:
        """Insert a new post record in 'drafted' status, return its id."""
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO posts (created_at, title, theme, script_json, status)
                   VALUES (?, ?, ?, ?, 'drafted')""",
                (self._now(), title, theme, script_json),
            )
            return int(cur.lastrowid)

    def update_post_paths(self, post_id: int, vertical: str, horizontal: str) -> None:
        with self._tx() as conn:
            conn.execute(
                """UPDATE posts SET vertical_path = ?, horizontal_path = ?
                   WHERE id = ?""",
                (vertical, horizontal, post_id),
            )

    def record_publish_url(self, post_id: int, platform: str, url: str) -> None:
        col = f"{platform}_url"
        allowed = {"youtube_url", "facebook_url", "instagram_url",
                   "threads_url", "tiktok_url"}
        if col not in allowed:
            raise ValueError(f"unknown platform column: {col}")
        with self._tx() as conn:
            conn.execute(
                f"UPDATE posts SET {col} = ?, status = 'published' WHERE id = ?",
                (url, post_id),
            )

    def mark_post_failed(self, post_id: int) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE posts SET status = 'failed' WHERE id = ?", (post_id,)
            )

    # ------------------------------------------------------------ generations
    def log_generation(
        self,
        provider: str,
        status: str,
        *,
        post_id: int | None = None,
        scene_index: int | None = None,
        cost_usd: float = 0.0,
        error: str | None = None,
        clip_path: str | None = None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO generations
                   (created_at, post_id, provider, scene_index, status,
                    cost_usd, error, clip_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (self._now(), post_id, provider, scene_index, status,
                 cost_usd, error, clip_path),
            )

    def monthly_spend_usd(self, *, year: int | None = None, month: int | None = None) -> float:
        """Total PAID generation cost in the given calendar month (UTC)."""
        now = datetime.now(timezone.utc)
        year = year or now.year
        month = month or now.month
        prefix = f"{year:04d}-{month:02d}"
        with self._tx() as conn:
            row = conn.execute(
                """SELECT COALESCE(SUM(cost_usd), 0) AS total
                   FROM generations
                   WHERE status = 'success' AND created_at LIKE ? || '%'""",
                (prefix,),
            ).fetchone()
        return float(row["total"] or 0.0)

    # ---------------------------------------------------------------- credits
    def credit_remaining(self, provider: str, period_kind: str, cap: int) -> int:
        """
        Return remaining free credits for a provider in the current period.

        Lazily seeds the row when a new period starts, so callers just need
        the configured `cap` per provider. Returns 0 when exhausted.
        """
        now = datetime.now(timezone.utc)
        if period_kind == "daily":
            period_key = now.strftime("%Y-%m-%d")
        elif period_kind == "monthly":
            period_key = now.strftime("%Y-%m")
        else:
            raise ValueError(f"unknown period_kind: {period_kind}")

        with self._tx() as conn:
            row = conn.execute(
                """SELECT remaining FROM credits
                   WHERE provider = ? AND period_key = ? AND period_kind = ?""",
                (provider, period_key, period_kind),
            ).fetchone()
            if row is None:
                conn.execute(
                    """INSERT INTO credits (provider, period_key, period_kind, remaining, cap)
                       VALUES (?, ?, ?, ?, ?)""",
                    (provider, period_key, period_kind, cap, cap),
                )
                return cap
            return int(row["remaining"])

    def consume_credit(self, provider: str, period_kind: str, amount: int = 1,
                       *, cap: int | None = None) -> None:
        """
        Decrement a provider's free credits by `amount` (floor at 0).

        Seeds the row on first use for the period so callers can consume
        without a prior `credit_remaining()` call. Pass `cap` to control the
        seed value; if omitted, an existing row's cap is reused, or a sane
        default (1000) is used — credit_remaining() will re-seed correctly on
        the next read if needed.
        """
        now = datetime.now(timezone.utc)
        period_key = now.strftime("%Y-%m-%d" if period_kind == "daily" else "%Y-%m")
        with self._tx() as conn:
            existing = conn.execute(
                """SELECT remaining, cap FROM credits
                   WHERE provider = ? AND period_key = ? AND period_kind = ?""",
                (provider, period_key, period_kind),
            ).fetchone()
            if existing is None:
                seed_cap = cap if cap is not None else 1000
                new_remaining = max(0, seed_cap - amount)
                conn.execute(
                    """INSERT INTO credits (provider, period_key, period_kind, remaining, cap)
                       VALUES (?, ?, ?, ?, ?)""",
                    (provider, period_key, period_kind, new_remaining, seed_cap),
                )
            else:
                conn.execute(
                    """UPDATE credits SET remaining = MAX(0, remaining - ?)
                       WHERE provider = ? AND period_key = ? AND period_kind = ?""",
                    (amount, provider, period_key, period_kind),
                )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- accounts
    def add_account(
        self,
        provider: str,
        label: str,
        api_key: str,
        *,
        email: str | None = None,
        extra_json: str | None = None,
        enabled: bool = True,
        priority: int = 100,
    ) -> int:
        """Insert an API account row; return its id."""
        now = self._now()
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO accounts
                   (provider, label, email, api_key, extra_json,
                    enabled, priority, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (provider, label, email, api_key, extra_json,
                 1 if enabled else 0, priority, now, now),
            )
            return int(cur.lastrowid)

    def update_account(
        self, account_id: int, *,
        provider: str | None = None,
        label: str | None = None,
        api_key: str | None = None,
        email: str | None = None,
        extra_json: str | None = None,
        enabled: bool | None = None,
        priority: int | None = None,
    ) -> bool:
        """Patch any subset of an account row. Returns True if a row matched."""
        sets, params = [], []
        if provider is not None:
            sets.append("provider = ?"); params.append(provider)
        if label is not None:
            sets.append("label = ?"); params.append(label)
        if api_key is not None:
            sets.append("api_key = ?"); params.append(api_key)
        if email is not None:
            sets.append("email = ?"); params.append(email)
        if extra_json is not None:
            sets.append("extra_json = ?"); params.append(extra_json)
        if enabled is not None:
            sets.append("enabled = ?"); params.append(1 if enabled else 0)
        if priority is not None:
            sets.append("priority = ?"); params.append(priority)
        if not sets:
            return False
        sets.append("updated_at = ?"); params.append(self._now())
        params.append(account_id)
        with self._tx() as conn:
            cur = conn.execute(
                f"UPDATE accounts SET {', '.join(sets)} WHERE id = ?", params,
            )
            return cur.rowcount > 0

    def delete_account(self, account_id: int) -> bool:
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
            return cur.rowcount > 0

    def list_accounts(self, provider: str | None = None) -> list[dict]:
        """All accounts ordered by provider, priority. Optional provider filter."""
        sql = "SELECT * FROM accounts"
        params: list = []
        if provider is not None:
            sql += " WHERE provider = ?"; params.append(provider)
        sql += " ORDER BY provider, priority, id"
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def get_account(self, account_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM accounts WHERE id = ?", (account_id,)
            ).fetchone()
            return dict(row) if row else None

    def enabled_accounts(self, provider: str) -> list[dict]:
        """Enabled accounts for a provider, in priority order."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM accounts WHERE provider = ? AND enabled = 1
                   ORDER BY priority, id""",
                (provider,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ events
    def log_event(
        self,
        *,
        run_id: str,
        kind: str,
        message: str = "",
        post_id: int | None = None,
        scene_index: int | None = None,
        total_scenes: int | None = None,
        provider: str | None = None,
        account_id: int | None = None,
        elapsed_ms: int | None = None,
        detail_json: str | None = None,
    ) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO events
                   (run_id, post_id, ts, kind, scene_index, total_scenes,
                    provider, account_id, message, elapsed_ms, detail_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, post_id, self._now(), kind, scene_index, total_scenes,
                 provider, account_id, message, elapsed_ms, detail_json),
            )
            return int(cur.lastrowid)

    def list_events(self, run_id: str, *, after_id: int = 0) -> list[dict]:
        """Events for a run in id order. `after_id` supports incremental polling."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM events WHERE run_id = ? AND id > ?
                   ORDER BY id""",
                (run_id, after_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_runs(self, limit: int = 50) -> list[dict]:
        """
        One summary row per run_id, newest first. `latest_kind` is the most
        recent event's kind — used by the dashboard to show run status
        (running / finished / failed).
        """
        with self._lock:
            rows = self._conn.execute(
                """SELECT run_id,
                          MIN(ts)            AS started_at,
                          MAX(ts)            AS latest_ts,
                          COUNT(*)           AS event_count,
                          (SELECT kind FROM events e2
                              WHERE e2.run_id = events.run_id
                              ORDER BY e2.id DESC LIMIT 1) AS latest_kind,
                          (SELECT post_id FROM events e2
                              WHERE e2.run_id = events.run_id
                                AND e2.post_id IS NOT NULL
                              ORDER BY e2.id LIMIT 1) AS post_id
                   FROM events
                   GROUP BY run_id
                   ORDER BY started_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def current_run_id(self) -> str | None:
        """The run_id of the most recent run that hasn't reached a terminal
        state (finished / failed / cancelled)."""
        with self._lock:
            row = self._conn.execute(
                """SELECT e1.run_id FROM events e1
                   GROUP BY e1.run_id
                   HAVING (SELECT e2.kind FROM events e2
                             WHERE e2.run_id = e1.run_id
                             ORDER BY e2.id DESC LIMIT 1)
                          NOT IN ('run_finished', 'run_failed', 'run_cancelled')
                   ORDER BY MAX(e1.id) DESC LIMIT 1""",
            ).fetchone()
            return row["run_id"] if row else None

    # ------------------------------------------------------------- API logging
    def log_api_call(
        self,
        provider: str,
        status: str,
        *,
        account_id: int | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        error: str | None = None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO api_logs
                   (created_at, account_id, provider, status,
                    prompt_tokens, completion_tokens, cost_usd, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (self._now(), account_id, provider, status,
                 prompt_tokens, completion_tokens, cost_usd, error),
            )

    def get_api_usage_stats(
        self,
        start_iso: str | None = None,
        end_iso: str | None = None,
    ) -> list[dict]:
        """Get API usage aggregated by account within a time range."""
        sql = """
            SELECT
                account_id,
                provider,
                COUNT(*) as total_requests,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_requests,
                SUM(prompt_tokens) as total_prompt_tokens,
                SUM(completion_tokens) as total_completion_tokens,
                SUM(cost_usd) as total_cost_usd
            FROM api_logs
            WHERE 1=1
        """
        params = []
        if start_iso:
            sql += " AND created_at >= ?"
            params.append(start_iso)
        if end_iso:
            sql += " AND created_at <= ?"
            params.append(end_iso)
        sql += " GROUP BY account_id, provider"

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

