"""
EventBus — writes progress events into the `events` table.

The dashboard's live-run view reads these back via `Store.list_events(run_id)`.
The bus is intentionally a thin shim: it doesn't queue or async-dispatch, it
just stamps the current `run_id` (held in a contextvar so callers don't have
to thread it through every call site) and writes a row synchronously.

A NullEventBus is used by default (when no dashboard is attached), so cron
runs pay only the cost of a couple of `if self._is_null` checks per clip.
"""
from __future__ import annotations

import json
import time
from contextvars import ContextVar
from typing import Any

from src.store import Store


# Event kinds. Keep stable — the dashboard's status inference keys off these.
KIND_RUN_STARTED    = "run_started"
KIND_RUN_FINISHED   = "run_finished"
KIND_RUN_FAILED     = "run_failed"
KIND_PHASE_STARTED  = "phase_started"
KIND_PHASE_FINISHED = "phase_finished"
KIND_CLIP_STARTED   = "clip_started"
KIND_CLIP_FINISHED  = "clip_finished"
KIND_PROVIDER_TRY    = "provider_try"     # about to call a provider
KIND_PROVIDER_SKIP   = "provider_skip"    # skipped before any API call (credits/budget)
KIND_PROVIDER_FAIL   = "provider_fail"    # API was called and raised
KIND_PROVIDER_OK     = "provider_ok"      # API succeeded


# run_id for the current pipeline run. Set by Pipeline.run(); cleared on exit.
_current_run_id: ContextVar[str | None] = ContextVar("action_clip_run_id", default=None)
# post_id for the current run (set once the posts row exists).
_current_post_id: ContextVar[int | None] = ContextVar("action_clip_post_id", default=None)


def set_run_context(run_id: str | None, post_id: int | None = None) -> None:
    """Called by Pipeline to scope subsequent events to this run."""
    _current_run_id.set(run_id)
    if post_id is not None:
        _current_post_id.set(post_id)


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


class EventBus:
    """Writes progress events into SQLite for the dashboard to poll."""

    def __init__(self, store: Store) -> None:
        self.store = store
        # Per (run_id, scene_index, provider) monotonic clock so events can
        # report elapsed_ms without callers doing the math.
        self._starts: dict[tuple[str, int | None, str | None], int] = {}

    # ------------------------------------------------------------------ emit
    def emit(
        self,
        kind: str,
        *,
        message: str = "",
        post_id: int | None = None,
        scene_index: int | None = None,
        total_scenes: int | None = None,
        provider: str | None = None,
        account_id: int | None = None,
        elapsed_ms: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        run_id = _current_run_id.get()
        if run_id is None:
            return  # no run in progress — silent no-op
        # Explicit post_id wins; otherwise fall back to the contextvar set by
        # Pipeline once the posts row has been created.
        effective_post_id = post_id if post_id is not None else _current_post_id.get()
        self.store.log_event(
            run_id=run_id,
            kind=kind,
            message=message,
            post_id=effective_post_id,
            scene_index=scene_index,
            total_scenes=total_scenes,
            provider=provider,
            account_id=account_id,
            elapsed_ms=elapsed_ms,
            detail_json=json.dumps(detail) if detail else None,
        )

    # ------------------------------------------------------- high-level helpers
    def provider_try(
        self, provider: str, account_id: int | None, *,
        scene_index: int | None, total_scenes: int | None,
    ) -> int:
        """Record the start of a provider attempt; return a start token (ms)."""
        self.emit(
            KIND_PROVIDER_TRY, provider=provider, account_id=account_id,
            scene_index=scene_index, total_scenes=total_scenes,
            message=f"Trying {provider}",
        )
        token = _now_ms()
        self._starts[(_current_run_id.get() or "", scene_index, provider)] = token
        return token

    def provider_elapsed(self, token: int, *, scene_index: int | None,
                         provider: str) -> int:
        """How many ms have passed since `token` (for live 'elapsed' display)."""
        return max(0, _now_ms() - token)

    def provider_skip(
        self, provider: str, account_id: int | None, reason: str, *,
        scene_index: int | None, total_scenes: int | None,
    ) -> None:
        self.emit(
            KIND_PROVIDER_SKIP, provider=provider, account_id=account_id,
            scene_index=scene_index, total_scenes=total_scenes,
            message=f"Skipped {provider}: {reason}",
            detail={"reason": reason},
        )

    def provider_fail(
        self, provider: str, account_id: int | None, exc: BaseException, *,
        scene_index: int | None, total_scenes: int | None, elapsed_ms: int,
    ) -> None:
        self.emit(
            KIND_PROVIDER_FAIL, provider=provider, account_id=account_id,
            scene_index=scene_index, total_scenes=total_scenes,
            message=f"{provider} failed: {type(exc).__name__}",
            elapsed_ms=elapsed_ms,
            detail={"error": str(exc)[:500], "exc_type": type(exc).__name__},
        )

    def provider_ok(
        self, provider: str, account_id: int | None, *,
        scene_index: int | None, total_scenes: int | None, elapsed_ms: int,
        cost_usd: float, clip_path: str,
    ) -> None:
        self.emit(
            KIND_PROVIDER_OK, provider=provider, account_id=account_id,
            scene_index=scene_index, total_scenes=total_scenes,
            message=f"{provider} succeeded",
            elapsed_ms=elapsed_ms,
            detail={"cost_usd": cost_usd, "clip_path": clip_path},
        )


class NullEventBus:
    """
    No-op stand-in used when no Store is wired (e.g. unit tests, dry imports).
    Every method is a cheap no-op so the instrumented code paths stay uniform.
    """

    def emit(self, *a, **kw) -> None: ...
    def provider_try(self, *a, **kw) -> int: return _now_ms()
    def provider_elapsed(self, *a, **kw) -> int: return 0
    def provider_skip(self, *a, **kw) -> None: ...
    def provider_fail(self, *a, **kw) -> None: ...
    def provider_ok(self, *a, **kw) -> None: ...
