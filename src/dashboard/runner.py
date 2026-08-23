"""
Background runner — enqueues Pipeline.run() in a single worker thread.

The dashboard POSTs /generate → runner.submit() → redirects to /runs/{run_id}.
The run itself happens in a ThreadPoolExecutor(max_workers=1) so only one
pipeline can be active at a time (mirrors the cron flock lock).

The UI polls /api/runs/{run_id}/events to get incremental progress.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class RunState:
    """Tracks the lifecycle of one pipeline run."""
    run_id: str
    status: str = "pending"     # pending | running | finished | failed | cancelled
    exit_code: int | None = None
    error: str | None = None
    dry_run: bool = False
    theme: str | None = None
    quick_test: bool = False
    script_json: str | None = None


class PipelineRunner:
    """Single-flight job runner — at most one pipeline at a time."""

    def __init__(self, make_pipeline: Callable[..., object]) -> None:
        """
        Args:
            make_pipeline: factory(dry_run, theme, run_id) → Pipeline instance.
            The runner calls .run() on the returned pipeline.
        """
        self._make_pipeline = make_pipeline
        self._lock = threading.Lock()
        self._current: RunState | None = None
        self._runs: dict[str, RunState] = {}
        self._run_order: list[str] = []  # insertion-ordered run_id list
        self._stop_event = threading.Event()

    @property
    def is_busy(self) -> bool:
        return self._current is not None

    @property
    def current_run_id(self) -> str | None:
        return self._current.run_id if self._current else None

    def get_run(self, run_id: str) -> RunState | None:
        return self._runs.get(run_id)

    def recent_runs(self, limit: int = 20) -> list[RunState]:
        """Most recent runs, newest first (insertion order)."""
        return [self._runs[rid] for rid in reversed(self._run_order)
                if rid in self._runs][:limit]

    def cancel(self) -> bool:
        """Signal the running pipeline to stop after the current scene.
        Returns True if a run was active, False if nothing was running."""
        with self._lock:
            if self._current is None:
                return False
            self._stop_event.set()
            self._current.status = "cancelling"
            log.info("cancel requested for run %s", self._current.run_id[:8])
            return True

    def submit(self, *, dry_run: bool = False, theme: str | None = None,
               quick_test: bool = False, script_json: str | None = None) -> RunState:
        """
        Enqueue a pipeline run. Raises RuntimeError if one is already active.
        Returns the RunState immediately (the run proceeds in the background).
        """
        with self._lock:
            if self._current is not None:
                raise RuntimeError(
                    f"A run is already in progress ({self._current.run_id}). "
                    "Wait for it to finish or check /api/status."
                )
            run_id = uuid.uuid4().hex
            state = RunState(
                run_id=run_id, status="running",
                dry_run=dry_run or quick_test,  # quick_test implies dry_run
                theme=theme,
                quick_test=quick_test,
                script_json=script_json,
            )
            self._runs[run_id] = state
            self._run_order.append(run_id)
            self._current = state
            self._stop_event.clear()  # reset from any previous cancel

        thread = threading.Thread(
            target=self._execute, args=(state,), daemon=True,
        )
        thread.start()
        log.info("submitted pipeline run %s (dry_run=%s, quick_test=%s, theme=%s, has_manual_script=%s)",
                 run_id[:8], dry_run, quick_test, theme, script_json is not None)
        return state

    def _execute(self, state: RunState) -> None:
        """Run the pipeline in this thread; update state on completion."""
        try:
            pipeline = self._make_pipeline(
                dry_run=state.dry_run, theme=state.theme, run_id=state.run_id,
                stop_event=self._stop_event, quick_test=state.quick_test,
                script_json=state.script_json,
            )
            exit_code = pipeline.run()
            # Distinguish clean stop from natural completion
            if self._stop_event.is_set():
                state.status = "cancelled"
                state.exit_code = exit_code
            else:
                state.exit_code = exit_code
                state.status = "finished" if exit_code == 0 else "failed"
        except Exception as exc:
            log.error("pipeline runner error: %s", exc)
            state.status = "failed"
            state.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._current = None
            log.info("run %s ended with status=%s", state.run_id[:8], state.status)
