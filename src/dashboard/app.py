"""
FastAPI dashboard application.

Mounts all routes, HTTP Basic Auth, Jinja2 templates, and static files.
Started via `python -m src.dashboard` or the `action-clip-bot-dashboard` script.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import PROJECT_ROOT, db_path, get_settings, get_providers_config
from src.dashboard.accounts import AccountStore
from src.dashboard.events import EventBus
from src.dashboard.runner import PipelineRunner
from src.store import Store

log = logging.getLogger("action-clip-bot.dashboard")


def _app_version() -> str:
    """Single source of truth for the footer / health endpoint."""
    # Prefer the repo's pyproject (editable source of truth) over stale
    # installed dist metadata.
    try:
        import tomllib

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as fh:
            v = str(tomllib.load(fh).get("project", {}).get("version", ""))
            if v:
                return v
    except Exception:
        pass
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("action-clip-bot")
    except Exception:
        return "1.0.0"


APP_VERSION = _app_version()


def _dash_cfg() -> dict:
    """UI tunables from settings.yaml `dashboard:` (editable on Settings page)."""
    cfg = get_settings().get("dashboard", {}) or {}
    return cfg if isinstance(cfg, dict) else {}


def _base_context(request: Request, active_page: str) -> dict:
    """Shared template vars so every page gets nav state + version for free."""
    cfg = _dash_cfg()
    return {
        "request": request,
        "active_page": active_page,
        "app_version": APP_VERSION,
        # Live JS timings — templates/JS read these instead of hardcoded ms.
        "ui": {
            "event_poll_ms": int(cfg.get("event_poll_sec", 2)) * 1000,
            "event_backoff_max_ms": int(cfg.get("event_backoff_max_sec", 15)) * 1000,
            "logs_poll_ms": int(cfg.get("logs_poll_ms", 1500)),
            "toast_ms": int(cfg.get("toast_ms", 3500)),
        },
    }


# ---------------------------------------------------------------------------
# Logging — wire ALL loggers (pipeline + dashboard) → data/dashboard.log
# ---------------------------------------------------------------------------
def _setup_file_logging() -> None:
    """Attach a FileHandler to the root logger so every log.info/warning/error
    from the pipeline worker thread also lands in data/dashboard.log.
    Called once at module import time so it is in place before any run starts."""
    log_path = PROJECT_ROOT / "data" / "dashboard.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    # Only add if we haven't already (e.g. reload in dev mode).
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    root.setLevel(logging.INFO)


_setup_file_logging()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Action Clip Bot — Dashboard", docs_url=None, redoc_url=None)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Shared state (thread-safe — Store uses threading.Lock internally).
store = Store(db_path())

def _cleanup_stale_runs(store: Store) -> None:
    """Transition any stale run (which was interrupted by server crash/restart) to failed."""
    with store._lock:
        rows = store._conn.execute("""
            SELECT run_id FROM events e1
            GROUP BY run_id
            HAVING (
                SELECT kind FROM events e2
                WHERE e2.run_id = e1.run_id
                ORDER BY e2.id DESC LIMIT 1
            ) NOT IN ('run_finished', 'run_failed', 'run_cancelled')
        """).fetchall()
        stale_run_ids = [r["run_id"] for r in rows]
        if stale_run_ids:
            log.info("Cleaning up %d stale runs: %s", len(stale_run_ids), stale_run_ids)
            now = store._now()
            for rid in stale_run_ids:
                store._conn.execute("""
                    INSERT INTO events (run_id, ts, kind, message)
                    VALUES (?, ?, 'run_failed', 'Process died or server restarted mid-run')
                """, (rid, now))

_cleanup_stale_runs(store)

acct_store = AccountStore(store)
events = EventBus(store)

# ---------------------------------------------------------------------------
# Provider metadata — signup URLs + credential hints shown in the accounts UI
# ---------------------------------------------------------------------------
PROVIDER_INFO: dict[str, dict] = {
    # ── Video generators (local / free VPS) ──────────────────────────────────
    "local": {
        "url": "https://www.runpod.io/",
        "desc": "Local / RunPod Cloud GPU Wan 2.2 API URL (e.g. http://localhost:8000 or proxy.runpod.net link)",
    },
    "colab": {
        "url": "https://colab.research.google.com/",
        "desc": "Colab tunnel URL from the notebook (https://….trycloudflare.com) — used when GPU backend is colab",
    },
    # ── Content & music helper APIs ─────────────────────────────────────────
    "gemini": {
        "url": "https://aistudio.google.com/app/apikey",
        "desc": "Google Gemini · free tier · no CC",
    },
    "groq": {
        "url": "https://console.groq.com/keys",
        "desc": "Groq · free tier (Llama 3) · no CC",
    },
    "jamendo": {
        "url": "https://developer.jamendo.com/v3.0",
        "desc": "Jamendo Developer API · Royalty-Free Music by Theme · free client_id",
    },
    "freesound": {
        "url": "https://freesound.org/apiv2/apply/",
        "desc": "Freesound API · Royalty-Free Sound Effects & Ambient Previews · free API key",
    },
    "elevenlabs": {
        "url": "https://elevenlabs.io/",
        "desc": "ElevenLabs API · Premium AI-Generated Sound Effects · API key",
    },
}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _check_auth(request: Request) -> bool:
    """Return True if the request carries valid credentials."""
    user = os.environ.get("DASHBOARD_USER", "")
    pw = os.environ.get("DASHBOARD_PASSWORD", "")
    if not user and not pw:
        return True  # no auth configured — open access
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    import base64
    try:
        decoded = base64.b64decode(auth[6:]).decode("utf-8")
        provided_user, provided_pw = decoded.split(":", 1)
    except Exception:
        return False
    # Timing-safe compare
    return (
        hashlib.sha256(provided_user.encode()).hexdigest()
        == hashlib.sha256(user.encode()).hexdigest()
        and hashlib.sha256(provided_pw.encode()).hexdigest()
        == hashlib.sha256(pw.encode()).hexdigest()
    )


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Skip auth for static files and favicon
    if request.url.path.startswith("/static") or request.url.path == "/favicon.ico":
        return await call_next(request)
    if not _check_auth(request):
        # Browsers hitting a page get a native login prompt; API/JS callers
        # get JSON so fetch() handlers can show a toast instead of HTML.
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            return HTMLResponse(
                content="<h1>401 Unauthorized</h1><p>Dashboard login required.</p>",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Action Clip Bot"'},
            )
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized"},
            headers={"WWW-Authenticate": 'Basic realm="Action Clip Bot"'},
        )
    response = await call_next(request)
    # Minimal hardening headers (no external deps).
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


@app.get("/healthz", include_in_schema=False)
async def healthz():
    """Cheap liveness probe for docker / uptime monitors (no auth needed)."""
    return {"status": "ok", "version": APP_VERSION, "busy": runner.is_busy if "runner" in globals() else False}


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return templates.TemplateResponse(
            request, "error.html",
            {**_base_context(request, ""), "code": 404,
             "title": "Page not found",
             "message": f"No route matches {request.url.path}."},
            status_code=404,
        )
    return JSONResponse(status_code=404, content={"detail": "Not found"})


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    log.exception("unhandled dashboard error for %s", request.url.path)
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return templates.TemplateResponse(
            request, "error.html",
            {**_base_context(request, ""), "code": 500,
             "title": "Something went wrong",
             "message": "The dashboard hit an unexpected error. Check Logs for details."},
            status_code=500,
        )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def _make_pipeline(*, dry_run: bool = False, theme: str | None = None,
                   run_id: str | None = None, stop_event=None,
                   quick_test: bool = False, script_json: str | None = None):
    from src.pipeline import Pipeline
    return Pipeline(dry_run=dry_run, theme=theme, run_id=run_id,
                    stop_event=stop_event, quick_test=quick_test,
                    script_json=script_json)


runner = PipelineRunner(_make_pipeline)


# ---------------------------------------------------------------------------
# Helper: format durations / timestamps
# ---------------------------------------------------------------------------
def _fmt_ts(iso: str) -> str:
    """ISO → human-readable short format."""
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso or "—"


def _fmt_duration_ms(ms: int | None) -> str:
    if ms is None:
        return ""
    s = ms / 1000
    if s < 60:
        return f"{s:.1f}s"
    return f"{s // 60:.0f}m {s % 60:.0f}s"


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    settings = get_settings()
    try:
        spend = store.monthly_spend_usd()
    except Exception:
        spend = 0.0
    cap = settings.budget_cap_usd
    try:
        current_run = store.current_run_id()
    except Exception:
        current_run = None
    try:
        runs = store.list_runs(limit=int(_dash_cfg().get("index_runs", 5)))
    except Exception:
        log.exception("failed to list runs")
        runs = []

    # Count posts this week — degrade gracefully if the DB is locked/busy.
    try:
        with store._lock:
            week_ago = (datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00"))
            week_count = store._conn.execute(
                "SELECT COUNT(*) AS c FROM posts WHERE created_at >= ?", (week_ago,)
            ).fetchone()["c"]
            total_posts = store._conn.execute(
                "SELECT COUNT(*) AS c FROM posts"
            ).fetchone()["c"]
    except Exception:
        log.exception("failed to count posts")
        week_count, total_posts = 0, 0

    # GSC-style activity overview: runs per day for the last 14 days.
    try:
        from datetime import timedelta

        today = datetime.now(timezone.utc).date()
        window = max(2, int(_dash_cfg().get("activity_days", 14)))
        days = [today - timedelta(days=i) for i in range(window - 1, -1, -1)]
        counts = {d.isoformat(): 0 for d in days}
        for r in store.list_runs(limit=int(_dash_cfg().get("activity_runs", 500))):
            started = (r.get("started_at") or "")[:10]
            if started in counts:
                counts[started] += 1
        activity = [
            {"label": d.strftime("%d %b"), "count": counts[d.isoformat()]}
            for d in days
        ]
        activity_max = max([a["count"] for a in activity] + [1])
        activity_total = sum(a["count"] for a in activity)
    except Exception:
        log.exception("failed to build activity overview")
        activity, activity_max, activity_total = [], 1, 0

    return templates.TemplateResponse(request, "index.html", {
        **_base_context(request, "dashboard"),
        "spend": spend,
        "cap": cap,
        "spend_pct": min(100, (spend / cap * 100) if cap > 0 else 0),
        "current_run": current_run,
        "runs": runs,
        "week_count": week_count,
        "total_posts": total_posts,
        "activity": activity,
        "activity_max": activity_max,
        "activity_total": activity_total,
    })


# --- Accounts ---
@app.get("/accounts", response_class=HTMLResponse)
async def accounts_page(request: Request):
    all_accounts = acct_store.list_all()
    providers_cfg = get_providers_config()
    providers_dict = providers_cfg.get("providers", {})
    
    free_gens = []
    paid_gens = []
    for p, spec in providers_dict.items():
        if spec.get("free", True):
            free_gens.append(p)
        else:
            paid_gens.append(p)
            
    order = providers_cfg.get("order", [])
    video_gens = sorted(free_gens + paid_gens, key=lambda x: order.index(x) if x in order else 999)
    
    provider_groups = {
        "Video Generators": video_gens,
        "Content & Music Helper APIs": ["gemini", "groq", "jamendo", "freesound", "elevenlabs"]
    }
    
    from collections import defaultdict
    accounts_by_provider = defaultdict(list)
    for a in all_accounts:
        accounts_by_provider[a["provider"]].append(a)
    
    cfg = _dash_cfg()
    return templates.TemplateResponse(request, "accounts.html", {
        **_base_context(request, "accounts"),
        "accounts": all_accounts,
        "accounts_by_provider": dict(accounts_by_provider),
        "provider_groups": provider_groups,
        "provider_info": PROVIDER_INFO,
        "api_key_max": int(cfg.get("api_key_max", 4000)),
        "label_max": int(cfg.get("label_max", 64)),
        "email_max": int(cfg.get("email_max", 254)),
    })


@app.post("/accounts/add", response_class=HTMLResponse)
async def account_add(
    request: Request,
    provider: str = Form(...),
    label: str = Form(...),
    email: str = Form(default=""),
    api_key: str = Form(...),
    priority: int | None = Form(default=None),
    enabled: bool = Form(default=True),
):
    cfg = _dash_cfg()
    if priority is None:
        priority = int(cfg.get("priority_default", 100))
    provider = (provider or "").strip().lower()[:int(cfg.get("label_max", 64))]
    label = (label or "").strip()[:int(cfg.get("label_max", 64))]
    email = (email or "").strip()[:int(cfg.get("email_max", 254))]
    api_key = (api_key or "").strip()
    if not provider or not label or not api_key:
        raise HTTPException(400, "Provider, label and API key are all required.")
    if len(api_key) > int(cfg.get("api_key_max", 4000)):
        raise HTTPException(400, f"API key is too long (max {cfg.get('api_key_max', 4000)} chars).")
    if priority < int(cfg.get("priority_min", 0)) or priority > int(cfg.get("priority_max", 9999)):
        raise HTTPException(400, "Priority out of allowed range.")
    acct_store.add(provider, label, api_key, email=email or None, extra=None,
                   enabled=enabled, priority=priority)
    # Clear providers config cache so the pool picks up new accounts
    from src.config import get_providers_config
    get_providers_config.cache_clear()
    return RedirectResponse(url="/accounts", status_code=303)


@app.post("/accounts/{account_id}/toggle", response_class=JSONResponse)
async def account_toggle(account_id: int):
    acct = acct_store.get(account_id)
    if not acct:
        raise HTTPException(404, "Account not found")
    new_state = not bool(acct["enabled"])
    acct_store.update(account_id, enabled=new_state)
    return {"enabled": new_state}


@app.post("/accounts/{account_id}/delete", response_class=JSONResponse)
async def account_delete(account_id: int):
    ok = acct_store.delete(account_id)
    if not ok:
        raise HTTPException(404, "Account not found")
    return {"deleted": True}


# --- Runs ---
@app.get("/runs", response_class=HTMLResponse)
async def runs_page(request: Request,
                    limit: int = Query(default=50, le=200)):
    cfg = _dash_cfg()
    limit = max(1, min(limit, int(cfg.get("history_max", 200))))
    runs = store.list_runs(limit=limit)
    return templates.TemplateResponse(request, "runs.html", {
        **_base_context(request, "runs"),
        "runs": runs,
        "is_busy": runner.is_busy,
        "current_run_id": runner.current_run_id,
    })


@app.post("/runs/stop", response_class=JSONResponse)
async def runs_stop():
    """Request the active pipeline run to stop cleanly after its current scene."""
    cancelled = runner.cancel()
    if cancelled:
        return {"ok": True, "message": "Stop signal sent — run will halt after the current scene."}
    return {"ok": False, "message": "No run is currently active."}


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str):
    run_id = (run_id or "").strip()[:int(_dash_cfg().get("run_id_max", 64))]
    all_events = store.list_events(run_id)
    # Also check runner state
    run_state = runner.get_run(run_id)
    return templates.TemplateResponse(request, "run_detail.html", {
        **_base_context(request, "runs"),
        "run_id": run_id,
        "events": all_events,
        "run_state": run_state,
        "fmt_ts": _fmt_ts,
        "fmt_duration_ms": _fmt_duration_ms,
    })


# --- Posts ---
@app.get("/posts", response_class=HTMLResponse)
async def posts_page(request: Request,
                     limit: int = Query(default=50, le=200)):
    limit = max(1, min(limit, int(_dash_cfg().get("history_max", 200))))
    try:
        with store._lock:
            rows = store._conn.execute(
                "SELECT * FROM posts ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            posts = [dict(r) for r in rows]
    except Exception:
        log.exception("failed to list posts")
        posts = []
    return templates.TemplateResponse(request, "posts.html", {
        **_base_context(request, "posts"),
        "posts": posts,
        "fmt_ts": _fmt_ts,
    })


def _llm_status() -> tuple[dict, bool]:
    """Configured LLM keys per provider + whether any scriptwriter path works."""
    status: dict = {}
    for prov, env_key in (("gemini", "GEMINI_API_KEY"), ("groq", "GROQ_API_KEY")):
        try:
            keys = sum(1 for a in acct_store.list_all(prov) if a.get("enabled"))
        except Exception:
            keys = 0
        env = bool(os.environ.get(env_key))
        status[prov] = {"keys": keys, "env": env}
    ready = any(v["keys"] or v["env"] for v in status.values())
    return status, ready


def _generate_error(request: Request, error: str, theme: str):
    settings = get_settings()
    themes = settings.llm.get("themes", [])
    llm_status, llm_ready = _llm_status()
    return templates.TemplateResponse(request, "generate.html", {
        **_base_context(request, "generate"),
        "themes": themes,
        "selected_theme": theme,
        "llm_status": llm_status,
        "llm_ready": llm_ready,
        "is_busy": runner.is_busy,
        "current_run_id": runner.current_run_id,
        "error": error,
    })


# --- Generate ---
@app.get("/generate", response_class=HTMLResponse)
async def generate_page(request: Request):
    settings = get_settings()
    themes = settings.llm.get("themes", [])
    llm_status, llm_ready = _llm_status()
    return templates.TemplateResponse(request, "generate.html", {
        **_base_context(request, "generate"),
        "themes": themes,
        "selected_theme": themes[0] if themes else "",
        "llm_status": llm_status,
        "llm_ready": llm_ready,
        "is_busy": runner.is_busy,
        "current_run_id": runner.current_run_id,
        "error": None,
    })


@app.post("/generate", response_class=HTMLResponse)
async def generate_submit(
    request: Request,
    theme: str = Form(default=""),
    quick_test: bool = Form(default=False),
):
    if runner.is_busy:
        return RedirectResponse(
            url=f"/runs/{runner.current_run_id}" if runner.current_run_id else "/generate",
            status_code=303,
        )

    theme = (theme or "").strip()
    themes = get_settings().llm.get("themes", [])
    if not theme or (themes and theme not in themes):
        return _generate_error(request, "Please choose a valid theme.", theme)

    state = runner.submit(
        dry_run=False,
        theme=theme,
        quick_test=quick_test,
        script_json=None,
    )
    return RedirectResponse(url=f"/runs/{state.run_id}", status_code=303)


# --- Usage ---
@app.get("/usage", response_class=HTMLResponse)
async def usage_page(request: Request):
    providers_cfg = get_providers_config()
    provider_names = providers_cfg.get("order", [])
    from src.generators.pool import _credit_spec

    # Per-provider credit usage this period
    credit_data = []
    for prov in provider_names:
        kind, cap = _credit_spec(providers_cfg, prov)
        remaining = store.credit_remaining(prov, kind, cap)
        used = cap - remaining
        accounts = acct_store.list_all(prov)
        credit_data.append({
            "provider": prov,
            "kind": kind,
            "cap": cap,
            "remaining": remaining,
            "used": used,
            "accounts": accounts,
        })

    spend = store.monthly_spend_usd()
    cap = get_settings().budget_cap_usd
    return templates.TemplateResponse(request, "usage.html", {
        **_base_context(request, "usage"),
        "credit_data": credit_data,
        "monthly_spend": spend,
        "budget_cap": cap,
    })


# --- Logs ---
@app.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request, source: str = "dashboard"):
    source = source if source in ("dashboard", "gpu") else "dashboard"
    return templates.TemplateResponse(request, "logs.html", {
        **_base_context(request, "logs"),
        "source": source,
    })


@app.get("/api/logs")
async def api_logs(source: str = "dashboard", limit: int = 500):
    cfg = _dash_cfg()
    limit = max(int(cfg.get("log_min", 50)), min(limit, int(cfg.get("log_max", 2000))))
    if source == "dashboard":
        log_file = PROJECT_ROOT / "data" / "dashboard.log"
        if not log_file.exists():
            return {"content": "No dashboard log file found yet."}
        try:
            return {"content": _tail_file(log_file, limit)}
        except Exception as e:
            return {"content": f"Error reading dashboard log: {e}"}

    elif source == "gpu":
        # Cached: the GPU fetch does RunPod HTTPS + SSH (up to ~15s when the
        # pod is dead). Without caching, the 1.5s auto-refresh piles up
        # blocked threads until the whole dashboard hangs.
        return _cached_gpu_logs(limit, cfg)

    else:
        return {"content": "Invalid log source specified."}


def _tail_file(path: Path, max_lines: int, chunk_bytes: int = 65536) -> str:
    """Read the last *max_lines* of a file without loading it whole.

    Seeks from the end in binary chunks, so a multi-MB dashboard.log costs
    milliseconds instead of a full read + split on every 1.5s poll.
    """
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        data = b""
        lines: list[bytes] = []
        while size > 0 and len(lines) <= max_lines:
            step = min(chunk_bytes, size)
            size -= step
            fh.seek(size)
            data = fh.read(step) + data
            lines = data.split(b"\n")
        tail = lines[-max_lines:] if len(lines) > max_lines else lines
    return b"\n".join(tail).decode("utf-8", errors="ignore")[-200_000:]


_gpu_log_cache: dict = {"at": 0.0, "limit": 0, "content": ""}
_gpu_log_lock = threading.Lock()


def _cached_gpu_logs(limit: int, cfg: dict) -> dict:
    """TTL-cached GPU log fetch with singleflight.

    Concurrent polls share one in-flight fetch and reuse the result for
    `gpu_log_ttl_sec` (default 10s), so a dead pod can never pile up
    blocked threads faster than the cache expires.
    """
    import time as _time

    ttl = float(cfg.get("gpu_log_ttl_sec", 10))
    now = _time.monotonic()
    with _gpu_log_lock:
        if now - _gpu_log_cache["at"] < ttl and _gpu_log_cache["limit"] >= limit:
            return {"content": _gpu_log_cache["content"], "cached": True}
        # Singleflight: hold the lock for the whole fetch so concurrent
        # polls wait on one fetch instead of each starting their own.
        content = _fetch_gpu_logs(limit, cfg)["content"]
        _gpu_log_cache.update(at=_time.monotonic(), limit=limit, content=content)
        return {"content": content, "cached": False}


def _fetch_gpu_logs(limit: int, cfg: dict) -> dict:
    runpod_key = os.environ.get("RUNPOD_API_KEY")
    runpod_id = os.environ.get("RUNPOD_POD_ID")
    if not runpod_key or not runpod_id:
        return {"content": "RunPod configuration (RUNPOD_API_KEY / RUNPOD_POD_ID) is missing in .env."}

    try:
        import httpx
        headers = {"Authorization": f"Bearer {runpod_key}"}
        r = httpx.get(f"https://rest.runpod.io/v1/pods/{runpod_id}", headers=headers,
                      timeout=float(cfg.get("log_fetch_timeout_sec", 5.0)))
        if r.status_code != 200:
            return {"content": f"Failed to fetch pod details from RunPod (status {r.status_code})."}
        pod_info = r.json()
        if pod_info.get("desiredStatus") != "RUNNING":
            return {"content": f"GPU pod status is {pod_info.get('desiredStatus', 'UNKNOWN')} (not running)."}

        public_ip = pod_info.get("publicIp")
        ports = pod_info.get("portMappings", {})
        ssh_port = ports.get("22/tcp") or ports.get("22")
        if not public_ip or not ssh_port:
            return {"content": "GPU pod is starting but SSH port/IP is not assigned yet."}

        # Execute SSH tail command
        import subprocess
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={cfg.get('log_ssh_timeout_sec', 5)}",
            "-p", str(ssh_port),
            f"root@{public_ip}",
            f"tail -n {limit} /workspace/gpu_server.log 2>/dev/null || echo 'No gpu_server.log file found on remote volume.'"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=int(cfg.get("log_ssh_cmd_sec", 10)))
        if res.returncode != 0:
            return {"content": f"Failed to connect to GPU Pod via SSH (exit code {res.returncode}):\n{res.stderr}"}
        return {"content": res.stdout}
    except Exception as e:
        return {"content": f"Error querying remote GPU server logs: {e}"}


# ---------------------------------------------------------------------------
# Settings — every tunable in one place (no code edits needed)
# ---------------------------------------------------------------------------
# Fields whose live value can be forced by an env var (file/UI value loses).
ENV_WINS = {
    "budget.monthly_cap_usd": "BUDGET_CAP",
    "whisper.model": "WHISPER_MODEL",
    "runpod.default_model": "WAN_MODEL_ID",
}


def _settings_file_defaults() -> dict:
    """settings.yaml values WITHOUT dashboard overrides (for the UI)."""
    from src.config import _load_yaml
    from src.dashboard.settings_spec import all_keys
    raw = _load_yaml("settings.yaml")
    out: dict = {}
    for key in all_keys():
        node = raw
        ok = True
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                ok = False
                break
        if ok:
            out[key] = node
    return out


def _settings_view_model() -> list[dict]:
    """Groups with per-field file/effective values + overridden flags."""
    from src.dashboard.settings_spec import SPEC
    file_defaults = _settings_file_defaults()
    try:
        overridden = {r["key"] for r in store.list_overrides()}
    except Exception:
        overridden = set()
    settings = get_settings()
    groups = []
    for g in SPEC:
        fields = []
        for f in g["fields"]:
            spec_default = f.get("default")
            file_default = file_defaults.get(f["key"], spec_default)
            effective = settings.get(f["key"], file_default)
            fields.append({
                **f,
                "file_default": file_default,
                "effective": effective,
                "overridden": f["key"] in overridden,
                "env_var": ENV_WINS.get(f["key"]),
            })
        groups.append({"group": g["group"], "desc": g.get("desc", ""), "fields": fields})
    return groups


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, saved: int = Query(default=0),
                        error: str = Query(default="")):
    return templates.TemplateResponse(request, "settings.html", {
        **_base_context(request, "settings"),
        "groups": _settings_view_model(),
        "saved": saved,
        "error": error or None,
    })


@app.post("/settings/save", response_class=HTMLResponse)
async def settings_save(request: Request):
    from src.dashboard.settings_spec import SPEC
    import json as _json
    form = await request.form()
    file_defaults = _settings_file_defaults()
    errors: list[str] = []
    changed = 0
    for g in SPEC:
        for f in g["fields"]:
            key = f["key"]
            ftype = f.get("type", "text")
            raw_val = form.get(f"v:{key}")
            try:
                if ftype == "bool":
                    # Unchecked checkboxes are absent from the form. Compare
                    # against the rendered current state (cur:<key>) so an
                    # untouched toggle never writes a spurious override —
                    # this holds even without the page's JS dirtiness filter.
                    # (With JS, both v: and cur: are disabled when untouched.)
                    submitted = raw_val in ("on", "true", "1", "yes")
                    current_raw = form.get(f"cur:{key}")
                    if current_raw is None:
                        if raw_val is None:
                            continue
                        value = submitted
                    else:
                        current = current_raw in ("on", "true", "1", "yes", "True")
                        if submitted == current:
                            continue
                        value = submitted
                elif raw_val is None or str(raw_val).strip() == "":
                    continue  # untouched — leave as-is
                elif ftype == "int":
                    value = int(str(raw_val).strip())
                elif ftype == "float":
                    value = float(str(raw_val).strip())
                elif ftype == "list":
                    value = [ln.strip() for ln in str(raw_val).splitlines() if ln.strip()]
                else:
                    value = str(raw_val).strip()
                    if ftype == "select" and f.get("options") and value not in f["options"]:
                        raise ValueError(f"must be one of: {', '.join(f['options'])}")
                if ftype in ("int", "float"):
                    if f.get("min") is not None and value < f["min"]:
                        raise ValueError(f"must be ≥ {f['min']}")
                    if f.get("max") is not None and value > f["max"]:
                        raise ValueError(f"must be ≤ {f['max']}")
            except (ValueError, TypeError) as exc:
                errors.append(f"{f.get('label', key)}: {exc}")
                continue
            file_default = file_defaults.get(key, f.get("default"))
            if value == file_default:
                if store.delete_override(key):
                    changed += 1
            else:
                store.set_override(key, _json.dumps(value))
                changed += 1
    get_settings.cache_clear()
    if errors:
        return RedirectResponse(url="/settings?error=" + "; ".join(errors)[:300], status_code=303)
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@app.post("/settings/reset", response_class=HTMLResponse)
async def settings_reset(request: Request):
    """Delete overrides: scope=all, or scope=group:<name>, or a single key."""
    form = await request.form()
    scope = str(form.get("scope", "all"))
    from src.dashboard.settings_spec import SPEC
    if scope == "all":
        keys = [f["key"] for g in SPEC for f in g["fields"]]
    elif scope.startswith("group:"):
        name = scope[len("group:"):]
        keys = [f["key"] for g in SPEC if g["group"] == name for f in g["fields"]]
    else:
        keys = [scope]
    for k in keys:
        store.delete_override(k)
    get_settings.cache_clear()
    return RedirectResponse(url="/settings?saved=1", status_code=303)


# ---------------------------------------------------------------------------
# GPU backend (Default Gpu menu) + Colab Code menu
# ---------------------------------------------------------------------------
def _gpu_view_model() -> dict:
    """Current backend selection + connection facts for the Default Gpu page."""
    from src.config import _load_yaml
    raw = _load_yaml("settings.yaml")

    def file_default(key, fallback):
        node = raw
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return fallback
        return node

    settings = get_settings()
    try:
        overridden = {r["key"] for r in store.list_overrides()}
    except Exception:
        overridden = set()
    backend = str(settings.get("gpu.backend", "runpod"))
    colab_url = str(settings.get("gpu.colab_url", "") or "")
    return {
        "backend": backend,
        "backend_default": file_default("gpu.backend", "runpod"),
        "backend_custom": "gpu.backend" in overridden,
        "colab_url": colab_url,
        "colab_url_default": file_default("gpu.colab_url", ""),
        "colab_url_custom": "gpu.colab_url" in overridden,
        "colab_env": bool(os.environ.get("COLAB_TUNNEL_URL")),
        "runpod_key": bool(os.environ.get("RUNPOD_API_KEY")),
        "runpod_pod": bool(os.environ.get("RUNPOD_POD_ID")),
    }


@app.get("/gpu", response_class=HTMLResponse)
async def gpu_page(request: Request, saved: int = Query(default=0),
                   error: str = Query(default="")):
    return templates.TemplateResponse(request, "gpu.html", {
        **_base_context(request, "gpu"),
        "gpu": _gpu_view_model(),
        "saved": saved,
        "error": error or None,
    })


@app.post("/gpu/save", response_class=HTMLResponse)
async def gpu_save(request: Request):
    import json as _json
    from src.config import _load_yaml
    form = await request.form()
    backend = str(form.get("backend", "")).strip().lower()
    colab_url = str(form.get("colab_url", "")).strip().rstrip("/")
    if backend not in ("runpod", "colab"):
        return RedirectResponse(url="/gpu?error=Pick+runpod+or+colab.", status_code=303)
    if backend == "colab" and colab_url and not colab_url.startswith("http"):
        return RedirectResponse(url="/gpu?error=Tunnel+URL+must+start+with+http.", status_code=303)
    raw = _load_yaml("settings.yaml")
    if colab_url == raw.get("gpu", {}).get("colab_url", ""):
        store.delete_override("gpu.colab_url")
    else:
        store.set_override("gpu.colab_url", _json.dumps(colab_url))
    if backend == raw.get("gpu", {}).get("backend", "runpod"):
        store.delete_override("gpu.backend")
    else:
        store.set_override("gpu.backend", _json.dumps(backend))
    get_settings.cache_clear()
    return RedirectResponse(url="/gpu?saved=1", status_code=303)


@app.post("/api/gpu/test", response_class=JSONResponse)
async def gpu_test():
    """Live connection check for the selected backend (short timeouts)."""
    import time as _time
    import httpx
    vm = _gpu_view_model()
    t0 = _time.monotonic()
    if vm["backend"] == "colab":
        if not vm["colab_url"]:
            return {"ok": False, "message": "No tunnel URL set. Run the notebook and paste it above."}
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(f"{vm['colab_url']}/ready")
            ms = int((_time.monotonic() - t0) * 1000)
            if r.status_code == 200:
                return {"ok": True, "message": f"Colab server is ready ({ms}ms)."}
            if r.status_code == 503:
                try:
                    detail = r.json().get("detail", "")
                except Exception:
                    detail = ""
                return {"ok": False,
                        "message": f"Model still loading: {detail or 'warming up'}."}
            return {"ok": False, "message": f"Tunnel answered HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as exc:
            return {"ok": False, "message": f"Unreachable: {exc}. Is the notebook + tunnel running?"}
    # runpod: report API + pod state (no pod start here)
    if not (vm["runpod_key"] and vm["runpod_pod"]):
        return {"ok": False, "message": "RUNPOD_API_KEY / RUNPOD_POD_ID missing in .env."}
    try:
        from src.generators.runpod_manager import RunPodManager
        status = RunPodManager(
            os.environ["RUNPOD_API_KEY"], os.environ["RUNPOD_POD_ID"]).get_status()
        ms = int((_time.monotonic() - t0) * 1000)
        if status == "RUNNING":
            return {"ok": True, "message": f"Pod is RUNNING ({ms}ms). Bootstrapping happens at run start."}
        return {"ok": False, "message": f"Pod status: {status}. It will auto-start on the next run."}
    except Exception as exc:
        return {"ok": False, "message": f"RunPod API error: {exc}"}


def _render_nb_markdown(src: str) -> str:
    """Tiny markdown renderer for exactly what our notebook uses.

    Supports: # / ## headings, **bold**, `code`, ``` fences, - lists,
    > quotes, [text](url) links. Everything else is escaped plain text.
    """
    import html as _html
    import re as _re

    lines = src.split("\n")
    out: list[str] = []
    in_fence = False
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for line in lines:
        if line.strip().startswith("```"):
            close_list()
            out.append("<pre class='nb-code'>" if not in_fence else "</pre>")
            in_fence = not in_fence
            continue
        if in_fence:
            out.append(_html.escape(line))
            continue
        s = line.strip()
        if not s:
            close_list()
            continue
        if s.startswith("## "):
            close_list()
            out.append(f"<h4>{_html.escape(s[3:])}</h4>")
            continue
        if s.startswith("# "):
            close_list()
            out.append(f"<h3>{_html.escape(s[2:])}</h3>")
            continue
        if s.startswith("- "):
            if not in_list:
                out.append("<ul class='nb-list'>")
                in_list = True
            item = _html.escape(s[2:])
            item = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", item)
            item = _re.sub(r"`(.+?)`", r"<code>\1</code>", item)
            out.append(f"<li>{item}</li>")
            continue
        close_list()
        if s.startswith("> "):
            s = s[2:]
        esc = _html.escape(s)
        esc = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc)
        esc = _re.sub(r"`(.+?)`", r"<code>\1</code>", esc)
        esc = _re.sub(r"\[(.+?)\]\((.+?)\)", r"<a href='\2' target='_blank' rel='noopener'>\1</a>", esc)
        out.append(f"<p>{esc}</p>")
    close_list()
    if in_fence:
        out.append("</pre>")
    return "\n".join(out)


def _load_notebook_cells() -> list[dict]:
    """Parse colab/ActionClipBot_Colab.ipynb into render-ready cells."""
    import json as _json
    from src.config import PROJECT_ROOT
    nb = _json.loads((PROJECT_ROOT / "colab" / "ActionClipBot_Colab.ipynb").read_text())
    cells = []
    for i, c in enumerate(nb.get("cells", [])):
        src = "".join(c.get("source", []))
        kind = c.get("cell_type", "code")
        cells.append({
            "n": i + 1,
            "kind": kind,
            "source": src,
            "html": _render_nb_markdown(src) if kind == "markdown" else "",
        })
    return cells


@app.get("/colab-code", response_class=HTMLResponse)
async def colab_code_page(request: Request):
    try:
        cells = _load_notebook_cells()
        error = None
    except Exception as exc:
        cells, error = [], f"Could not load notebook: {exc}"
    return templates.TemplateResponse(request, "colab_code.html", {
        **_base_context(request, "colab"),
        "cells": cells,
        "error": error,
    })


@app.get("/colab-code/download", include_in_schema=False)
async def colab_code_download():
    from fastapi.responses import FileResponse
    from src.config import PROJECT_ROOT
    path = PROJECT_ROOT / "colab" / "ActionClipBot_Colab.ipynb"
    if not path.exists():
        raise HTTPException(404, "Notebook not found")
    return FileResponse(path, filename="ActionClipBot_Colab.ipynb")


# ---------------------------------------------------------------------------
# API endpoints (JSON — used by JS polling on the run detail page)
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def api_status():
    return {
        "busy": runner.is_busy,
        "current_run_id": runner.current_run_id,
    }


@app.get("/api/runs/{run_id}/events")
async def api_run_events(run_id: str, after_id: int = Query(default=0)):
    """Incremental event polling endpoint. Returns events after `after_id`."""
    evts = store.list_events(run_id, after_id=after_id)
    run_state = runner.get_run(run_id)
    return {
        "run_id": run_id,
        "status": run_state.status if run_state else "unknown",
        "events": [{
            "id": e["id"],
            "kind": e["kind"],
            "message": e.get("message", ""),
            "provider": e.get("provider"),
            "scene_index": e.get("scene_index"),
            "total_scenes": e.get("total_scenes"),
            "elapsed_ms": e.get("elapsed_ms"),
            "detail": e.get("detail_json"),
            "ts": e.get("ts", ""),
        } for e in evts],
    }


@app.get("/api/accounts")
async def api_accounts(provider: str | None = Query(default=None)):
    return {"accounts": acct_store.list_all(provider)}


@app.get("/api/usage/stats")
async def api_usage_stats(
    range: str = Query(default="24h"),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    start_iso = None
    end_iso = None
    
    if range == "24h":
        start_iso = (now - timedelta(hours=24)).isoformat(timespec="seconds")
    elif range == "7d":
        start_iso = (now - timedelta(days=7)).isoformat(timespec="seconds")
    elif range == "30d" or range == "monthly":
        start_iso = (now - timedelta(days=30)).isoformat(timespec="seconds")
    elif range == "custom":
        if start:
            start_iso = f"{start}T00:00:00"
        if end:
            end_iso = f"{end}T23:59:59"
            
    stats_rows = store.get_api_usage_stats(start_iso, end_iso)
    
    stats_by_acc = {}
    for r in stats_rows:
        key = r["account_id"]
        stats_by_acc[key] = {
            "total_requests": r["total_requests"],
            "success_requests": r["success_requests"],
            "total_prompt_tokens": r["total_prompt_tokens"] or 0,
            "total_completion_tokens": r["total_completion_tokens"] or 0,
            "total_cost_usd": r["total_cost_usd"] or 0.0,
        }
        
    all_accounts = acct_store.list_all()
    accounts_stats = []
    
    for acc in all_accounts:
        acc_id = acc["id"]
        stat = stats_by_acc.get(acc_id, {
            "total_requests": 0,
            "success_requests": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost_usd": 0.0,
        })
        accounts_stats.append({
            "account_id": acc_id,
            "provider": acc["provider"],
            "label": acc["label"],
            "email": acc["email"],
            "enabled": acc["enabled"],
            "stats": stat,
        })
        
    fallback_stat = stats_by_acc.get(None)
    if fallback_stat:
        accounts_stats.append({
            "account_id": None,
            "provider": "env_fallback",
            "label": "Environment Fallback API Keys",
            "email": "env",
            "enabled": True,
            "stats": fallback_stat,
        })
        
    return {
        "range": range,
        "start_iso": start_iso,
        "end_iso": end_iso,
        "accounts_stats": accounts_stats,
    }


@app.get("/api/usage")
async def api_usage():
    providers_cfg = get_providers_config()
    from src.generators.pool import _credit_spec
    data = []
    for prov in providers_cfg.get("order", []):
        kind, cap = _credit_spec(providers_cfg, prov)
        remaining = store.credit_remaining(prov, kind, cap)
        data.append({"provider": prov, "kind": kind, "cap": cap, "remaining": remaining})
    return {
        "credits": data,
        "monthly_spend": store.monthly_spend_usd(),
        "budget_cap": get_settings().budget_cap_usd,
    }
