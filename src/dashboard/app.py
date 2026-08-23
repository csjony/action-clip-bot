"""
FastAPI dashboard application.

Mounts all routes, HTTP Basic Auth, Jinja2 templates, and static files.
Started via `python -m src.dashboard` or the `action-clip-bot-dashboard` script.
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import db_path, get_settings, get_providers_config
from src.dashboard.accounts import AccountStore
from src.dashboard.events import EventBus
from src.dashboard.runner import PipelineRunner
from src.store import Store

log = logging.getLogger("action-clip-bot.dashboard")


# ---------------------------------------------------------------------------
# Logging — wire ALL loggers (pipeline + dashboard) → data/dashboard.log
# ---------------------------------------------------------------------------
def _setup_file_logging() -> None:
    """Attach a FileHandler to the root logger so every log.info/warning/error
    from the pipeline worker thread also lands in data/dashboard.log.
    Called once at module import time so it is in place before any run starts."""
    log_path = Path("data") / "dashboard.log"
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
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


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
    spend = store.monthly_spend_usd()
    cap = settings.budget_cap_usd
    current_run = store.current_run_id()
    runs = store.list_runs(limit=5)

    # Count posts this week
    with store._lock:
        week_ago = (datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00"))
        week_count = store._conn.execute(
            "SELECT COUNT(*) AS c FROM posts WHERE created_at >= ?", (week_ago,)
        ).fetchone()["c"]
        total_posts = store._conn.execute(
            "SELECT COUNT(*) AS c FROM posts"
        ).fetchone()["c"]

    return templates.TemplateResponse(request, "index.html", {
        "spend": spend,
        "cap": cap,
        "spend_pct": min(100, (spend / cap * 100) if cap > 0 else 0),
        "current_run": current_run,
        "runs": runs,
        "week_count": week_count,
        "total_posts": total_posts,
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
    
    return templates.TemplateResponse(request, "accounts.html", {
        "accounts": all_accounts,
        "accounts_by_provider": dict(accounts_by_provider),
        "provider_groups": provider_groups,
        "provider_info": PROVIDER_INFO,
    })


@app.post("/accounts/add", response_class=HTMLResponse)
async def account_add(
    request: Request,
    provider: str = Form(...),
    label: str = Form(...),
    email: str = Form(default=""),
    api_key: str = Form(...),
    priority: int = Form(default=100),
    enabled: bool = Form(default=True),
):
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
async def runs_page(request: Request, limit: int = Query(default=50, le=200)):
    runs = store.list_runs(limit=limit)
    return templates.TemplateResponse(request, "runs.html", {
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
    all_events = store.list_events(run_id)
    # Also check runner state
    run_state = runner.get_run(run_id)
    return templates.TemplateResponse(request, "run_detail.html", {
        "run_id": run_id,
        "events": all_events,
        "run_state": run_state,
        "fmt_ts": _fmt_ts,
        "fmt_duration_ms": _fmt_duration_ms,
    })


# --- Posts ---
@app.get("/posts", response_class=HTMLResponse)
async def posts_page(request: Request, limit: int = Query(default=50, le=200)):
    with store._lock:
        rows = store._conn.execute(
            "SELECT * FROM posts ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        posts = [dict(r) for r in rows]
    return templates.TemplateResponse(request, "posts.html", {
        "posts": posts,
        "fmt_ts": _fmt_ts,
    })


# --- Generate ---
@app.get("/generate", response_class=HTMLResponse)
async def generate_page(request: Request):
    settings = get_settings()
    themes = settings.llm.get("themes", [])
    
    # Pre-render prompts for all themes
    from src.content.scriptwriter import Scriptwriter
    sw = Scriptwriter(settings)
    video_cfg = settings.video
    target_duration = video_cfg.get("target_duration_sec", 70)
    scene_count = video_cfg.get("max_clips", 12)
    
    prompts = {}
    for t in themes:
        try:
            prompts[t] = sw._render_prompt(t, scene_count, target_duration)
        except Exception as exc:
            prompts[t] = f"Error rendering prompt for theme {t}: {exc}"
            
    return templates.TemplateResponse(request, "generate.html", {
        "themes": themes,
        "prompts": prompts,
        "is_busy": runner.is_busy,
        "current_run_id": runner.current_run_id,
        "error": None,
        "script_json": "",
    })


@app.post("/generate", response_class=HTMLResponse)
async def generate_submit(
    request: Request,
    theme: str = Form(default=""),
    dry_run: bool = Form(default=False),
    quick_test: bool = Form(default=False),
    script_json: str = Form(default=""),
):
    if runner.is_busy:
        return RedirectResponse(
            url=f"/runs/{runner.current_run_id}" if runner.current_run_id else "/generate",
            status_code=303,
        )
    
    script_json_clean = script_json.strip() or None
    if not script_json_clean:
        settings = get_settings()
        themes = settings.llm.get("themes", [])
        
        from src.content.scriptwriter import Scriptwriter
        sw = Scriptwriter(settings)
        video_cfg = settings.video
        target_duration = video_cfg.get("target_duration_sec", 70)
        scene_count = video_cfg.get("max_clips", 12)
        
        prompts = {}
        for t in themes:
            try:
                prompts[t] = sw._render_prompt(t, scene_count, target_duration)
            except Exception:
                prompts[t] = f"Error rendering prompt for theme {t}"
                
        return templates.TemplateResponse(request, "generate.html", {
            "themes": themes,
            "prompts": prompts,
            "is_busy": runner.is_busy,
            "current_run_id": runner.current_run_id,
            "error": "Manual Script JSON is required to start a run.",
            "script_json": "",
        })

    # Quick validation
    try:
        import json
        text = script_json_clean
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        json.loads(text)
        script_json_clean = text # save normalized json
    except Exception as exc:
        settings = get_settings()
        themes = settings.llm.get("themes", [])
        
        from src.content.scriptwriter import Scriptwriter
        sw = Scriptwriter(settings)
        video_cfg = settings.video
        target_duration = video_cfg.get("target_duration_sec", 70)
        scene_count = video_cfg.get("max_clips", 12)
        
        prompts = {}
        for t in themes:
            try:
                prompts[t] = sw._render_prompt(t, scene_count, target_duration)
            except Exception:
                prompts[t] = f"Error rendering prompt for theme {t}"
                
        return templates.TemplateResponse(request, "generate.html", {
            "themes": themes,
            "prompts": prompts,
            "is_busy": runner.is_busy,
            "current_run_id": runner.current_run_id,
            "error": f"Invalid JSON: {exc}",
            "script_json": script_json,
        })

    state = runner.submit(
        dry_run=dry_run,
        theme=theme or None,
        quick_test=quick_test,
        script_json=script_json_clean
    )
    return RedirectResponse(url=f"/runs/{state.run_id}", status_code=303)


# --- Usage ---
@app.get("/usage", response_class=HTMLResponse)
async def usage_page(request: Request):
    providers_cfg = get_providers_config()
    provider_names = providers_cfg.get("order", [])
    from src.generators.pool import _DEFAULT_CAPS
    caps = _DEFAULT_CAPS

    # Per-provider credit usage this period
    credit_data = []
    for prov in provider_names:
        cap_info = caps.get(prov, {})
        kind = cap_info.get("kind", "daily")
        cap = cap_info.get("cap", 0)
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
        "credit_data": credit_data,
        "monthly_spend": spend,
        "budget_cap": cap,
    })


# --- Logs ---
@app.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request, source: str = "dashboard"):
    return templates.TemplateResponse(request, "logs.html", {
        "source": source,
    })


@app.get("/api/logs")
async def api_logs(source: str = "dashboard", limit: int = 500):
    if source == "dashboard":
        log_file = Path("data/dashboard.log")
        if not log_file.exists():
            return {"content": "No dashboard log file found yet."}
        try:
            # Read last N lines
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                return {"content": "".join(lines[-limit:])}
        except Exception as e:
            return {"content": f"Error reading dashboard log: {e}"}
            
    elif source == "gpu":
        runpod_key = os.environ.get("RUNPOD_API_KEY")
        runpod_id = os.environ.get("RUNPOD_POD_ID")
        if not runpod_key or not runpod_id:
            return {"content": "RunPod configuration (RUNPOD_API_KEY / RUNPOD_POD_ID) is missing in .env."}
        
        try:
            import httpx
            headers = {"Authorization": f"Bearer {runpod_key}"}
            r = httpx.get(f"https://rest.runpod.io/v1/pods/{runpod_id}", headers=headers, timeout=5.0)
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
                "-o", "ConnectTimeout=5",
                "-p", str(ssh_port),
                f"root@{public_ip}",
                f"tail -n {limit} /workspace/gpu_server.log 2>/dev/null || echo 'No gpu_server.log file found on remote volume.'"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode != 0:
                return {"content": f"Failed to connect to GPU Pod via SSH (exit code {res.returncode}):\n{res.stderr}"}
            return {"content": res.stdout}
        except Exception as e:
            return {"content": f"Error querying remote GPU server logs: {e}"}
            
    else:
        return {"content": "Invalid log source specified."}


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
    from src.generators.pool import _DEFAULT_CAPS
    caps = _DEFAULT_CAPS
    data = []
    for prov in providers_cfg.get("order", []):
        ci = caps.get(prov, {})
        kind = ci.get("kind", "daily")
        cap = ci.get("cap", 0)
        remaining = store.credit_remaining(prov, kind, cap)
        data.append({"provider": prov, "kind": kind, "cap": cap, "remaining": remaining})
    return {
        "credits": data,
        "monthly_spend": store.monthly_spend_usd(),
        "budget_cap": get_settings().budget_cap_usd,
    }
