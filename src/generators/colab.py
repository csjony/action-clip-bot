"""
Colab backend — a fully SEPARATE implementation from the RunPod flow.

Nothing here is shared with local.py at runtime (only stdlib + httpx +
config helpers). The RunPod provider in local.py is intentionally untouched:
when `gpu.backend` is "runpod" this module is never even instantiated.

How it works:
  1. The user runs colab/ActionClipBot_Colab.ipynb and pastes the printed
     Cloudflare tunnel URL into the dashboard (Default Gpu menu) or the
     COLAB_TUNNEL_URL env var.
  2. This generator polls /ready (long wait — first boot installs deps and
     downloads weights), POSTs /generate jobs, polls /status and downloads
     /result — the same async job API gpu_server.py exposes.
  3. Keep-alive: the tight /status cadence (gpu.colab_poll_sec, default 3s)
     means constant HTTP traffic, and every request is logged to the
     notebook's output — steady activity that keeps the Colab runtime awake
     during generation. The notebook additionally logs a GPU heartbeat
     every 60s between jobs.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx

from src.config import get_settings
from src.generators.base import VideoGenerator

log = logging.getLogger(__name__)

# Browser user-agent: plain API clients get an interstitial/password page
# from some tunnel providers (localtunnel), while browser UAs pass through.
# Colab-only concern — the RunPod path is untouched.
_BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_HEADERS = {"User-Agent": _BROWSER_UA}


def selected_backend() -> str:
    """Global GPU backend switch from settings.yaml `gpu.backend`."""
    try:
        cfg = get_settings().get("gpu", {}) or {}
        return str(cfg.get("backend", "runpod")).lower()
    except Exception:
        return "runpod"


def resolve_colab_url(explicit: str = "") -> str:
    """Tunnel URL precedence: explicit arg → COLAB_TUNNEL_URL env → settings."""
    import os
    url = (explicit or os.environ.get("COLAB_TUNNEL_URL", "")).strip()
    if not url:
        try:
            cfg = get_settings().get("gpu", {}) or {}
            url = str(cfg.get("colab_url", "")).strip()
        except Exception:
            url = ""
    return url.rstrip("/")


def _gpu_cfg() -> dict:
    try:
        cfg = get_settings().get("gpu", {}) or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _client_cfg() -> dict:
    try:
        cfg = get_settings().get("generator_client", {}) or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


class ColabGenerator(VideoGenerator):
    """Video generator driving a Colab-hosted gpu_server via tunnel URL."""

    name = "colab"
    is_free = True
    cost_per_clip_usd = 0.0
    watermark_free = True
    model = "wan2.2-14b"

    def __init__(self, tunnel_url: str = "") -> None:
        super().__init__(env_value=tunnel_url or "")
        self._model_ready: bool = False

    @property
    def tunnel_url(self) -> str:
        if self.env_value and self.env_value.strip():
            return self.env_value.strip().rstrip("/")
        return resolve_colab_url()

    @property
    def is_configured(self) -> bool:
        return bool(self.tunnel_url)

    # ------------------------------------------------------------- internals
    def _wait_ready(self, base_url: str) -> None:
        """Block until the notebook server reports /ready (long first boot)."""
        if self._model_ready:
            log.debug("Colab server already confirmed ready — skipping /ready poll.")
            return
        g = _gpu_cfg()
        c = _client_cfg()
        max_wait = int(g.get("colab_ready_wait_sec", 3600))
        poll = int(g.get("colab_poll_sec", 3))
        timeout = float(c.get("ready_timeout_sec", 10.0))
        start = time.time()
        log.info("Colab backend: waiting for tunnel %s /ready (up to %d min)...",
                 base_url, max_wait // 60)
        while time.time() - start < max_wait:
            elapsed = int(time.time() - start)
            try:
                with httpx.Client(timeout=timeout, headers=_HEADERS) as client:
                    resp = client.get(f"{base_url}/ready")
                if resp.status_code == 200:
                    self._model_ready = True
                    log.info("Colab server ready (took %ds).", elapsed)
                    return
                if resp.status_code == 500:
                    detail = resp.text
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        pass
                    raise RuntimeError(f"Colab server failed to load model: {detail}")
                # 503 carries live load progress in its detail
                # ("downloading shard-0003 (42%)") — log it, don't alarm.
                progress = ""
                if resp.status_code == 503:
                    try:
                        progress = f" — {resp.json().get('detail', '')}"
                    except Exception:
                        pass
                log.info("Colab server still starting (%d%s, %ds elapsed) — retrying in %ds...",
                         resp.status_code, progress[:160], elapsed, poll)
            except httpx.RequestError as exc:
                log.info("Colab tunnel unreachable (%s, %ds elapsed) — retrying in %ds...",
                         exc, elapsed, poll)
            time.sleep(poll)
        raise RuntimeError(
            f"Colab server at {base_url} not ready after {max_wait}s. "
            "Is the notebook running with the tunnel cell active?")

    def _submit(self, base_url: str, prompt: str, duration_sec: int,
                scene_index: int) -> str:
        """Queue one clip, return the job_id."""
        c = _client_cfg()
        gen = get_settings().video.get("generation", {}) or {}
        fps = max(8, min(int(gen.get("fps", 16)), 30))
        anchor = max(1, int(gen.get("anchor_steps", 20)))
        subsequent = max(1, int(gen.get("subsequent_steps", 12)))
        res = gen.get("resolution", {}) or {}
        width = max(256, int(res.get("width", 1280)))
        height = max(256, int(res.get("height", 720)))
        steps = anchor if scene_index == 0 else subsequent
        log.info("Colab clip %d: %d steps at %dx%d @ %dfps",
                 scene_index, steps, width, height, fps)
        payload = {
            "prompt": prompt,
            "duration": duration_sec,
            "aspect_ratio": "16:9",
            "num_steps": steps,
            "width": width,
            "height": height,
            "fps": fps,
        }
        negative = gen.get("negative_prompt")
        if negative:
            payload["negative_prompt"] = negative
        with httpx.Client(timeout=float(c.get("submit_timeout_sec", 30.0)), headers=_HEADERS) as client:
            resp = client.post(f"{base_url}/generate", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Colab server refused job (status={resp.status_code}): {resp.text[:500]}")
        try:
            job_id = resp.json().get("job_id")
        except Exception as exc:
            raise RuntimeError(
                f"Colab server non-JSON reply: {resp.text[:500]}") from exc
        if not job_id:
            raise RuntimeError(f"Colab server gave no job_id: {resp.text[:500]}")
        return job_id

    def _await_job(self, base_url: str, job_id: str) -> None:
        """Poll /status until done — each poll is a keep-alive heartbeat."""
        c = _client_cfg()
        g = _gpu_cfg()
        max_wait = int(c.get("job_wait_sec", 3600))
        poll = int(g.get("colab_poll_sec", 3))
        status_timeout = float(c.get("status_timeout_sec", 15.0))
        breaker = int(c.get("breaker_502_count", 20))
        bad502 = 0
        start = time.time()
        log.info("Colab job %s queued — heartbeat poll every %ds...", job_id[:8], poll)
        while time.time() - start < max_wait:
            time.sleep(poll)
            elapsed = int(time.time() - start)
            try:
                with httpx.Client(timeout=status_timeout, headers=_HEADERS) as client:
                    resp = client.get(f"{base_url}/status/{job_id}")
            except httpx.RequestError as exc:
                # Tunnel hiccups happen (Cloudflare blips, Colab throttling).
                # Don't fail fast here — the job may still be rendering.
                log.warning("Colab poll failed (%ds elapsed): %s — heartbeat continues...",
                            elapsed, exc)
                continue
            if resp.status_code == 502:
                bad502 += 1
                if bad502 >= breaker:
                    raise RuntimeError(
                        f"Colab tunnel returned {breaker} consecutive 502s — "
                        "the notebook runtime likely died. Check the notebook output.")
                continue
            if resp.status_code != 200:
                log.warning("Colab status %d (%ds elapsed) — retrying...",
                            resp.status_code, elapsed)
                continue
            try:
                data = resp.json()
            except Exception:
                continue
            status = data.get("status", "unknown")
            log.info("Colab job %s: %s (%ds elapsed, heartbeat alive)",
                     job_id[:8], status, elapsed)
            if status == "done":
                return
            if status == "failed":
                raise RuntimeError(
                    f"Colab job failed: {data.get('error', 'unknown error')}")
        raise RuntimeError(f"Colab job {job_id[:8]} timed out after {max_wait}s")

    def _download(self, base_url: str, job_id: str, out_path: Path) -> None:
        c = _client_cfg()
        log.info("Fetching Colab result for job %s...", job_id[:8])
        with httpx.Client(timeout=float(c.get("result_timeout_sec", 120.0)), headers=_HEADERS) as client:
            resp = client.get(f"{base_url}/result/{job_id}")
        if resp.status_code != 200:
            raise RuntimeError(
                f"Colab result fetch failed (status={resp.status_code}): {resp.text[:500]}")
        out_path.write_bytes(resp.content)
        log.info("Saved Colab clip to %s (%d bytes)", out_path, out_path.stat().st_size)

    # ------------------------------------------------------------------ _run
    def _run(self, prompt: str, duration_sec: int, out_path: Path,
             scene_index: int = 0) -> None:
        url = self.tunnel_url
        if not url:
            raise ValueError(
                "Colab backend is selected but no tunnel URL is configured. "
                "Run colab/ActionClipBot_Colab.ipynb and paste its "
                "https://….trycloudflare.com URL into the Default Gpu menu "
                "(or set COLAB_TUNNEL_URL).")
        if not url.startswith("http"):
            raise ValueError(f"Colab tunnel URL looks invalid: {url!r}")
        self._wait_ready(url)
        job_id = self._submit(url, prompt, duration_sec, scene_index)
        self._await_job(url, job_id)
        self._download(url, job_id, out_path)
