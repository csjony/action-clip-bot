from __future__ import annotations

import logging
from pathlib import Path
import httpx

from src.config import get_settings
from src.generators.base import VideoGenerator

log = logging.getLogger(__name__)


class LocalGenerator(VideoGenerator):
    name = "local"
    is_free = True
    cost_per_clip_usd = 0.0
    watermark_free = True
    model = "wan2.2-14b"

    @staticmethod
    def _client_cfg() -> dict:
        """Polling/timeout tunables from settings.yaml `generator_client:`."""
        from src.config import get_settings
        cfg = get_settings().get("generator_client", {}) or {}
        return cfg if isinstance(cfg, dict) else {}

    def __init__(self, server_url: str) -> None:
        # server_url can be passed via the dashboard account key field or config
        super().__init__(env_value=server_url)
        # Cached once the /ready endpoint returns 200. Subsequent clips from the
        # same generator instance (same pipeline run) skip the full polling wait
        # and go straight to the /generate POST, preventing repeated 6-minute
        # blocking waits for every scene.
        self._model_ready: bool = False

    @property
    def is_configured(self) -> bool:
        import os
        has_runpod = bool(os.environ.get("RUNPOD_API_KEY") and os.environ.get("RUNPOD_POD_ID"))
        return bool(self.env_value) or has_runpod

    def _run(self, prompt: str, duration_sec: int, out_path: Path, scene_index: int = 0) -> None:
        import os
        runpod_key = os.environ.get("RUNPOD_API_KEY")
        runpod_id = os.environ.get("RUNPOD_POD_ID")
        
        # If RunPod config is present, always ensure the pod is started/bootstrapped
        # before the first clip attempt. We call start_pod() whenever the model is not
        # yet confirmed ready — even if a cached URL exists — so that _bootstrap_gpu_server
        # fires correctly after pod migrations or restarts.
        if runpod_key and runpod_id:
            if not self._model_ready:
                from src.generators.runpod_manager import RunPodManager
                log.info("RunPod auto-start: ensuring pod %s is running and gpu_server is bootstrapped...", runpod_id)
                try:
                    self.env_value = RunPodManager(runpod_key, runpod_id).start_pod()
                except Exception as exc:
                    raise RuntimeError(f"Failed to auto-start RunPod instance: {exc}") from exc

        if not self.env_value:
            raise ValueError("Local generator server URL is not configured.")
        
        base_url = self.env_value.rstrip("/")
        
        import time

        headers = {}

        if not self._model_ready:
            # 40 minutes: Cloud GPU cold-start requires downloading the large 28 GB 14B weights
            # from HuggingFace. This typically takes 15–25 min on first-time runs.
            cfg = self._client_cfg()
            max_ready_wait = int(cfg.get("ready_wait_sec", 2400))
            poll_interval = int(cfg.get("ready_poll_sec", 15))  # seconds between /ready polls
            start_time = time.time()

            log.info("Polling local generator /ready (will wait up to %d min)...",
                     max_ready_wait // 60)
            while time.time() - start_time < max_ready_wait:
                elapsed = int(time.time() - start_time)
                try:
                    with httpx.Client(headers=headers, timeout=float(self._client_cfg().get("ready_timeout_sec", 10.0))) as client:
                        resp = client.get(f"{base_url}/ready")
                        if resp.status_code == 200:
                            self._model_ready = True
                            log.info("Local generator model is ready (took %ds).", elapsed)
                            break
                        elif resp.status_code == 503:
                            log.info(
                                "Local generator model still loading... %ds elapsed, retrying in %ds.",
                                elapsed, poll_interval,
                            )
                        elif resp.status_code == 500:
                            err_detail = resp.text
                            try:
                                err_detail = resp.json().get("detail", resp.text)
                            except Exception:
                                pass
                            raise RuntimeError(
                                f"Local generator model loading failed on server: {err_detail}"
                            )
                        else:
                            log.warning(
                                "Local generator /ready returned unexpected status %d: %s "
                                "(%ds elapsed). Retrying in %ds...",
                                resp.status_code, resp.text, elapsed, poll_interval,
                            )
                except httpx.RequestError as exc:
                    log.info(
                        "Local generator /ready unreachable: %s (%ds elapsed). "
                        "Retrying in %ds...",
                        exc, elapsed, poll_interval,
                    )

                time.sleep(poll_interval)

            if not self._model_ready:
                elapsed = int(time.time() - start_time)
                raise RuntimeError(
                    f"Local generator model loading timed out after {elapsed}s "
                    f"(limit={max_ready_wait}s) or server is unreachable at {base_url}."
                )
        else:
            log.debug("Local generator already confirmed ready — skipping /ready poll.")
        
        # --- Async job API ---
        # POST /generate returns a job_id immediately (no long-lived connection).
        # We then poll /status/{job_id} until done and fetch /result/{job_id}.
        #
        # Quality strategy is configurable in settings.yaml so H200-backed runs
        # do not stay stuck on the old 4-step / 832x480 speed-optimized defaults.
        generation_cfg = get_settings().video.get("generation", {})
        fps = max(8, min(int(generation_cfg.get("fps", 16)), 30))
        anchor_steps = max(1, int(generation_cfg.get("anchor_steps", 20)))
        subsequent_steps = max(1, int(generation_cfg.get("subsequent_steps", 12)))
        resolution_cfg = generation_cfg.get("resolution", {})
        width = max(256, int(resolution_cfg.get("width", 1280)))
        height = max(256, int(resolution_cfg.get("height", 720)))
        negative_prompt = generation_cfg.get("negative_prompt")

        num_steps = anchor_steps if scene_index == 0 else subsequent_steps
        log.info(
            "Clip %d: using %d denoising steps at %dx%d @ %dfps (%s mode)",
            scene_index, num_steps, width, height, fps,
            "full-quality anchor" if scene_index == 0 else "fast subsequent",
        )

        # Dynamic photographic styling enhancement to permanently prevent CGI/gaming look:
        enhancers = list(self._client_cfg().get("prompt_enhancers", []) or [])
        added_enhancers = []
        prompt_lower = prompt.lower()
        
        # Check to avoid adding redundant/duplicate terms
        for e in enhancers:
            if e == "shot on 35mm" and ("35mm" in prompt_lower or "film" in prompt_lower):
                continue
            if e == "photorealistic" and ("photorealistic" in prompt_lower or "photo-realistic" in prompt_lower):
                continue
            if e not in prompt_lower:
                added_enhancers.append(e)
                
        if added_enhancers:
            enhanced_prompt = prompt.rstrip("., ") + ", " + ", ".join(added_enhancers)
        else:
            enhanced_prompt = prompt

        log.info("Submitting prompt (enhanced: %s)", enhanced_prompt)

        url = f"{base_url}/generate"
        payload = {
            "prompt": enhanced_prompt,
            "duration": duration_sec,
            "aspect_ratio": "16:9",
            "num_steps": num_steps,
            "width": width,
            "height": height,
            "fps": fps,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        log.info("Submitting async generation job to local server: %s", url)

        with httpx.Client(headers=headers, timeout=float(self._client_cfg().get("submit_timeout_sec", 30.0))) as client:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Local generator failed to queue job (status={resp.status_code}): {resp.text[:500]}"
                )
            try:
                job_data = resp.json()
            except Exception as exc:
                raise RuntimeError(
                    f"Local generator returned non-JSON response during /generate (status={resp.status_code}): {resp.text[:500]}"
                ) from exc
            
            job_id = job_data.get("job_id")
            if not job_id:
                raise RuntimeError(f"Local generator did not return a job_id: {resp.text[:500]}")

        log.info("Generation job queued: %s. Polling for completion...", job_id)

        # Poll /status/{job_id} until done or failed.
        # 60 min ceiling for long-running generation jobs.
        # Circuit-breaker: 20 consecutive 502 responses (~60s) means the GPU server
        # has crashed (OOM kill, container restart, etc.) — fail fast with a clear error
        # instead of retrying silently for 60 minutes.
        cfg = self._client_cfg()
        max_wait = int(cfg.get("job_wait_sec", 3600))
        poll_interval = int(cfg.get("job_poll_sec", 3))
        consecutive_502 = 0
        max_consecutive_502 = int(cfg.get("breaker_502_count", 20))  # server-down before giving up
        start = time.time()
        while time.time() - start < max_wait:
            time.sleep(poll_interval)
            elapsed = int(time.time() - start)
            try:
                with httpx.Client(headers=headers, timeout=float(self._client_cfg().get("status_timeout_sec", 15.0))) as client:
                    status_resp = client.get(f"{base_url}/status/{job_id}")
                if status_resp.status_code == 502:
                    consecutive_502 += 1
                    log.warning(
                        "Status poll returned 502 (%ds elapsed) — consecutive: %d/%d",
                        elapsed, consecutive_502, max_consecutive_502,
                    )
                    if consecutive_502 >= max_consecutive_502:
                        raise RuntimeError(
                            f"GPU server appears to have crashed (received {consecutive_502} "
                            f"consecutive 502 responses over ~{consecutive_502 * poll_interval}s). "
                            f"Job {job_id[:8]} cannot be retrieved. "
                            f"Check /workspace/gpu_server.log on the pod for OOM or traceback details."
                        )
                    continue
                if status_resp.status_code != 200:
                    consecutive_502 = 0
                    log.warning("Status poll returned non-200 (%ds elapsed): %d", elapsed, status_resp.status_code)
                    continue
                consecutive_502 = 0
                try:
                    status_data = status_resp.json()
                except Exception:
                    log.warning("Status poll returned non-JSON response (%ds elapsed): %s", elapsed, status_resp.text[:200])
                    continue

                status = status_data.get("status", "unknown")
                log.info("Job %s status: %s (%ds elapsed)", job_id[:8], status, elapsed)
                if status == "done":
                    break
                elif status == "failed":
                    raise RuntimeError(
                        f"Local generator job failed: {status_data.get('error', 'unknown error')}"
                    )
                # "queued" or "running" → keep polling
            except httpx.RequestError as exc:
                consecutive_502 = 0
                log.warning("Status poll failed (%ds elapsed): %s — retrying...", elapsed, exc)
        else:
            raise RuntimeError(f"Local generator timed out after {max_wait}s (job {job_id[:8]})")

        # Fetch the result video.
        log.info("Fetching result for job %s...", job_id[:8])
        with httpx.Client(headers=headers, timeout=float(self._client_cfg().get("result_timeout_sec", 120.0))) as client:
            result_resp = client.get(f"{base_url}/result/{job_id}")
            if result_resp.status_code != 200:
                raise RuntimeError(
                    f"Failed to fetch job result (status={result_resp.status_code}): {result_resp.text[:500]}"
                )
            out_path.write_bytes(result_resp.content)

        log.info("Saved local/GPU generated video clip to %s", out_path)
