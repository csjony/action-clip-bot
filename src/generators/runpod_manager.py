from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

# Path to the project .env file — used to persist the new pod ID after auto-provisioning
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class PodUnavailableError(RuntimeError):
    """Raised when the RunPod GPU is no longer available and auto-provisioning is not configured."""




class RunPodManager:
    def __init__(self, api_key: str, pod_id: str) -> None:
        self.api_key = api_key
        self.pod_id = pod_id
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.base_url = "https://rest.runpod.io/v1"

    # ──────────────────────────────────────────────────────────────────
    # Status helpers
    # ──────────────────────────────────────────────────────────────────

    def get_pod_info(self) -> dict:
        """Returns the full raw pod JSON from RunPod API."""
        url = f"{self.base_url}/pods/{self.pod_id}"
        try:
            with httpx.Client(headers=self.headers, timeout=15.0) as client:
                resp = client.get(url)
                if resp.status_code != 200:
                    log.warning("RunPod pod info check failed (status=%d): %s", resp.status_code, resp.text)
                    return {}
                return resp.json()
        except Exception as exc:
            log.warning("RunPod API connection failed: %s", exc)
            return {}

    def get_status(self) -> str:
        """Returns current pod status (e.g. 'RUNNING', 'EXITED', 'PAUSED')."""
        pod_data = self.get_pod_info()
        if not pod_data:
            return "UNKNOWN"
        return (
            pod_data.get("runtime", {}).get("gpus", [{}])[0].get("status", "UNKNOWN")
            if "runtime" in pod_data
            else pod_data.get("desiredStatus", pod_data.get("status", "UNKNOWN"))
        )

    # ──────────────────────────────────────────────────────────────────
    # Start / Stop
    # ──────────────────────────────────────────────────────────────────

    def _wait_for_running(self, wait_timeout_sec: int) -> str:
        """Poll until the pod reaches RUNNING state and return its proxy URL."""
        start_time = time.time()
        log.info("Waiting for pod %s to enter RUNNING state...", self.pod_id)
        while time.time() - start_time < wait_timeout_sec:
            status = self.get_status()
            log.info("RunPod pod status: %s", status)
            if status == "RUNNING":
                server_url = f"https://{self.pod_id}-8000.proxy.runpod.net"
                log.info("RunPod pod is RUNNING. Target URL: %s", server_url)
                return server_url
            time.sleep(10)
        raise RuntimeError(
            f"RunPod pod {self.pod_id!r} failed to reach RUNNING status within {wait_timeout_sec}s"
        )

    def start_pod(self, wait_timeout_sec: int = 300) -> str:
        """Start the pod, wait for RUNNING, bootstrap the GPU server, and return its proxy URL.

        Bootstrap always runs regardless of whether the pod was freshly started or already RUNNING,
        so gpu_server.py is guaranteed to be active before generation begins.
        """
        url = f"{self.base_url}/pods/{self.pod_id}/start"
        log.info("Sending start command to RunPod pod %s...", self.pod_id)

        with httpx.Client(headers=self.headers, timeout=30.0) as client:
            resp = client.post(url)
            if resp.status_code not in (200, 201):
                body = resp.text.lower()
                if "already running" in body:
                    log.info("RunPod pod %s is already running — will still bootstrap.", self.pod_id)
                else:
                    raise RuntimeError(
                        f"Failed to start RunPod pod (status={resp.status_code}): {resp.text}"
                    )

        server_url = self._wait_for_running(wait_timeout_sec)
        self._bootstrap_gpu_server()
        return server_url

    def _bootstrap_gpu_server(self) -> None:
        """Poll the SSH port, connect, and run the GPU server in the background."""
        # Wait up to 60 seconds for publicIp and portMappings to be populated by RunPod
        start_wait = time.time()
        public_ip = None
        ssh_port = None
        while time.time() - start_wait < 60:
            pod_info = self.get_pod_info()
            public_ip = pod_info.get("publicIp")
            port_mappings = pod_info.get("portMappings") or {}
            ssh_port = port_mappings.get("22")
            if public_ip and ssh_port:
                break
            log.info("Waiting for RunPod to assign public IP and port mappings...")
            time.sleep(3)

        if not public_ip or not ssh_port:
            log.warning("No public IP or SSH port mapping found. Skipping SSH bootstrap.")
            return

        import socket
        import subprocess

        # 1. Wait for SSH port to accept TCP connections (up to 60s)
        ssh_ready = False
        start_ssh_wait = time.time()
        log.info("Waiting for SSH port %s:%s to open...", public_ip, ssh_port)
        while time.time() - start_ssh_wait < 60:
            try:
                with socket.create_connection((public_ip, int(ssh_port)), timeout=5):
                    ssh_ready = True
                    log.info("SSH port is open and accepting connections.")
                    break
            except (socket.timeout, ConnectionRefusedError, OSError):
                time.sleep(2)

        if not ssh_ready:
            raise RuntimeError(f"SSH port on {public_ip}:{ssh_port} did not open in time.")

        # 2. Upload the latest local gpu_server.py to the pod before starting it.
        #    This ensures code changes (model patches, memory opts) deploy automatically
        #    without needing manual JupyterLab uploads.
        local_gpu_server = Path(__file__).resolve().parents[2] / "gpu_server.py"
        if local_gpu_server.exists():
            log.info("Uploading latest gpu_server.py to remote pod at /workspace/gpu_server.py ...")
            try:
                with open(local_gpu_server, "r", encoding="utf-8") as f:
                    gpu_server_code = f.read()
                upload_cmd = [
                    "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                    "-p", str(ssh_port), f"root@{public_ip}",
                    "cat > /workspace/gpu_server.py",
                ]
                subprocess.run(upload_cmd, input=gpu_server_code, capture_output=True,
                               text=True, check=True, timeout=30)
                log.info("gpu_server.py uploaded successfully.")
            except Exception as exc:
                log.warning("Could not upload gpu_server.py (will use existing version): %s", exc)
        else:
            log.warning("Local gpu_server.py not found at %s — using existing remote copy.", local_gpu_server)

        # 3. Pipe a bootstrap shell script via stdin to `bash -s`.
        #    This is identical to what the user runs manually in the terminal and avoids
        #    all single-line SSH shell-escaping issues that caused exit code 255.
        hf_token = os.environ.get("HF_TOKEN", "")
        wan_model_id = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")

        # Build the bootstrap script as a multiline string
        bootstrap_script = "#!/bin/bash\n"
        if hf_token:
            bootstrap_script += f"export HF_TOKEN={hf_token}\n"
            bootstrap_script += f"export HUGGING_FACE_HUB_TOKEN={hf_token}\n"
        if wan_model_id:
            bootstrap_script += f"export WAN_MODEL_ID={wan_model_id}\n"
        bootstrap_script += "export PYTHONUNBUFFERED=1\n"

        # Compute the HuggingFace cache folder slug for the CURRENT model id.
        # HF converts "/" → "--" and spaces → "-" when naming cache dirs.
        # e.g. "Wan-AI/Wan2.2-T2V-A14B-Diffusers" → "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers"
        hf_cache_slug = "models--" + wan_model_id.replace("/", "--").replace(" ", "-")
        bootstrap_script += (
            "# ── Cache hygiene: remove any model folders that don't match current WAN_MODEL_ID ──\n"
            f"CURRENT_SLUG='{hf_cache_slug}'\n"
            "HF_HUB_DIR=/workspace/huggingface/hub\n"
            "if [ -d \"$HF_HUB_DIR\" ]; then\n"
            "  for d in \"$HF_HUB_DIR\"/models--*; do\n"
            "    [ -d \"$d\" ] || continue\n"
            "    BASENAME=$(basename \"$d\")\n"
            "    if [ \"$BASENAME\" != \"$CURRENT_SLUG\" ]; then\n"
            "      echo \"[cache] Removing stale model cache: $BASENAME (~$(du -sh \"$d\" 2>/dev/null | cut -f1) freed)\"\n"
            "      rm -rf \"$d\"\n"
            "    else\n"
            "      echo \"[cache] Keeping current model cache: $BASENAME\"\n"
            "    fi\n"
            "  done\n"
            "fi\n"
        )
        bootstrap_script += (
            "# Skip restart if server is already healthy, unless gpu_server.py was just updated\n"
            "FORCE_RESTART=0\n"
            "if [ -f \"/workspace/gpu_server.py\" ]; then\n"
            "  MOD_TIME=$(stat -c %Y /workspace/gpu_server.py)\n"
            "  NOW=$(date +%s)\n"
            "  AGE=$((NOW - MOD_TIME))\n"
            "  if [ $AGE -lt 60 ]; then\n"
            "    echo \"gpu_server.py was updated recently ($AGEs ago). Forcing restart.\"\n"
            "    FORCE_RESTART=1\n"
            "  fi\n"
            "fi\n"
            "if [ $FORCE_RESTART -eq 0 ] && curl -sf http://localhost:8000/ready > /dev/null 2>&1; then\n"
            "  echo 'GPU server already ready — skipping restart'\n"
            "  exit 0\n"
            "fi\n"
            "pkill -f 'gpu_server.py' || true\n"
            "sleep 2\n"
            "nohup python3 -u /workspace/gpu_server.py > /workspace/gpu_server.log 2>&1 &\n"
            "echo \"Started gpu_server.py PID: $!\"\n"
        )

        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-p", str(ssh_port),
            f"root@{public_ip}",
            "bash -s",
        ]
        log.info("Piping bootstrap script to remote pod via SSH...")
        try:
            res = subprocess.run(
                ssh_cmd,
                input=bootstrap_script,
                capture_output=True, text=True, check=True, timeout=30,
            )
            log.info("SSH bootstrap succeeded: %s", res.stdout.strip())
        except subprocess.CalledProcessError as exc:
            log.error("SSH bootstrap failed (exit=%d) stdout=%s stderr=%s",
                      exc.returncode, exc.stdout, exc.stderr)
            raise RuntimeError(f"Failed to bootstrap GPU server via SSH: {exc.stderr}") from exc
        except subprocess.TimeoutExpired as exc:
            log.error("SSH bootstrap timed out")
            raise RuntimeError("SSH bootstrap timed out") from exc

    def stop_pod(self) -> None:
        """Halt the pod to stop billing."""
        url = f"{self.base_url}/pods/{self.pod_id}/stop"
        log.info("Sending stop command to RunPod pod %s...", self.pod_id)
        with httpx.Client(headers=self.headers, timeout=30.0) as client:
            resp = client.post(url)
            if resp.status_code not in (200, 201):
                log.warning("Failed to stop RunPod pod (status=%d): %s", resp.status_code, resp.text)
            else:
                log.info("RunPod pod stop request succeeded.")
