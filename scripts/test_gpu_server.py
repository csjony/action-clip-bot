#!/usr/bin/env python3
"""
Direct GPU server smoke-test.

Usage:
    python scripts/test_gpu_server.py <GPU_SERVER_URL>

Example:
    python scripts/test_gpu_server.py https://<pod_id>-8000.proxy.runpod.net

What it does:
  1. Polls /ready until the model is loaded (up to 20 min).
  2. POSTs a single short generation request to /generate.
  3. Saves the returned video to /tmp/gpu_test_output.mp4.
  4. Prints duration + file size so you can verify it's a real video.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("httpx not found — run:  pip install httpx")


# ── config ────────────────────────────────────────────────────────────────────
TEST_PROMPT   = "An eagle soaring over snowy mountain peaks, cinematic drone shot"
TEST_DURATION = 3          # seconds — keep short for a quick smoke test
OUT_PATH      = Path("/tmp/gpu_test_output.mp4")

MAX_READY_WAIT  = 1200     # 20 min — same ceiling as local.py
POLL_INTERVAL   = 15       # seconds between /ready polls
GENERATE_TIMEOUT = 600     # 10 min — same as local.py's POST timeout

HEADERS: dict = {}
# ─────────────────────────────────────────────────────────────────────────────


def poll_ready(base_url: str) -> None:
    print(f"\n{'='*60}")
    print(f"  GPU server: {base_url}")
    print(f"  Polling /ready  (max wait: {MAX_READY_WAIT // 60} min)")
    print(f"{'='*60}\n")

    start = time.time()
    while True:
        elapsed = int(time.time() - start)
        if elapsed >= MAX_READY_WAIT:
            raise SystemExit(
                f"\n❌  Timed out after {elapsed}s — model did not become ready.\n"
                f"    Check that the GPU server is still running and the URL is correct."
            )
        try:
            with httpx.Client(headers=HEADERS, timeout=10.0) as client:
                resp = client.get(f"{base_url}/ready")
            if resp.status_code == 200:
                print(f"✅  Model is READY  ({elapsed}s elapsed)\n")
                return
            elif resp.status_code == 503:
                print(f"   [{elapsed:>4}s] Model still loading... (503)")
            else:
                print(f"   [{elapsed:>4}s] Unexpected status {resp.status_code}: {resp.text[:80]}")
        except httpx.RequestError as exc:
            print(f"   [{elapsed:>4}s] /ready unreachable: {exc}")

        time.sleep(POLL_INTERVAL)


def generate_video(base_url: str) -> None:
    print(f"🎬  Submitting async generation job...")
    print(f"    prompt   : {TEST_PROMPT!r}")
    print(f"    duration : {TEST_DURATION}s\n")

    payload = {
        "prompt": TEST_PROMPT,
        "duration": TEST_DURATION,
        "aspect_ratio": "16:9",
    }

    # Step 1: Submit job (returns immediately with job_id)
    try:
        with httpx.Client(headers=HEADERS, timeout=30.0) as client:
            resp = client.post(f"{base_url}/generate", json=payload)
    except httpx.RequestError as exc:
        raise SystemExit(f"\n❌  Connection error during /generate: {exc}")

    if resp.status_code != 200:
        raise SystemExit(f"\n❌  /generate returned {resp.status_code}:\n    {resp.text[:500]}")

    job_id = resp.json().get("job_id")
    if not job_id:
        raise SystemExit(f"\n❌  Server did not return a job_id: {resp.text[:200]}")
    print(f"✅  Job submitted: {job_id[:8]}...\n")

    # Step 2: Poll /status/{job_id}
    t0 = time.time()
    poll_interval = 20
    max_wait = 2400
    while True:
        elapsed = time.time() - t0
        if elapsed > max_wait:
            raise SystemExit(f"\n❌  Timed out after {int(elapsed)}s waiting for job {job_id[:8]}.")
        time.sleep(poll_interval)
        try:
            with httpx.Client(headers=HEADERS, timeout=15.0) as client:
                sr = client.get(f"{base_url}/status/{job_id}")
            data = sr.json()
            status = data.get("status", "unknown")
            print(f"   [{int(elapsed):>4}s] Job {job_id[:8]}: {status}")
            if status == "done":
                break
            elif status == "failed":
                raise SystemExit(f"\n❌  Generation failed: {data.get('error', 'unknown error')}")
        except httpx.RequestError as exc:
            print(f"   [{int(elapsed):>4}s] Status poll error: {exc} — retrying...")

    # Step 3: Fetch the result
    print(f"\n⬇️   Downloading result...")
    try:
        with httpx.Client(headers=HEADERS, timeout=120.0) as client:
            rr = client.get(f"{base_url}/result/{job_id}")
    except httpx.RequestError as exc:
        raise SystemExit(f"\n❌  Connection error fetching result: {exc}")

    if rr.status_code != 200:
        raise SystemExit(f"\n❌  /result returned {rr.status_code}:\n    {rr.text[:500]}")

    OUT_PATH.write_bytes(rr.content)
    size_kb = OUT_PATH.stat().st_size / 1024
    elapsed = time.time() - t0

    print(f"✅  Video saved to: {OUT_PATH}")
    print(f"    Size   : {size_kb:.1f} KB")
    print(f"    Time   : {elapsed:.1f}s")

    if size_kb < 10:
        print("\n⚠️   File is very small — may be an error response rather than a real video.")
        print(f"    Content preview: {rr.content[:200]}")
    else:
        print(f"\n🎉  Success! Open the file to verify:")
        print(f"    xdg-open {OUT_PATH}   # Linux")
        print(f"    open {OUT_PATH}        # macOS")


def main() -> None:
    if len(sys.argv) < 2:
        # Try to pull URL from the bot's DB as a convenience fallback
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from src.config import db_path
            from src.dashboard.accounts import AccountStore
            from src.store import Store
            accounts = AccountStore(Store(db_path())).resolve("local")
            if accounts:
                url = accounts[0].api_key.rstrip("/")
                print(f"ℹ️   No URL argument given — found configured GPU URL in DB: {url}")
            else:
                sys.exit(
                    "Usage: python scripts/test_gpu_server.py <GPU_SERVER_URL>\n"
                    "Example: python scripts/test_gpu_server.py https://<pod_id>-8000.proxy.runpod.net"
                )
        except Exception:
            sys.exit(
                "Usage: python scripts/test_gpu_server.py <GPU_SERVER_URL>\n"
                "Example: python scripts/test_gpu_server.py https://calm-colts-stay.loca.lt"
            )
    else:
        url = sys.argv[1].rstrip("/")

    poll_ready(url)
    generate_video(url)


if __name__ == "__main__":
    main()
