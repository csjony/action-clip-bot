"""
Shared HTTP helpers for generator providers.

All provider API calls funnel through here so retry/timeout behaviour is
uniform and easy to mock in tests.
"""
from __future__ import annotations

from typing import Any

import httpx

DEFAULT_TIMEOUT = 30.0
POLL_TIMEOUT = 300.0   # async video jobs can take minutes


def post_json(url: str, payload: dict[str, Any], *, headers: dict[str, str],
              timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


def get_json(url: str, *, headers: dict[str, str], params: dict[str, Any] | None = None,
             timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()


def download(url: str, out_path: str, *, headers: dict[str, str] | None = None,
             timeout: float = DEFAULT_TIMEOUT) -> None:
    with httpx.Client(timeout=timeout) as client:
        with client.stream("GET", url, headers=headers or {}) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)


def poll_until(predicate, fetch, *, interval: float = 5.0, timeout: float = POLL_TIMEOUT):
    """
    Poll `fetch()` every `interval` seconds until `predicate(state)` is True
    or `timeout` is reached. Returns the last state. Raises TimeoutError.
    """
    import time
    deadline = time.monotonic() + timeout
    state = fetch()
    while not predicate(state):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"poll_until timed out after {timeout}s")
        time.sleep(interval)
        state = fetch()
    return state
