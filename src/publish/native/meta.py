"""
Native Meta publisher — Facebook Page, Instagram Reels, Threads.

All three use the Meta Graph API with a single Page/IG-Business access token.
Threads is technically a separate graph endpoint but uses the same token base.

Flow per platform:
  1. Upload the media to the platform's container endpoint (resumable phases).
  2. Publish the container.
  3. Capture the returned permalink.

Endpoint specifics change frequently — these follow the v19.0 Graph API shape.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from src.config import Settings, get_settings
from src.publish.base import PostAssets, apply_ai_disclosure

log = logging.getLogger(__name__)

GRAPH = "https://graph.facebook.com/v19.0"


class MetaPublisher:
    """Covers Facebook Page + Instagram Business + Threads via one token."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.token = self.settings.env("META_ACCESS_TOKEN")
        self.page_id = self.settings.env("META_PAGE_ID")
        self.ig_id = self.settings.env("META_IG_BUSINESS_ACCOUNT_ID")
        self.threads_id = self.settings.env("THREADS_USER_ID")

    @property
    def configured(self) -> bool:
        return bool(self.token and self.page_id)

    # --------------------------------------------------------------- helpers
    def _post(self, url: str, *, json_body: dict | None = None,
              files: dict | None = None, params: dict | None = None) -> dict:
        import httpx

        merged_params = {"access_token": self.token, **(params or {})}
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(url, params=merged_params, json=json_body, files=files)
            resp.raise_for_status()
            return resp.json()

    def _get(self, url: str, params: dict) -> dict:
        import httpx

        with httpx.Client(timeout=60.0) as client:
            resp = client.get(url, params={"access_token": self.token, **params})
            resp.raise_for_status()
            return resp.json()

    def _wait_for_processing(self, container_id: str, edge: str) -> str:
        """Poll a Meta container until its status is FINISHED, return permalink."""
        for _ in range(60):
            info = self._get(f"{GRAPH}/{container_id}", {"fields": "status,status_code,permalink"})
            status = info.get("status", {}).get("video_status") or info.get("status_code")
            if status in ("FINISHED", "published", "ok"):
                return info.get("permalink") or info.get("id")
            if status in ("ERROR", "FAILED"):
                raise RuntimeError(f"Meta processing failed: {info}")
            time.sleep(5)
        raise TimeoutError(f"Meta container {container_id} did not finish in time")

    # --------------------------------------------------------------- facebook
    def publish_facebook(self, assets: PostAssets, ai_prefix: str) -> str | None:
        if not self.page_id:
            return None
        media_path = assets.horizontal if (assets.horizontal and assets.horizontal.exists()) else assets.vertical
        # Step 1: start an upload session.
        try:
            init = self._post(
                f"{GRAPH}/{self.page_id}/videos",
                json_body={
                    "upload_phase": "start",
                    "file_size": str(Path(media_path).stat().st_size),
                },
            )
            init["upload_session_id"]
            # Step 2: we cheat and use file_url-less direct upload via /videos
            # with source — simplest path for small clips.
            with open(media_path, "rb") as fh:
                publish = self._post(
                    f"{GRAPH}/{self.page_id}/videos",
                    params={
                        "description": apply_ai_disclosure(
                            assets.captions.get("facebook", ""), ai_prefix),
                        "title": assets.title[:80],
                    },
                    files={"source": (Path(media_path).name, fh, "video/mp4")},
                )
            return f"https://facebook.com/{self.page_id}/videos/{publish.get('id')}"
        except Exception as exc:  # noqa: BLE001
            log.warning("Facebook publish failed: %s", exc)
            return None

    # ------------------------------------------------------------- instagram
    def publish_instagram(self, assets: PostAssets, ai_prefix: str) -> str | None:
        if not self.ig_id:
            return None
        # Instagram requires a public URL for media. For a self-hosted bot the
        # simplest path is to serve the file via a temporary presigned URL; for
        # this bootstrap we fall back to "video_url" — fill in a hosting URL
        # in .env (e.g. a public S3/GCS presigned link) before relying on IG.
        ig_video_url = self.settings.env("INSTAGRAM_VIDEO_URL")
        if not ig_video_url:
            log.info("Instagram native: INSTAGRAM_VIDEO_URL not set — skipping. "
                     "Set a public URL for the rendered video to enable IG.")
            return None
        try:
            container = self._post(
                f"{GRAPH}/{self.ig_id}/media",
                json_body={
                    "media_type": "REELS",
                    "video_url": ig_video_url,
                    "caption": apply_ai_disclosure(
                        assets.captions.get("instagram", ""), ai_prefix),
                },
            )
            creation_id = container["id"]
            publish = self._post(
                f"{GRAPH}/{self.ig_id}/media_publish",
                json_body={"creation_id": creation_id},
            )
            return f"https://instagram.com/reel/{publish.get('id')}"
        except Exception as exc:  # noqa: BLE001
            log.warning("Instagram publish failed: %s", exc)
            return None

    # ---------------------------------------------------------------- threads
    def publish_threads(self, assets: PostAssets, ai_prefix: str) -> str | None:
        if not self.threads_id:
            return None
        # Threads requires a public media URL too (like Instagram).
        th_video_url = self.settings.env("THREADS_VIDEO_URL") or \
                       self.settings.env("INSTAGRAM_VIDEO_URL")
        if not th_video_url:
            log.info("Threads native: no public video URL set — skipping.")
            return None
        try:
            container = self._post(
                f"{GRAPH}/{self.threads_id}/threads",
                json_body={
                    "media_type": "VIDEO",
                    "video_url": th_video_url,
                    "text": apply_ai_disclosure(
                        assets.captions.get("threads", "")[:490], ai_prefix),
                },
            )
            creation_id = container["id"]
            publish = self._post(
                f"{GRAPH}/{self.threads_id}/threads_publish",
                json_body={"creation_id": creation_id},
            )
            return f"https://threads.net/@{self.threads_id}/post/{publish.get('id')}"
        except Exception as exc:  # noqa: BLE001
            log.warning("Threads publish failed: %s", exc)
            return None
