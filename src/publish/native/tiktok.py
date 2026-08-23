"""
Native TikTok publisher — Content Posting API.

Two paths:
  * Sandbox (no app review) — posts marked "from an unaudited app". Good for
    the bootstrap phase; no waiting on TikTok's review.
  * Direct Post — requires app review (days-weeks) but clean attribution.

This module starts in sandbox mode by default; flip DIRECT_POST=true after
approval. TikTok REJECTS watermarked content, which is why our providers.yaml
prioritises watermark-free generators.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from src.config import Settings, get_settings
from src.publish.base import PostAssets, apply_ai_disclosure

log = logging.getLogger(__name__)

BASE = "https://open.tiktokapis.com/v2"


class TikTokPublisher:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client_key = self.settings.env("TIKTOK_CLIENT_KEY")
        self.client_secret = self.settings.env("TIKTOK_CLIENT_SECRET")
        self.access_token = self.settings.env("TIKTOK_ACCESS_TOKEN")
        self.direct = self.settings.env("TIKTOK_DIRECT_POST", "").lower() in {"1", "true", "yes"}

    @property
    def configured(self) -> bool:
        return bool(self.client_key and self.access_token)

    def publish(self, assets: PostAssets, ai_prefix: str) -> str | None:
        if not self.configured:
            log.info("TikTok native: not configured — skipping.")
            return None
        try:
            import httpx
        except ImportError:
            log.warning("httpx not available — skipping TikTok.")
            return None

        media_path = assets.vertical
        # TikTok requires the video be uploaded in chunks via the v2 endpoint.
        headers = {"Authorization": f"Bearer {self.access_token}"}
        try:
            with httpx.Client(timeout=300.0) as client:
                # Step 1: initialise upload.
                file_size = Path(media_path).stat().st_size
                init = client.post(
                    f"{BASE}/post/publish/video/init/",
                    headers={**headers, "Content-Type": "application/json; charset=UTF-8"},
                    json={
                        "post_info": {
                            "title": apply_ai_disclosure(
                                assets.captions.get("tiktok", ""), ai_prefix)[:150],
                            "privacy_level": "PUBLIC_TO_EVERYONE" if self.direct
                                             else "SELF_ONLY",
                            "disable_commercial_audio": False,
                            "ai_info": {"is_ai_generated": True},  # required disclosure
                        },
                        "source_info": {
                            "source": "FILE_UPLOAD",
                            "video_size": file_size,
                            "chunk_size": file_size,
                            "total_chunk_count": 1,
                        },
                    },
                )
                init.raise_for_status()
                data = init.json()["data"]
                upload_url = data["upload_url"]
                publish_id = data["publish_id"]

                # Step 2: PUT the video bytes.
                with open(media_path, "rb") as fh:
                    put = client.put(
                        upload_url,
                        headers={"Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
                                 "Content-Length": str(file_size)},
                        content=fh.read(),
                    )
                    put.raise_for_status()

                # Step 3: poll status (Direct Post is synchronous-ish).
                # Sandbox/SELF_ONLY posts are still queryable.
                url = None
                for _ in range(40):
                    time.sleep(5)
                    status = client.get(
                        f"{BASE}/post/publish/status/fetch/",
                        headers=headers,
                        json={"publish_id": publish_id},
                    )
                    body = status.json().get("data", {})
                    if body.get("status") == "PUBLISH_COMPLETE":
                        url = (body.get("publicaly_available_post_info", {}) or {}) \
                              .get("share_url")
                        break
                if not url:
                    url = f"https://www.tiktok.com/@_/video/{publish_id}"
                log.info("TikTok published: %s", url)
                return url
        except Exception as exc:  # noqa: BLE001
            log.warning("TikTok publish failed: %s", exc)
            return None
