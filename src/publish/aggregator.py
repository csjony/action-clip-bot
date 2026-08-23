"""
AggregatorPublisher — one API call publishes to all 5 platforms.

This is the PRIMARY publishing path. Aggregators (Upload-Post, Postproxy,
Bundle.social) wrap each platform's native API behind a uniform REST endpoint,
so we avoid juggling 5 OAuth flows + TikTok's app review.

Order is set in settings.yaml `publishing.aggregator_priority`. The first
configured + working aggregator wins. If its free quota is exhausted, callers
fall back to native APIs (see src/publish/native/).

NOTE on Ayrshare: its free tier is images-only, so it's intentionally absent
from the priority list. If you upgrade to a paid Ayrshare plan, add an
AyrshareBackend here.
"""
from __future__ import annotations

import logging

from src.config import Settings, get_settings
from src.publish.base import (
    PostAssets,
    PublishOutcome,
    apply_ai_disclosure,
)

log = logging.getLogger(__name__)


class _AggregatorBackend:
    """One concrete aggregator (Upload-Post / Postproxy / Bundle)."""

    name: str = "base"
    base_url: str = ""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def publish(self, assets: PostAssets, platforms: list[str],
                ai_prefix: str) -> dict[str, str | None]:
        """POST to the aggregator's /post endpoint. Returns {platform: url}."""
        raise NotImplementedError


class UploadPostBackend(_AggregatorBackend):
    name = "upload_post"
    base_url = "https://api.upload-post.com/v1/post"

    def publish(self, assets, platforms, ai_prefix):
        import httpx

        # Upload-Post takes a single media + per-platform caption map.
        # Vertical is the master; YouTube/FB receive horizontal as alt_media.
        payload = {
            "platforms": platforms,
            "media_url": None,   # populated below via multipart
            "title": assets.title,
            "captions": {p: apply_ai_disclosure(assets.captions.get(p, ""), ai_prefix)
                         for p in platforms},
            "hashtags": assets.hashtags,
        }
        with open(assets.vertical, "rb") as fh:
            files = {"media": (assets.vertical.name, fh, "video/mp4")}
            data = {k: (str(v) if v is not None else "") for k, v in payload.items()
                    if k != "captions"}
            data["captions"] = __import__("json").dumps(payload["captions"])
            with httpx.Client(timeout=180.0) as client:
                resp = client.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files, data=data,
                )
                resp.raise_for_status()
                body = resp.json()
        return {p: (body.get("posts", {}) or {}).get(p, {}).get("url") for p in platforms}


class PostproxyBackend(_AggregatorBackend):
    name = "postproxy"
    base_url = "https://api.postproxy.dev/v2/publish"

    def publish(self, assets, platforms, ai_prefix):
        import httpx

        with open(assets.vertical, "rb") as fh:
            files = {"file": (assets.vertical.name, fh, "video/mp4")}
            data = {
                "platforms": ",".join(platforms),
                "title": assets.title,
                "caption": apply_ai_disclosure(
                    assets.captions.get("youtube", ""), ai_prefix),
            }
            with httpx.Client(timeout=180.0) as client:
                resp = client.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files, data=data,
                )
                resp.raise_for_status()
                body = resp.json()
        return {p: (body.get("results", {}) or {}).get(p, {}).get("url") for p in platforms}


class BundleBackend(_AggregatorBackend):
    name = "bundle_social"
    base_url = "https://api.bundle.social/v1/post"

    def publish(self, assets, platforms, ai_prefix):
        import httpx

        payload = {
            "platforms": platforms,
            "title": assets.title,
            "captions": {p: apply_ai_disclosure(assets.captions.get(p, ""), ai_prefix)
                         for p in platforms},
            "hashtags": assets.hashtags,
            "video_path": str(assets.vertical),
        }
        with open(assets.vertical, "rb") as fh:
            files = {"video": (assets.vertical.name, fh, "video/mp4")}
            data = {k: str(v) for k, v in payload.items() if k != "captions"}
            data["captions"] = __import__("json").dumps(payload["captions"])
            with httpx.Client(timeout=180.0) as client:
                resp = client.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files, data=data,
                )
                resp.raise_for_status()
                body = resp.json()
        return {p: (body.get("posts", {}) or {}).get(p, {}).get("url") for p in platforms}


_BACKENDS = {
    "upload_post":   (UploadPostBackend,   "UPLOAD_POST_API_KEY"),
    "postproxy":     (PostproxyBackend,    "POSTPROXY_API_KEY"),
    "bundle_social": (BundleBackend,       "BUNDLE_SOCIAL_API_KEY"),
}


class AggregatorPublisher:
    """Implements Publisher — tries aggregators in priority order."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.platforms = self.settings.publishing.get("platforms",
                          ["youtube", "facebook", "instagram", "threads", "tiktok"])
        self.ai_prefix = self.settings.publishing.get("ai_disclosure_prefix",
                                                       "🤖 AI-generated. ")
        self._backends = self._build()

    def _build(self) -> list[_AggregatorBackend]:
        order = self.settings.publishing.get("aggregator_priority",
                                              ["upload_post", "postproxy", "bundle_social"])
        out: list[_AggregatorBackend] = []
        for name in order:
            cls, env_key = _BACKENDS.get(name, (None, None))
            if cls is None:
                log.warning("unknown aggregator %r in priority list — skipping", name)
                continue
            key = self.settings.env(env_key)
            if key:
                out.append(cls(key))
        return out

    @property
    def configured(self) -> bool:
        return bool(self._backends)

    def publish_all(self, assets: PostAssets) -> PublishOutcome:
        outcome = PublishOutcome(urls={p: None for p in self.platforms})
        if not self._backends:
            outcome.errors["_aggregator"] = "no aggregator configured (set one of UPLOAD_POST/POSTPROXY/BUNDLE keys)"
            return outcome

        for backend in self._backends:
            try:
                urls = backend.publish(assets, self.platforms, self.ai_prefix)
                outcome.urls.update(urls)
                log.info("aggregator %s published: %s", backend.name, urls)
                # If the aggregator succeeded for the platforms we care about,
                # stop. If it failed entirely (returned mostly None), try next.
                if any(urls.values()):
                    return outcome
            except Exception as exc:  # noqa: BLE001 — try next aggregator
                log.warning("aggregator %s failed: %s", backend.name, exc)
                outcome.errors[backend.name] = str(exc)
                continue
        return outcome
