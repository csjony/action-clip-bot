"""
PublishCoordinator — the single entry point the pipeline uses.

Strategy:
  1. Try the aggregator first (one call → all 5 platforms).
  2. For any platform that came back without a URL, attempt the matching
     native publisher as a fallback.

This way the cheap/free aggregator does most of the work, and we only touch
native APIs (with their OAuth + review overhead) when we have to.
"""
from __future__ import annotations

import logging

from src.config import Settings, get_settings
from src.publish.aggregator import AggregatorPublisher
from src.publish.base import PostAssets, PublishOutcome, Publisher
from src.publish.native.meta import MetaPublisher
from src.publish.native.tiktok import TikTokPublisher
from src.publish.native.youtube import YouTubePublisher

log = logging.getLogger(__name__)


class PublishCoordinator(Publisher):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.platforms = self.settings.publishing.get("platforms",
                          ["youtube", "facebook", "instagram", "threads", "tiktok"])
        self.ai_prefix = self.settings.publishing.get("ai_disclosure_prefix",
                                                       "🤖 AI-generated. ")
        self.aggregator = AggregatorPublisher(self.settings)
        self.youtube = YouTubePublisher(self.settings)
        self.meta = MetaPublisher(self.settings)
        self.tiktok = TikTokPublisher(self.settings)

    def publish_all(self, assets: PostAssets) -> PublishOutcome:
        outcome = PublishOutcome(urls={p: None for p in self.platforms})

        # ---- 1. Aggregator primary -------------------------------------------
        if self.aggregator.configured:
            try:
                agg_outcome = self.aggregator.publish_all(assets)
                outcome.urls.update(agg_outcome.urls)
                outcome.errors.update(agg_outcome.errors)
            except Exception as exc:  # noqa: BLE001
                log.warning("aggregator crashed: %s — using native fallbacks", exc)
                outcome.errors["_aggregator"] = str(exc)
        else:
            outcome.errors["_aggregator"] = "no aggregator configured"

        # ---- 2. Native fallbacks for missing platforms -----------------------
        missing = [p for p in self.platforms if not outcome.urls.get(p)]
        if missing:
            log.info("native fallback for: %s", missing)
            for platform in missing:
                try:
                    url = self._native_publish(platform, assets)
                    if url:
                        outcome.urls[platform] = url
                except Exception as exc:  # noqa: BLE001
                    outcome.errors[platform] = f"native failed: {exc}"
                    log.warning("native %s failed: %s", platform, exc)

        # ---- 3. Publish split shorts parts to short-form platforms -----------
        shorts_platforms = ["tiktok", "instagram", "youtube", "facebook"]
        if assets.vertical_part1 and assets.vertical_part2:
            log.info("publishing split shorts (Part 1 & Part 2) to: %s", shorts_platforms)
            for part_idx, part_path in enumerate([assets.vertical_part1, assets.vertical_part2], 1):
                part_label = f"Part {part_idx}"
                part_assets = PostAssets(
                    vertical=part_path,
                    horizontal=None,
                    title=f"{assets.title} — {part_label}",
                    captions={p: f"{part_label} | {assets.captions.get(p, '')}"
                              for p in assets.captions},
                    hashtags=assets.hashtags,
                )
                for platform in shorts_platforms:
                    if platform not in self.platforms:
                        continue
                    url_key = f"{platform}_short_pt{part_idx}"
                    try:
                        url = self._native_publish(platform, part_assets)
                        if url:
                            outcome.urls[url_key] = url
                            log.info("shorts %s published to %s: %s", part_label, platform, url)
                    except Exception as exc:  # noqa: BLE001
                        outcome.errors[url_key] = f"shorts {part_label} failed: {exc}"
                        log.warning("shorts %s %s failed: %s", part_label, platform, exc)

        return outcome

    def _native_publish(self, platform: str, assets: PostAssets) -> str | None:
        if platform == "youtube" and self.youtube.configured:
            return self.youtube.publish(assets, self.ai_prefix)
        if platform in ("facebook", "instagram", "threads") and self.meta.configured:
            method = {
                "facebook":  self.meta.publish_facebook,
                "instagram": self.meta.publish_instagram,
                "threads":   self.meta.publish_threads,
            }[platform]
            return method(assets, self.ai_prefix)
        if platform == "tiktok" and self.tiktok.configured:
            return self.tiktok.publish(assets, self.ai_prefix)
        log.info("native %s not configured — leaving blank", platform)
        return None
