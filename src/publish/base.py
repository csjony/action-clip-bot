"""
Publisher base + the per-platform post model.

Every publisher (aggregator + native APIs) implements `publish_all()`,
returning a dict of platform -> post URL (or None on per-platform failure).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)


@dataclass
class PostAssets:
    """Everything needed to publish one video across all platforms."""

    vertical: Path       # 9:16  → tiktok, instagram, threads, youtube (Shorts)
    title: str
    captions: dict       # {platform_name: caption_text}
    horizontal: Path | None = None  # 16:9  → youtube, facebook (optional)
    hashtags: list[str] = field(default_factory=list)
    vertical_part1: Path | None = None  # first half split for shorts
    vertical_part2: Path | None = None  # second half split for shorts


@dataclass
class PublishOutcome:
    """Result of one publish_all() call."""

    urls: dict[str, str | None] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def succeeded(self) -> list[str]:
        return [k for k, v in self.urls.items() if v]

    @property
    def all_failed(self) -> bool:
        return not self.succeeded


class Publisher(Protocol):
    """Common interface for aggregator + native publishers."""

    def publish_all(self, assets: PostAssets) -> PublishOutcome: ...


def apply_ai_disclosure(text: str, prefix: str) -> str:
    """Prepend the AI-disclosure prefix (compliance for monetization)."""
    if not text:
        return prefix.strip()
    if "AI-generated" in text or "🤖" in text:
        return text  # already disclosed
    return f"{prefix}{text}"
