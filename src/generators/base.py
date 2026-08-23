"""
Abstract base for video-clip generators.

Every provider (Hailuo, PixVerse, Kling, fal, paid safety-net) implements
the same interface so the pool can swap them transparently.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationResult:
    """The outcome of a single clip generation attempt."""

    clip_path: str
    provider: str
    cost_usd: float          # 0.0 for free-tier runs
    model: str


class QuotaExceeded(Exception):
    """Raised when a provider has no free credits left for this period."""


class ContentRejected(Exception):
    """Raised when a provider refuses a prompt on content-policy grounds."""


class VideoGenerator(ABC):
    """
    One concrete provider. Subclasses set `name`, `is_free`,
    `cost_per_clip_usd`, `watermark_free`, and implement `_run()`.

    The public `generate()` wraps `_run()` with the bookkeeping every
    provider shares: env-key presence check, async→sync, error typing.
    """

    name: str = "base"
    is_free: bool = True
    cost_per_clip_usd: float = 0.0
    watermark_free: bool = True
    model: str = ""

    def __init__(self, env_value: str) -> None:
        # `env_value` is whatever credential the provider needs (API key,
        # access token, etc.). Pool resolves it from config/providers.yaml.
        self.env_value = env_value

    @property
    def is_configured(self) -> bool:
        return bool(self.env_value)

    @abstractmethod
    def _run(self, prompt: str, duration_sec: int, out_path: Path, scene_index: int = 0) -> None:
        """
        Perform the actual generation. Must raise one of:
          QuotaExceeded, ContentRejected, or any Exception (treated as transient).

        Implementations must write a valid video file to `out_path` on success.
        `scene_index` is 0-based: clip 0 is the anchor (full quality),
        clips 1+ are fast subsequent clips.
        """

    def generate(
        self, prompt: str, duration_sec: int, out_path: Path,
        scene_index: int = 0,
    ) -> GenerationResult:
        """Public entry point — validates config then delegates to `_run`."""
        if not self.is_configured:
            raise RuntimeError(f"{self.name}: missing credential (env key not set)")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._run(prompt=prompt, duration_sec=duration_sec, out_path=out_path,
                  scene_index=scene_index)
        if not out_path.exists():
            raise RuntimeError(f"{self.name}: _run() returned but no file at {out_path}")
        return GenerationResult(
            clip_path=str(out_path),
            provider=self.name,
            cost_usd=0.0 if self.is_free else self.cost_per_clip_usd,
            model=self.model,
        )
