"""
Content-plan models.

Production: backed by pydantic (v1 or v2) for strict validation.
Fallback: when pydantic is unavailable (e.g. a bare test interpreter), a
zero-dependency shim provides the same surface area we actually use:
attribute access, field defaults, and a `parse()` classmethod. This keeps the
pipeline runnable during development without forcing a full dependency install.

The whole pipeline speaks `ContentPlan` — every downstream module reads from
this one object so the shape can never drift out of sync.
"""
from __future__ import annotations

try:
    from pydantic import BaseModel, Field  # noqa: F401
    try:
        from pydantic import field_validator  # pydantic v2
        _PYDANTIC_V2 = True
    except ImportError:                      # pragma: no cover — pydantic v1
        from pydantic import validator as field_validator  # type: ignore
        _PYDANTIC_V2 = False
    _HAS_PYDANTIC = True
except ImportError:  # bare-interpreter fallback
    _HAS_PYDANTIC = False
    _PYDANTIC_V2 = False


if _HAS_PYDANTIC:

    def _validator(*args, **kwargs):
        """Wrap pydantic's decorator so the same code works on v1 and v2."""
        if _PYDANTIC_V2:
            return field_validator(*args, **kwargs)
        kwargs.pop("mode", None)
        kwargs.setdefault("allow_reuse", True)
        return field_validator(*args, **kwargs)

    class Scene(BaseModel):
        index: int
        prompt: str = Field(..., min_length=10)
        duration_sec: int = Field(6, ge=3, le=12)
        sound_query: str = ""

        @_validator("duration_sec", mode="before")
        @classmethod
        def _clamp_duration(cls, v: int) -> int:
            """Clamp LLM-generated duration into the valid [3, 12] range instead of crashing."""
            return max(3, min(12, int(v)))

    class Captions(BaseModel):
        youtube: str = ""
        facebook: str = ""
        instagram: str = ""
        threads: str = ""
        tiktok: str = ""

        def for_platform(self, name: str) -> str:
            return getattr(self, name, "") or self.youtube

    class ContentPlan(BaseModel):
        title: str
        theme: str
        hook: str
        scenes: "list[Scene]"
        narration: str
        hashtags: "list[str]" = Field(default_factory=list)
        captions: Captions

        @property
        def total_duration_sec(self) -> int:
            return sum(s.duration_sec for s in self.scenes)

        @_validator("scenes")
        @classmethod
        def _check_indices(cls, v: "list[Scene]") -> "list[Scene]":
            idxs = [s.index for s in v]
            if idxs != list(range(len(v))):
                raise ValueError(f"scene indices must be 0..n-1 contiguous, got {idxs}")
            return v

else:  # pragma: no cover — exercised in bare-env smoke tests only
    from dataclasses import dataclass, field as _dc_field

    def Field(default=None, **_kw):  # type: ignore
        return default if default is not None else _dc_field(default_factory=list
                                                              if _kw.get("default_factory") else None)

    @dataclass
    class _ShimMixin:
        @classmethod
        def parse(cls, data: dict):
            return cls(**data)

    @dataclass
    class Scene(_ShimMixin):
        index: int
        prompt: str
        duration_sec: int = 6
        sound_query: str = ""

    @dataclass
    class Captions(_ShimMixin):
        youtube: str = ""
        facebook: str = ""
        instagram: str = ""
        threads: str = ""
        tiktok: str = ""

        def for_platform(self, name: str) -> str:
            return getattr(self, name, "") or self.youtube

    @dataclass
    class ContentPlan(_ShimMixin):
        title: str
        theme: str
        hook: str
        scenes: list
        narration: str
        captions: Captions
        hashtags: list = _dc_field(default_factory=list)

        @property
        def total_duration_sec(self) -> int:
            return sum(s.duration_sec for s in self.scenes)
