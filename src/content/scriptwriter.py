"""
Scriptwriter — turns a rotating theme into a validated ContentPlan.

Primary backend: Google Gemini (generous free tier).
Fallback backend: Groq (Llama 3.1, also free).
Offline fallback: a deterministic template used when no LLM key is set, so the
pipeline can be developed/tested without API access.
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Protocol

from src.config import Settings, get_settings
from src.content.models import Captions, ContentPlan, Scene
from src.store import Store

log = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates" / "prompts"

# Per-theme visual scaffolding for the offline fallback. Each entry lists
# 3-5 scene prompts that chain into a mini action beat.
_FALLBACK_SCENES: dict[str, list[dict[str, str]]] = {
    "neon_city_chase": [
        {"prompt": "Rain-soaked neon Tokyo street at night, motorbike skidding around a corner, shot on 35mm film, low angle, motion blur, cinematic teal and magenta lighting, Panavision anamorphic lens", "sound_query": "heavy rain night street motorbike skid"},
        {"prompt": "POV rider weaving between trucks, sparks flying, live-action film capture, dynamic handheld camera, volumetric light, film grain, high contrast", "sound_query": "motorcycle engine acceleration loud speed"},
        {"prompt": "Pursuer in black SUV ramming through a market stall, cinematic movie scene, slow-motion debris, orange sodium streetlights, shallow depth of field", "sound_query": "car crash smash wooden stall impact"},
        {"prompt": "Bike launches off a loading ramp onto a rooftop, vertigo shot, city skyline backdrop, lens flare, shot on 35mm film", "sound_query": "whoosh ramp jump airborne wind"},
        {"prompt": "Both vehicles screech to a halt at a dead end, standoff, rain falling, dramatic low-key lighting, photograph, realistic lighting", "sound_query": "tires screeching halt rainy ambiance"},
    ],
    "cyberpunk_duel": [
        {"prompt": "Rooftop of a cyberpunk megacity at dusk, two masked figures facing off, holographic billboards illuminating, wide establishing shot, shot on 35mm film, Panavision anamorphic lens", "sound_query": "hologram hum city wind ambient"},
        {"prompt": "Slow draw of reflective katana blades, electric blue and pink reflections, extreme close-up on eyes, cinematic movie scene, shallow depth of field", "sound_query": "sword draw metal ring blade"},
        {"prompt": "Blades clash mid-air, sparks cascade, live-action film capture, slow motion 480fps look, dynamic arc camera, volumetric haze", "sound_query": "metal clash swords sparks"},
        {"prompt": "One fighter backflips over a parapet, vertigo dolly zoom, neon rain, motion streaks, realistic lighting", "sound_query": "whoosh jump rain storm"},
        {"prompt": "Final lunge, both freeze, dramatic silhouette against a giant holographic face, photograph, cinematic film grain", "sound_query": "electricity crackle neon sparks"},
    ],
    "desert_gunfight": [
        {"prompt": "Empty desert main street at high noon, heat haze, vulture circling, wide western shot, dusty sepia grade, shot on 35mm film", "sound_query": "wind blowing desert dust whistle"},
        {"prompt": "Two gunslingers squaring off, hands hovering over holsters, extreme close-up, sweat detail, photograph, harsh sunlight, realistic lighting", "sound_query": "heartbeat tense silence"},
        {"prompt": "Quick-draw muzzle flash, slow motion, smoke rings, cinematic movie scene, dynamic whip-pan, shallow depth of field", "sound_query": "gunshot revolver loud bang"},
        {"prompt": "Rolling tumble behind a water trough, sand spraying, live-action film capture, low tracking shot, lens flare, film grain", "sound_query": "wood splintering bullet impact"},
        {"prompt": "Lone figure walking away, dust settling behind, long shadow, dramatic backlight, photograph, realistic lighting, Panavision anamorphic lens", "sound_query": "footsteps boots gravel walking away"},
    ],
}


class LLMBackend(Protocol):
    """Minimal interface every LLM backend implements."""

    def generate_json(self, system_prompt: str) -> str: ...


# ----------------------------------------------------------------- Gemini
class GeminiBackend:
    """Google Gemini — default LLM with a generous free tier."""

    def __init__(self, api_key: str, model: str = "gemini-3.5-flash", account_id: int | None = None, store: Store | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.account_id = account_id
        self.store = store

    def generate_json(self, system_prompt: str) -> str:
        # Imported lazily so tests/devs without google-genai can still import
        # the scriptwriter module (e.g. to use the offline fallback).
        from google import genai
        from google.genai import types

        prompt_tokens = 0
        completion_tokens = 0
        resp_text = ""
        try:
            client = genai.Client(api_key=self.api_key)
            resp = client.models.generate_content(
                model=self.model,
                contents=system_prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You output ONLY valid JSON. No markdown fences.",
                    temperature=0.9,
                    response_mime_type="application/json",
                ),
            )
            resp_text = resp.text
            if resp.usage_metadata:
                prompt_tokens = resp.usage_metadata.prompt_token_count or 0
                completion_tokens = resp.usage_metadata.candidates_token_count or 0

            # Log to store
            if self.store:
                cost = prompt_tokens * 0.000000075 + completion_tokens * 0.00000030
                self.store.log_api_call(
                    provider="gemini",
                    status="success",
                    account_id=self.account_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost,
                )
            return resp_text
        except Exception as exc:
            if self.store:
                self.store.log_api_call(
                    provider="gemini",
                    status="failed",
                    account_id=self.account_id,
                    error=str(exc),
                )
            raise


# ----------------------------------------------------------------- Groq
class GroqBackend:
    """Groq — blazing-fast Llama 3.1 inference, free tier."""

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant", account_id: int | None = None, store: Store | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.account_id = account_id
        self.store = store

    def generate_json(self, system_prompt: str) -> str:
        import httpx

        prompt_tokens = 0
        completion_tokens = 0
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "temperature": 0.9,
                        "messages": [
                            {"role": "system",
                             "content": "You output ONLY valid JSON. No markdown fences."},
                            {"role": "user", "content": system_prompt},
                        ],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                content = data["choices"][0]["message"]["content"]
                
                if self.store:
                    cost = prompt_tokens * 0.00000005 + completion_tokens * 0.00000008
                    self.store.log_api_call(
                        provider="groq",
                        status="success",
                        account_id=self.account_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost_usd=cost,
                    )
                return content
        except Exception as exc:
            if self.store:
                self.store.log_api_call(
                    provider="groq",
                    status="failed",
                    account_id=self.account_id,
                    error=str(exc),
                )
            raise



# ----------------------------------------------------------------- offline
class OfflineBackend:
    """Deterministic template generator — used when no LLM key is configured."""

    def __init__(self, *, scene_duration: int = 6) -> None:
        self.scene_duration = scene_duration

    def generate_json(self, system_prompt: str) -> str:
        # Re-extract the theme from the rendered prompt so the offline path
        # still respects the caller's choice.
        theme = "neon_city_chase"
        for line in system_prompt.splitlines():
            if line.startswith("THEME:"):
                theme = line.split(":", 1)[1].strip() or theme
                break
        scenes_data = _FALLBACK_SCENES.get(theme, _FALLBACK_SCENES["neon_city_chase"])
        scenes = [
            {
                "index": i,
                "prompt": item["prompt"],
                "duration_sec": self.scene_duration,
                "sound_query": item["sound_query"]
            }
            for i, item in enumerate(scenes_data)
        ]
        narration = (
            f"In a world that never sleeps, every shadow hides a story. "
            f"Tonight, the hunt begins. {theme.replace('_', ' ').title()} — "
            f"only the fast survive the night."
        )
        hashtags = ["#action", "#cinematic", "#shorts", "#reels", "#fyp", "#viral"]
        plan = {
            "title": f"{theme.replace('_', ' ').title()} — Short Action",
            "theme": theme,
            "hook": "He had one rule: never look back.",
            "scenes": scenes,
            "narration": narration,
            "hashtags": hashtags,
            "captions": {
                "youtube": f"{narration}\n\n" + " ".join(hashtags),
                "facebook": f"{narration}\n\n" + " ".join(hashtags),
                "instagram": f"{narration}\n\n" + " ".join(hashtags),
                "threads": f"{narration[:220]}\n\n" + " ".join(hashtags[:3]),
                "tiktok": f"{narration[:180]}\n\n" + " ".join(hashtags[:4]),
            },
        }
        return json.dumps(plan)


# ----------------------------------------------------------- orchestration
class Scriptwriter:
    """Picks a theme, renders the system prompt, parses LLM output into a plan."""

    def __init__(self, settings: Settings | None = None, store: Store | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self._backends: list[LLMBackend] = self._build_backends()

    def _build_backends(self) -> list[LLMBackend]:
        """Build the backends list. LLM API script generation has been removed
        as per user requirements. Only OfflineBackend is retained as a local fallback.
        """
        backends: list[LLMBackend] = []
        backends.append(OfflineBackend(scene_duration=self.settings.video.get("scene_duration_sec", 6)))
        log.debug("Scriptwriter backends (1): OfflineBackend (LLM generation disabled)")
        return backends

    def pick_theme(self) -> str:
        themes = self.settings.llm.get("themes") or list(_FALLBACK_SCENES.keys())
        return random.choice(themes)

    def _render_prompt(self, theme: str, scene_count: int, target_duration: int) -> str:
        tmpl = (_TEMPLATE_DIR / "scriptwriter_system.txt").read_text(encoding="utf-8")
        return tmpl.format(
            theme=theme,
            scene_count=scene_count,
            target_duration=target_duration,
            word_budget=int(target_duration * 2.5),
        )

    def write(self, *, theme: str | None = None,
              scene_count: int | None = None) -> ContentPlan:
        """Generate a content plan. Theme auto-picked if not supplied."""
        theme = theme or self.pick_theme()
        video_cfg = self.settings.video
        target_duration = video_cfg.get("target_duration_sec", 45)
        scene_count = scene_count or video_cfg.get("max_clips", 7)
        prompt = self._render_prompt(theme, scene_count, target_duration)

        # Separate OfflineBackend from LLM backends — OfflineBackend is the
        # unconditional terminal fallback and is never retried in the cycle.
        llm_backends = [b for b in self._backends if not isinstance(b, OfflineBackend)]
        offline_backend = next((b for b in self._backends if isinstance(b, OfflineBackend)), None)

        # We allow up to MAX_RATE_LIMIT_CYCLES full passes through all LLM keys.
        # When every key in a pass is rate-limited, we wait RATE_LIMIT_WAIT_SEC
        # seconds for the per-minute quota to reset and then try the whole list again.
        # This handles the case where 24 free-tier keys all get rate-limited within
        # the same 60-second window and ALL would otherwise be wasted in a single pass.
        MAX_RATE_LIMIT_CYCLES = 3
        RATE_LIMIT_WAIT_SEC = 65  # slightly over 60s to ensure quota window rolls over

        last_err: Exception | None = None
        for cycle in range(MAX_RATE_LIMIT_CYCLES):
            # Track whether every backend this cycle was rate-limited. If so, it is
            # worth waiting for the quota window to roll over and retrying. If even
            # one backend had a non-rate-limit error (e.g. bad JSON, network), we
            # still try ALL remaining backends in this cycle before deciding.
            any_rate_limited = False
            any_non_rate_limited_err = False

            for backend in llm_backends:
                try:
                    raw = backend.generate_json(prompt)
                    plan = self._parse(raw, theme)
                    log.info(
                        "Scriptwriter produced plan via %s (cycle %d): %s (%d scenes)",
                        type(backend).__name__, cycle + 1, plan.title, len(plan.scenes),
                    )
                    return plan
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    is_rate_limit = (
                        "429" in msg or "quota" in msg.lower()
                        or "rate" in msg.lower() or "resourceexhausted" in msg.lower()
                    )
                    if is_rate_limit:
                        any_rate_limited = True
                        log.warning(
                            "backend %s quota-exhausted (cycle %d) — trying next backend",
                            type(backend).__name__, cycle + 1,
                        )
                    else:
                        any_non_rate_limited_err = True
                        log.warning(
                            "backend %s failed (cycle %d): %s — trying next backend",
                            type(backend).__name__, cycle + 1, exc,
                        )
                    last_err = exc
                    # Always continue to the next backend regardless of error type.
                    continue

            # All backends in this cycle were attempted. Decide whether to retry.
            all_rate_limited = any_rate_limited and not any_non_rate_limited_err
            if all_rate_limited and cycle < MAX_RATE_LIMIT_CYCLES - 1:
                log.warning(
                    "All %d LLM backends were rate-limited in cycle %d/%d. "
                    "Waiting %ds for quota to reset before retrying...",
                    len(llm_backends), cycle + 1, MAX_RATE_LIMIT_CYCLES, RATE_LIMIT_WAIT_SEC,
                )
                time.sleep(RATE_LIMIT_WAIT_SEC)
            elif not any_rate_limited:
                # No backend was rate-limited — retrying won't help, stop cycling.
                break

        # All cycles exhausted — use the OfflineBackend as terminal fallback.
        if offline_backend is not None:
            log.warning(
                "All LLM backends exhausted after %d cycle(s) (last error: %s) — "
                "using offline template for this run.",
                MAX_RATE_LIMIT_CYCLES, last_err,
            )
            raw = offline_backend.generate_json(prompt)
            plan = self._parse(raw, theme)
            return plan

        # Should never reach here — OfflineBackend is always built.
        raise RuntimeError(f"all scriptwriter backends failed: {last_err!r}")

    @staticmethod
    def _parse(raw: str, theme: str) -> ContentPlan:
        # Strip accidental markdown fences some models add.
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        data = json.loads(text)
        # Normalize scene indices to 0..n-1 to avoid validation failures
        # if the LLM happens to generate 1-based indices or arbitrary keys.
        scenes_data = data.get("scenes", [])
        scenes = []
        for i, s in enumerate(scenes_data):
            if isinstance(s, dict):
                s["index"] = i
                scenes.append(Scene(**s))
            else:
                scenes.append(s)

        return ContentPlan(
            title=data["title"],
            theme=data.get("theme", theme),
            hook=data["hook"],
            scenes=scenes,
            narration=data["narration"],
            hashtags=data.get("hashtags", []),
            captions=Captions(**data["captions"]),
        )
