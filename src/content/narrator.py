"""
Narrator — synthesises the voiceover MP3 via Edge-TTS (free, Microsoft).

Edge-TTS is chosen over paid options (ElevenLabs) for the bootstrap budget.
Voices are tuned for action/cinematic content: deep, authoritative, energetic.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# en-US-ChristopherNeural — deep, authoritative, cinematic American male.
# Best Edge-TTS voice for action/sports/drama narration.
# Defaults below are fallbacks; live values come from settings.yaml `tts:`
# (dashboard-editable) so voices can change without code edits.
DEFAULT_VOICE = "en-US-ChristopherNeural"
ALT_VOICE = "en-US-GuyNeural"          # alternative deep male voice


def _tts_defaults() -> dict:
    from src.config import get_settings
    cfg = get_settings().get("tts", {}) or {}
    return cfg if isinstance(cfg, dict) else {}


class Narrator:
    def __init__(self, voice: str | None = None, rate: str | None = None,
                 pitch: str | None = None) -> None:
        cfg = _tts_defaults()
        self.voice = voice or str(cfg.get("voice", DEFAULT_VOICE))
        self.rate = rate or str(cfg.get("rate", "+4%"))        # slight speed boost for urgency without sounding robotic
        self.pitch = pitch or str(cfg.get("pitch", "+2Hz"))    # tiny pitch bump adds energy

    def synthesize(self, text: str, out_path: Path | str) -> Path:
        """Render `text` to an MP3 file at out_path. Returns the path."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # edge-tts is async; bridge to sync for the pipeline's linear flow.
        import edge_tts  # imported lazily so tests can import this module

        async def _run() -> None:
            communicate = edge_tts.Communicate(
                text=text, voice=self.voice, rate=self.rate, pitch=self.pitch
            )
            await communicate.save(str(out_path))

        try:
            asyncio.run(_run())
        except Exception as exc:  # noqa: BLE001 — surface a clear message
            raise RuntimeError(f"Edge-TTS synthesis failed: {exc}") from exc

        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(f"Edge-TTS produced no audio at {out_path}")
        log.info("Narration synthesised: %s (%d bytes)",
                 out_path.name, out_path.stat().st_size)
        return out_path
