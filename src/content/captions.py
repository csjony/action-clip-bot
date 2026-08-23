"""
Captions — transcribe the narration MP3 to timed subtitle segments.

Primary backend: faster-whisper (local, free, runs on CPU).
Output: a list of (start_sec, end_sec, text) tuples the editor burns in
as styled subtitles via a libass (.ass) file.

Whisper is heavy to import, so it's loaded lazily and cached per-instance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptionSegment:
    start: float    # seconds
    end: float
    text: str


class Captioner:
    def __init__(self, model_size: str = "base") -> None:
        self.model_size = model_size
        self._model = None  # lazy-loaded

    def _load(self):
        if self._model is None:
            # Lazy import: faster-whisper pulls torch etc., which we don't
            # want at module-import time (e.g. for unit tests).
            from faster_whisper import WhisperModel  # type: ignore

            log.info("Loading whisper model '%s' (CPU)...", self.model_size)
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, audio_path: Path | str) -> list[CaptionSegment]:
        """Transcribe audio → ordered, time-stamped segments."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        model = self._load()
        segments_gen, _info = model.transcribe(
            str(audio_path), word_timestamps=False, vad_filter=True,
        )
        out: list[CaptionSegment] = []
        for seg in segments_gen:
            text = seg.text.strip()
            if text:
                out.append(CaptionSegment(start=seg.start, end=seg.end, text=text))
        log.info("Whisper produced %d caption segments", len(out))
        return out

    def to_ass(self, segments: list[CaptionSegment], out_path: Path | str,
               *, style_name: str = "ActionSub") -> Path:
        """
        Write a libass (.ass) subtitle file with TikTok-style styling.

        The actual look (font, colors, animation) lives in
        templates/captions.ass — we keep the style block there and just emit
        the Dialogue events.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        style_header = self._style_header(style_name)

        def _ts(t: float) -> str:
            # libass timestamp: H:MM:SS.cs
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            cs = int((t - int(t)) * 100)
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        lines = ["[Events]",
                 "Format: Layer, Start, End, Style, Name, "
                 "MarginL, MarginR, MarginV, Effect, Text"]
        for seg in segments:
            # Bold, all-caps reads better in vertical short-form.
            text = seg.text.upper()
            lines.append(
                f"Dialogue: 0,{_ts(seg.start)},{_ts(seg.end)},{style_name},"
                f",0,0,0,,{text}"
            )
        out_path.write_text(style_header + "\n".join(lines) + "\n", encoding="utf-8")
        log.info("Wrote %d caption events to %s", len(segments), out_path)
        return out_path

    @staticmethod
    def _style_header(style_name: str) -> str:
        # If templates/captions.ass exists, use its [V4+ Styles] block;
        # otherwise emit a sensible default. Default = bold white with black
        # stroke, centered low — the classic TikTok look.
        template = (Path(__file__).resolve().parent.parent.parent
                    / "templates" / "captions.ass")
        if template.exists():
            head = template.read_text(encoding="utf-8")
            # Keep everything up to and including [Events] handled by us.
            if "[Events]" in head:
                head = head.split("[Events]")[0]
            return head + "\n"
        return (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1080\n"
            "PlayResY: 1920\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, BackColour, "
            "Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, "
            "MarginR, MarginV, Encoding\n"
            f"Style: {style_name},Arial Black,72,"
            "&H00FFFFFF,&H00000000,-1,0,1,4,1,2,80,80,360,1\n\n"
        )
