"""
Editor — the ffmpeg composition engine.

Given a list of clip paths + narration + (optional) music + captions subtitle
file, produces TWO final videos:
  * vertical.mp4   1080x1920  (TikTok / Reels / Shorts / Threads)
  * horizontal.mp4 1920x1080  (YouTube / Facebook)

Pipeline per render:
  1. Concatenate clips with a crossfade between each.
  2. Layer narration audio.
  3. Layer music (ducked under narration via sidechain compression).
  4. Burn the .ass captions in.
  5. Append the branded outro frame (still image held for N seconds).
  6. Normalise loudness to -14 LUFS.
  7. Scale + pad to target aspect ratio.

ffmpeg is invoked as a subprocess with -y to overwrite. Errors are surfaced
with the last 50 lines of stderr for diagnosis.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates"


@dataclass(frozen=True)
class SceneSFX:
    path: Path
    start_sec: float
    duration_sec: float


@dataclass(frozen=True)
class CompositionInputs:
    clips: list[Path]                 # ordered, one per scene
    narration_audio: Path | None      # voiceover mp3
    music_audio: Path | None          # background music (optional)
    captions_ass: Path | None         # libass subtitles (optional)
    outro_image: Path | None          # branded outro frame (optional)
    sfx_list: list[SceneSFX] = field(default_factory=list)



@dataclass(frozen=True)
class CompositionResult:
    vertical: Path
    horizontal: Path
    vertical_part1: Path | None = None
    vertical_part2: Path | None = None


def _ffprobe_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip() or 0.0)


def _ffprobe_resolution(path: Path) -> tuple[int, int]:
    """Return (width, height) of a video file."""
    import json
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    stream = data.get("streams", [{}])[0]
    return int(stream.get("width", 1280)), int(stream.get("height", 720))


def _run_ffmpeg(args: list[str]) -> None:
    """Invoke ffmpeg, surfacing the tail of stderr on failure."""
    log.debug("ffmpeg: %s", " ".join(args))
    proc = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                           *args], capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-50:])
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}):\n{tail}")


class Editor:
    def __init__(self, *, fps: int = 30, loudness_lufs: float = -14.0,
                 outro_duration_sec: float = 2.0,
                 vertical: tuple[int, int] = (1080, 1920),
                 horizontal: tuple[int, int] = (1920, 1080),
                 transition_type: str = "fade",
                 transition_duration_sec: float = 0.5,
                 split_threshold_sec: float = 60.0) -> None:
        self.fps = fps
        self.loudness_lufs = loudness_lufs
        self.outro_duration_sec = outro_duration_sec
        self.vertical = vertical
        self.horizontal = horizontal
        self.transition_type = transition_type
        self.transition_duration_sec = transition_duration_sec
        self.split_threshold_sec = split_threshold_sec

    # --------------------------------------------------------------- public
    def compose(self, inputs: CompositionInputs, out_dir: Path | str,
                *, render: tuple[str, ...] = ("vertical", "horizontal")
                ) -> CompositionResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build a single concatenated video stream (clips + outro).
        concat_video = self._concat_with_outro(inputs, out_dir)

        # 2. Build the final audio track (narration + ducked music).
        final_audio = self._mix_audio(inputs, out_dir, target_duration=_ffprobe_duration(concat_video))

        # 3. For each requested aspect, scale+pad, burn captions, mux audio,
        #    and normalise loudness.
        paths: dict[str, Path] = {}
        for variant in render:
            w, h = self.vertical if variant == "vertical" else self.horizontal
            out_path = out_dir / f"{variant}.mp4"
            self._render_variant(concat_video, final_audio, inputs.captions_ass,
                                 out_path, w, h)
            self._normalise_loudness(out_path)
            paths[variant] = out_path
            log.info("rendered %s -> %s", variant, out_path)

            # If vertical was rendered and exceeds configured threshold, split it into two parts for shorts
            if variant == "vertical":
                duration = _ffprobe_duration(out_path)
                if duration >= self.split_threshold_sec:
                    half = duration / 2.0
                    part1_path = out_dir / "vertical_part1.mp4"
                    part2_path = out_dir / "vertical_part2.mp4"
                    log.info("splitting vertical video into two parts for shorts (duration %.2fs)...", duration)
                    _run_ffmpeg([
                        "-ss", "0", "-to", f"{half:.3f}", "-i", str(out_path),
                        "-c", "copy", str(part1_path)
                    ])
                    _run_ffmpeg([
                        "-ss", f"{half:.3f}", "-i", str(out_path),
                        "-c", "copy", str(part2_path)
                    ])
                    paths["vertical_part1"] = part1_path
                    paths["vertical_part2"] = part2_path
                    log.info("split complete: part1=%s, part2=%s", part1_path, part2_path)

        # Always return both keys; if only one was rendered, point the other
        # at the same file so callers don't break.
        return CompositionResult(
            vertical=paths.get("vertical", paths.get("horizontal", out_dir / "vertical.mp4")),
            horizontal=paths.get("horizontal", paths.get("vertical", out_dir / "horizontal.mp4")),
            vertical_part1=paths.get("vertical_part1"),
            vertical_part2=paths.get("vertical_part2"),
        )

    # --------------------------------------------------------------- steps
    def _concat_with_outro(self, inputs: CompositionInputs, out_dir: Path) -> Path:
        """Concatenate clips with dynamic xfade transitions + append outro.

        When ``transition_duration_sec > 0`` and there are 2+ segments, this
        builds an FFmpeg ``filter_complex`` with chained ``xfade`` filters
        that create smooth crossfade transitions between each pair of clips.
        Falls back to the simple concat-demuxer approach when only 1 clip
        exists or transitions are disabled (duration == 0).
        """
        # Normalise each clip to a common fps/codec/resolution so xfade works.
        norm_dir = out_dir / "_normalized"
        norm_dir.mkdir(exist_ok=True)

        # Force high-quality 720p upscale for normalized clips (monetization resolution strategy)
        target_w, target_h = 1280, 720

        norm_clips: list[Path] = []
        for i, clip in enumerate(inputs.clips):
            norm = norm_dir / f"clip_{i:02d}.mp4"
            _run_ffmpeg([
                "-i", str(clip),
                "-vf", f"minterpolate=fps={self.fps}:mi_mode=blend,"
                       f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
                       f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18",
                "-an",                       # drop source audio; we add our own
                str(norm),
            ])
            norm_clips.append(norm)

        # Append the outro as a still image held for N seconds.
        if inputs.outro_image and inputs.outro_image.exists():
            outro_clip = norm_dir / "outro.mp4"
            _run_ffmpeg([
                "-loop", "1", "-i", str(inputs.outro_image),
                "-t", str(self.outro_duration_sec),
                "-r", str(self.fps),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18",
                "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                       f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black",
                str(outro_clip),
            ])
            norm_clips.append(outro_clip)

        concat_out = out_dir / "_concat.mp4"
        td = self.transition_duration_sec
        tt = self.transition_type

        # ---- xfade path: 2+ clips and transition enabled ------------------
        if len(norm_clips) >= 2 and td > 0:
            durations = [_ffprobe_duration(c) for c in norm_clips]
            concat_out = self._xfade_chain(norm_clips, durations, td, tt, concat_out)
        else:
            # ---- simple concat fallback (1 clip or transitions disabled) ---
            list_file = out_dir / "_concat.txt"
            list_file.write_text(
                "\n".join(f"file '{c.absolute()}'" for c in norm_clips) + "\n",
                encoding="utf-8",
            )
            _run_ffmpeg([
                "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c", "copy", str(concat_out),
            ])
        return concat_out

    @staticmethod
    def _xfade_chain(clips: list[Path], durations: list[float],
                     td: float, tt: str, out_path: Path) -> Path:
        """Build and execute an FFmpeg xfade filter chain across *clips*.

        Parameters
        ----------
        clips : list[Path]
            Normalised clip files (same resolution/fps/codec).
        durations : list[float]
            Duration of each clip in seconds (same order as *clips*).
        td : float
            Transition (crossfade) duration in seconds.
        tt : str
            Transition type name accepted by FFmpeg ``xfade``
            (e.g. ``fade``, ``wipeleft``, ``circleopen``).
        out_path : Path
            Where the final stitched video is written.

        Returns
        -------
        Path
            ``out_path`` after a successful encode.
        """
        n = len(clips)
        # Build input arguments: -i clip0.mp4 -i clip1.mp4 …
        input_args: list[str] = []
        for c in clips:
            input_args += ["-i", str(c)]

        # Build the xfade filter chain.
        # For N clips we need N-1 xfade filters chained together.
        #   offset_k = sum(durations[0..k]) - k * td
        filters: list[str] = []
        cumulative = durations[0]
        prev_label = "[0:v]"

        for k in range(1, n):
            offset = cumulative - td
            if offset < 0:
                offset = 0.0
            out_label = f"[xf{k}]" if k < n - 1 else "[vout]"
            filters.append(
                f"{prev_label}[{k}:v]xfade=transition={tt}:duration={td:.3f}"
                f":offset={offset:.3f}{out_label}"
            )
            prev_label = out_label
            cumulative += durations[k] - td

        filter_complex = ";".join(filters)

        _run_ffmpeg([
            *input_args,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "medium", "-crf", "18",
            "-an",
            str(out_path),
        ])
        return out_path

    def _mix_sfx_tracks(self, sfx_list: list[SceneSFX], out_dir: Path, target_duration: float) -> Path | None:
        """
        Mix multiple scene-specific sound effects into a single audio track,
        each delayed by its start time in the video.
        """
        valid_sfx = [s for s in sfx_list if s.path and s.path.exists()]
        if not valid_sfx:
            return None

        out_sfx = out_dir / "_sfx_track.m4a"
        
        args = []
        filter_inputs = []
        for i, sfx in enumerate(valid_sfx):
            args.extend(["-i", str(sfx.path)])
            delay_ms = int(sfx.start_sec * 1000)
            # Apply delay to all channels, trim to duration, set default volume
            filter_inputs.append(
                f"[{i}:a]adelay={delay_ms}:all=true,atrim=end={sfx.duration_sec},volume=0.5[sfx_{i}]"
            )

        if len(valid_sfx) == 1:
            filter_complex = f"[0:a]adelay={int(valid_sfx[0].start_sec * 1000)}:all=true,atrim=end={valid_sfx[0].duration_sec},volume=0.5,apad=whole_dur={target_duration:.3f}[a]"
        else:
            mix_inputs = "".join(f"[sfx_{i}]" for i in range(len(valid_sfx)))
            mix_filter = f"{mix_inputs}amix=inputs={len(valid_sfx)}:duration=longest"
            pad_filter = f"{mix_filter},apad=whole_dur={target_duration:.3f}[a]"
            filter_complex = ";".join(filter_inputs) + ";" + pad_filter
        
        _run_ffmpeg([
            *args,
            "-filter_complex", filter_complex,
            "-map", "[a]",
            "-vn",
            "-t", f"{target_duration:.3f}",
            "-c:a", "aac", "-b:a", "192k",
            str(out_sfx),
        ])
        return out_sfx

    def _mix_audio(self, inputs: CompositionInputs, out_dir: Path,
                   *, target_duration: float) -> Path:
        """
        Produce a single audio track:
          narration at full volume
          + (music + sfx combined) ducked under narration via sidechain compression
        """
        out = out_dir / "_audio.m4a"
        
        # 1. Pre-mix the sound effects if any exist
        sfx_track = self._mix_sfx_tracks(inputs.sfx_list, out_dir, target_duration)

        # 2. Combine BGM (music) and SFX into a background track
        bg_track = None
        if inputs.music_audio and sfx_track:
            music_vol = 0.55 if inputs.narration_audio else 0.75
            _run_ffmpeg([
                "-stream_loop", "-1",
                "-i", str(inputs.music_audio),
                "-i", str(sfx_track),
                "-filter_complex",
                f"[0:a]volume={music_vol}[m];[1:a]volume=0.6[s];[m][s]amix=inputs=2:duration=longest[bg]",
                "-map", "[bg]",
                "-vn",
                "-t", f"{target_duration:.3f}",
                "-c:a", "aac", "-b:a", "192k", str(out_dir / "_bg_combined.m4a"),
            ])
            bg_track = out_dir / "_bg_combined.m4a"
        elif inputs.music_audio:
            music_vol = 0.55 if inputs.narration_audio else 0.85
            _run_ffmpeg([
                "-stream_loop", "-1",
                "-i", str(inputs.music_audio),
                "-af", f"volume={music_vol}",
                "-vn",
                "-t", f"{target_duration:.3f}",
                "-c:a", "aac", "-b:a", "192k", str(out_dir / "_bg_combined.m4a"),
            ])
            bg_track = out_dir / "_bg_combined.m4a"
        elif sfx_track:
            sfx_vol = 0.6 if inputs.narration_audio else 0.85
            _run_ffmpeg([
                "-i", str(sfx_track),
                "-af", f"volume={sfx_vol}",
                "-vn",
                "-t", f"{target_duration:.3f}",
                "-c:a", "aac", "-b:a", "192k", str(out_dir / "_bg_combined.m4a"),
            ])
            bg_track = out_dir / "_bg_combined.m4a"

        # 3. Final mix: narration + combined background (ducked under voice)
        if inputs.narration_audio and bg_track:
            _run_ffmpeg([
                "-i", str(inputs.narration_audio),
                "-i", str(bg_track),
                "-filter_complex",
                "[1:a][0:a]sidechaincompress=threshold=0.05:ratio=4:attack=5:release=150[ducked];"
                f"[0:a][ducked]amix=inputs=2:duration=longest:dropout_transition=2,"
                f"apad=whole_dur={target_duration:.3f}[a]",
                "-map", "[a]",
                "-vn",
                "-c:a", "aac", "-b:a", "192k",
                "-t", f"{target_duration:.3f}", str(out),
            ])
        elif inputs.narration_audio:
            _run_ffmpeg([
                "-i", str(inputs.narration_audio),
                "-af", f"apad=whole_dur={target_duration:.3f}",
                "-vn",
                "-c:a", "aac", "-b:a", "192k",
                "-t", f"{target_duration:.3f}", str(out),
            ])
        elif bg_track:
            _run_ffmpeg([
                "-i", str(bg_track),
                "-af", f"afade=t=out:st={target_duration - 2:.3f}:d=2",
                "-vn",
                "-t", f"{target_duration:.3f}",
                "-c:a", "aac", "-b:a", "192k", str(out),
            ])
        else:
            _run_ffmpeg([
                "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", f"{target_duration:.3f}",
                "-c:a", "aac", "-b:a", "128k", str(out),
            ])
        return out


    def _render_variant(self, concat_video: Path, audio: Path,
                        captions_ass: Path | None, out_path: Path,
                        w: int, h: int) -> None:
        """Scale/pad to (w,h), optionally burn captions, mux audio."""
        vf = [f"scale={w}:{h}:force_original_aspect_ratio=decrease",
              f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black",
              f"fps={self.fps}"]
        if captions_ass and captions_ass.exists():
            # libass subtitle burn-in. Escape colons in the Windows-style path.
            ass_path = str(captions_ass.absolute()).replace("\\", "/").replace(":", "\\:")
            vf.append(f"ass='{ass_path}'")
        # Determine bitrate based on resolution to ensure monetization-grade quality
        if max(w, h) >= 1920:
            bitrate_args = ["-b:v", "8M", "-maxrate", "12M", "-bufsize", "24M"]
        else:
            bitrate_args = ["-b:v", "5M", "-maxrate", "7M", "-bufsize", "14M"]

        _run_ffmpeg([
            "-i", str(concat_video),
            "-i", str(audio),
            "-vf", ",".join(vf),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
            "-crf", "18",
            *bitrate_args,
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",   # web-streaming friendly
            "-shortest", str(out_path),
        ])

    def _normalise_loudness(self, path: Path) -> None:
        """Two-pass loudnorm to the platform-standard -14 LUFS."""
        # Pass 1: measure.
        measure = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(path),
             "-af", f"loudnorm=I={self.loudness_lufs}:TP=-1.5:LRA=11:print_format=json",
             "-f", "null", "-"],
            capture_output=True, text=True,
        )
        import json as _json
        import re
        match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", measure.stderr, re.DOTALL)
        if not match:
            log.warning("loudnorm measurement failed — leaving %s as-is", path.name)
            return
        try:
            stats = _json.loads(match.group(0))
            # Guard against -inf / NaN when the audio is silent or near-silent.
            measured_i = float(stats.get("input_i", "-inf"))
            measured_tp = float(stats.get("input_tp", "-inf"))
            measured_lra = float(stats.get("input_lra", "0"))
            measured_thresh = float(stats.get("input_thresh", "-inf"))
            target_offset = float(stats.get("target_offset", "0"))
            if measured_i == float("-inf") or measured_i < -99:
                log.warning("audio is near-silent (I=%.1f LUFS) — skipping loudnorm", measured_i)
                return
        except (_json.JSONDecodeError, ValueError, KeyError):
            return
        tmp = path.with_suffix(".norm.mp4")
        _run_ffmpeg([
            "-i", str(path),
            "-af", (f"loudnorm=I={self.loudness_lufs}:TP=-1.5:LRA=11:"
                    f"measured_I={measured_i}:measured_TP={measured_tp}:"
                    f"measured_LRA={measured_lra}:"
                    f"measured_thresh={measured_thresh}:"
                    f"offset={target_offset}:linear=true:print_format=summary"),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            str(tmp),
        ])
        tmp.replace(path)


def make_outro_image(text: str = "FOLLOW FOR DAILY ACTION", out_path: Path | None = None) -> Path:
    """
    Generate a simple branded outro PNG using ffmpeg's drawtext filter.

    Black background + large bold white text, sized for the vertical video.
    """
    out_path = out_path or (TEMPLATE_DIR / "outro.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([
        "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1",
        "-vf", (
            f"drawtext=text='{text}':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
            f"fontsize=72:fontcolor=white:"
            f"x=(w-text_w)/2:y=(h-text_h)/2"
        ),
        "-vframes", "1", str(out_path),
    ])
    return out_path
