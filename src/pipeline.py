"""
Pipeline — orchestrates one full generation+publish run.

This is the single entry point invoked by cron (scripts/run_job.sh) and by
`python -m src.pipeline`. It wires together every phase:

    theme → scriptwriter → generators (fallback chain) → narrator
            → captions → music → editor → publisher → store + notify

Design notes:
  * Every step is wrapped so a failure in one phase triggers a Telegram alert
    and exits non-zero — cron then keeps the schedule for next time.
  * `--quick-run` generates a short 15-second video (3 clips × 5s) without
    publishing — useful for smoke-testing the GPU pipeline end-to-end quickly.
  * The budget guard lives inside GeneratorPool, but this module surfaces a
    pre-flight warning to Telegram when monthly spend is already >80% of cap.
  * Progress events are emitted to the dashboard's EventBus at every phase
    boundary and per-clip; see src/dashboard/events.py for the kind taxonomy.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.compose.editor import CompositionInputs, Editor, SceneSFX, make_outro_image
from src.config import db_path, get_settings
from src.content.captions import Captioner
from src.content.music import MusicPicker
from src.content.narrator import Narrator
from src.content.scriptwriter import Scriptwriter
from src.dashboard.events import (
    KIND_CLIP_FINISHED, KIND_CLIP_STARTED, KIND_PHASE_FINISHED, KIND_PHASE_STARTED,
    KIND_RUN_FAILED, KIND_RUN_FINISHED, KIND_RUN_STARTED, EventBus, set_run_context,
)
from src.generators.pool import BudgetExceeded, GenerationFailed, GeneratorPool
from src.notify import TelegramNotifier
from src.publish.base import PostAssets
from src.publish.coordinator import PublishCoordinator
from src.store import Store

log = logging.getLogger("action-clip-bot")

# Human-readable labels for each pipeline phase, shown on the dashboard.
_PHASES = [
    ("plan",     "scriptwriter"),
    ("generate", "generators"),
    ("audio",    "narration + captions + music"),
    ("compose",  "editor (vertical + horizontal)"),
    ("publish",  "publisher (coordinator)"),
]


class Pipeline:
    # Number of clips / clip duration used in quick-run mode.
    # 3 clips × 5s = 15s total output video — fast enough to verify the GPU
    # pipeline end-to-end without burning 130+ minutes on a full 13-scene run.
    _QUICK_TEST_CLIPS = 3
    _QUICK_TEST_CLIP_SEC = 5

    def __init__(self, *, dry_run: bool = False, theme: str | None = None,
                 run_id: str | None = None,
                 stop_event=None, quick_test: bool = False,
                 script_json: str | None = None) -> None:
        self.dry_run = dry_run
        self.quick_test = quick_test
        self.theme_override = theme
        self.script_json_override = script_json
        self.settings = get_settings()
        self.store = Store(db_path())
        self.notifier = TelegramNotifier(self.settings)
        # run_id may be supplied by the caller (e.g. the dashboard runner, so
        # its state and the events table share the same id). Falls back to a
        # fresh UUID for cron / CLI runs.
        self.run_id = run_id or uuid.uuid4().hex
        # Optional threading.Event that the dashboard sets to request cancellation.
        # Checked between scenes so the stop takes effect cleanly.
        self._stop_event = stop_event
        # EventBus writes progress rows the dashboard polls on. Constructed
        # unconditionally — it's a no-op for `run_id` not currently in context
        # until set_run_context() is called in run().
        self.events = EventBus(self.store)
        # Timestamped run dir under data/ so each video's assets are isolated.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(self.settings.data_dir) / "runs" / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.post_id: int | None = None

    # ------------------------------------------------------------- pod control
    def _stop_runpod_if_configured(self) -> None:
        """Stop the RunPod GPU pod if RUNPOD_API_KEY and RUNPOD_POD_ID are set.

        Safe to call multiple times — RunPod returns 200 even if the pod is
        already stopped, so the finally-block safety net won't double-bill.
        """
        runpod_key = self.settings.env("RUNPOD_API_KEY")
        runpod_id = self.settings.env("RUNPOD_POD_ID")
        if runpod_key and runpod_id:
            try:
                from src.generators.runpod_manager import RunPodManager
                log.info("Stopping RunPod GPU pod %s to save costs...", runpod_id)
                RunPodManager(runpod_key, runpod_id).stop_pod()
            except Exception as exc:
                log.warning("Failed to stop RunPod pod: %s", exc)

    # ------------------------------------------------------------- preflight
    def _preflight_budget_check(self) -> None:
        spend = self.store.monthly_spend_usd()
        cap = self.settings.budget_cap_usd
        if cap > 0 and spend >= cap * 0.8:
            self.notifier.notify_budget_warning(
                monthly_spend_usd=spend, budget_cap_usd=cap)

    def _phase(self, idx: int, label: str):
        """Context manager that emits phase_started/phase_finished events."""
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            self.events.emit(
                KIND_PHASE_STARTED, message=f"[{idx + 1}/{len(_PHASES)}] {label}",
                detail={"phase": label, "index": idx + 1, "total": len(_PHASES)},
            )
            log.info("[%d/%d] %s", idx + 1, len(_PHASES), label)
            try:
                yield
            finally:
                self.events.emit(KIND_PHASE_FINISHED, message=f"done: {label}")
        return _cm()

    # ----------------------------------------------------------------- steps
    def _plan_content(self):
        with self._phase(0, "scriptwriter"):
            if self.script_json_override:
                log.info("Using manual script JSON override")
                from src.content.scriptwriter import Scriptwriter
                plan = Scriptwriter._parse(self.script_json_override, self.theme_override or "manual")
            else:
                raise ValueError(
                    "Manual script JSON is required for all runs! "
                    "Please generate the script via ChatGPT/Gemini/Claude "
                    "using the prompts on the dashboard, and paste it into the JSON field."
                )
            
            if self.quick_test:
                log.info("Quick test mode: trimming scenes to first 3 clips")
                plan.scenes = plan.scenes[:3]
                # Also adjust durations for a quick test (5s per clip)
                for s in plan.scenes:
                    s.duration_sec = 5
                
            log.info("plan: %s (%d scenes, %ds)",
                     plan.title, len(plan.scenes), plan.total_duration_sec)
            return plan

    def _generate_clips(self, plan) -> list[Path]:
        # AccountStore is built lazily so the pipeline imports cleanly even
        # before any accounts have been added (env-var fallback then applies).
        from src.dashboard.accounts import AccountStore

        scenes = plan.scenes
        if self.quick_test:
            scenes = scenes[: self._QUICK_TEST_CLIPS]
            log.info(
                "quick_test mode — capping to %d clips × %ds each (~%ds total)",
                len(scenes), self._QUICK_TEST_CLIP_SEC,
                len(scenes) * self._QUICK_TEST_CLIP_SEC,
            )

        total = len(scenes)
        with self._phase(1, "generators"):
            pool = GeneratorPool(
                self.store, self.settings,
                account_store=AccountStore(self.store),
                event_bus=self.events,
            )
            clips_dir = self.run_dir / "clips"
            clip_paths: list[Path] = []
            # Create the post row now so generation attempts can link to it.
            post_id = self.store.create_post(
                title=plan.title, theme=plan.theme,
                script_json=json.dumps({"hook": plan.hook,
                                         "scenes": [s.prompt for s in plan.scenes]}),
            )
            self.post_id = post_id
            set_run_context(self.run_id, post_id)
            for scene in scenes:
                # Honour a stop request from the dashboard between scenes.
                if self._stop_event is not None and self._stop_event.is_set():
                    log.info("stop requested — halting generation after scene %d",
                             scene.index - 1)
                    self.events.emit(
                        KIND_PHASE_FINISHED,
                        message="run cancelled by user — stopping after current scene",
                    )
                    return clip_paths
                self.events.emit(
                    KIND_CLIP_STARTED, post_id=post_id,
                    scene_index=scene.index, total_scenes=total,
                    message=f"Clip {scene.index}/{total} starting",
                )
                result = pool.generate(
                    prompt=scene.prompt,
                    duration_sec=self._QUICK_TEST_CLIP_SEC if self.quick_test else scene.duration_sec,
                    out_dir=clips_dir,
                    post_id=post_id,
                    scene_index=scene.index,
                    total_scenes=total,
                )
                clip_paths.append(Path(result.clip_path))
                self.events.emit(
                    KIND_CLIP_FINISHED, post_id=post_id,
                    scene_index=scene.index, total_scenes=total,
                    provider=result.provider,
                    message=f"Clip {scene.index}/{total} done via {result.provider}",
                    detail={"cost_usd": result.cost_usd},
                )
            return clip_paths

    def _build_audio_captions(self, plan, clips=None):
        with self._phase(2, "narration + captions + music"):
            assets_dir = self.run_dir / "assets"
            assets_dir.mkdir(exist_ok=True)

            enable_voice = self.settings.video.get("enable_voice", False)
            if self.settings.env("ENABLE_VOICE"):
                enable_voice = self.settings.env("ENABLE_VOICE").lower() in ("true", "1", "yes")

            narration = None
            if enable_voice:
                # Narration: Edge-TTS if available, else a silent-tone fallback
                # (the video still renders and publishes; just no voice).
                narration = assets_dir / "narration.mp3"
                try:
                    Narrator().synthesize(plan.narration, narration)
                except Exception as exc:  # noqa: BLE001
                    log.warning("narration failed (%s) — generating silent audio", exc)
                    self._fallback_audio(narration, plan.total_duration_sec)

            # Captions: Whisper if available, else a text-only .ass from the script.
            captions = None
            if narration:
                try:
                    segments = Captioner(self.settings.whisper_model).transcribe(narration)
                    captions = Captioner().to_ass(segments, assets_dir / "captions.ass")
                except Exception:  # noqa: BLE001
                    log.info("captioning unavailable — building text-only .ass from script")
                    try:
                        captions = self._fallback_captions(plan.narration, assets_dir / "captions.ass")
                    except Exception:  # noqa: BLE001
                        captions = None
            else:
                log.info("narration disabled or unavailable — building text-only .ass from script")
                try:
                    captions = self._fallback_captions(plan.narration, assets_dir / "captions.ass")
                except Exception:  # noqa: BLE001
                    captions = None

            from src.dashboard.accounts import AccountStore
            acct_store = AccountStore(self.store)

            # Resolve Jamendo client_id: dashboard account takes priority,
            # then JAMENDO_CLIENT_ID env var.
            jamendo_accounts = acct_store.resolve("jamendo")
            jamendo_client_id: str | None = None
            jamendo_account_id: int | None = None
            if jamendo_accounts:
                jamendo_client_id = jamendo_accounts[0].api_key
                jamendo_account_id = jamendo_accounts[0].id
            if not jamendo_client_id:
                jamendo_client_id = self.settings.env("JAMENDO_CLIENT_ID") or None

            music = MusicPicker(
                account_id=jamendo_account_id,
                store=self.store,
                jamendo_client_id=jamendo_client_id,
            ).pick(
                plan.total_duration_sec + self.settings.video.get("outro_duration_sec", 2),
                assets_dir / "music.mp3",
                theme=plan.theme,
            )
            sfx_list = []
            sfx_settings = self.settings.video.get("sfx", {})
            sfx_enabled = sfx_settings.get("enable", True)
            sfx_provider = sfx_settings.get("provider", "freesound")

            # Resolve ElevenLabs API key
            elevenlabs_accounts = acct_store.resolve("elevenlabs")
            elevenlabs_api_key = None
            if elevenlabs_accounts:
                elevenlabs_api_key = elevenlabs_accounts[0].api_key
            if not elevenlabs_api_key:
                elevenlabs_api_key = self.settings.env("ELEVENLABS_API_KEY") or None

            # Resolve Freesound API key
            freesound_accounts = acct_store.resolve("freesound")
            freesound_api_key = None
            if freesound_accounts:
                freesound_api_key = freesound_accounts[0].api_key
            if not freesound_api_key:
                freesound_api_key = self.settings.env("FREESOUND_API_KEY") or None

            # Resolve Replicate API key
            replicate_accounts = acct_store.resolve("replicate")
            replicate_api_key = None
            if replicate_accounts:
                replicate_api_key = replicate_accounts[0].api_key
            if not replicate_api_key:
                replicate_api_key = self.settings.env("REPLICATE_API_TOKEN") or None

            if sfx_enabled:
                from src.content.sfx import SFXPicker
                picker = SFXPicker(
                    freesound_api_key=freesound_api_key,
                    elevenlabs_api_key=elevenlabs_api_key,
                    replicate_api_key=replicate_api_key,
                    provider=sfx_provider,
                )
                
                scenes_to_process = plan.scenes
                if clips is not None:
                    scenes_to_process = plan.scenes[:len(clips)]

                current_time = 0.0
                for scene in scenes_to_process:
                    sound_query = getattr(scene, "sound_query", "")
                    if sound_query:
                        out_sfx_path = assets_dir / f"sfx_{scene.index}.mp3"
                        video_path = None
                        if clips and (scene.index - 1) < len(clips):
                            video_path = clips[scene.index - 1]
                        resolved_path = picker.pick(sound_query, scene.duration_sec, out_sfx_path, video_path=video_path)
                        if resolved_path:
                            sfx_list.append(SceneSFX(
                                path=resolved_path,
                                start_sec=current_time,
                                duration_sec=float(scene.duration_sec)
                            ))
                    current_time += float(scene.duration_sec)

            outro = self.run_dir / "outro.png"
            try:
                make_outro_image(out_path=outro)
            except Exception:  # noqa: BLE001
                outro = None
            return narration, captions, music, sfx_list, outro

    @staticmethod
    def _fallback_audio(out_path: Path, duration_sec: int) -> None:
        """Generate a silent mp3 via ffmpeg when Edge-TTS is unavailable."""
        import subprocess
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
             "-t", str(duration_sec), "-c:a", "libmp3lame", str(out_path)],
            check=True,
        )

    @staticmethod
    def _fallback_captions(text: str, out_path: Path) -> Path:
        """Build a single-block .ass from the narration text when Whisper is unavailable."""
        from src.content.captions import Captioner, CaptionSegment
        # Split text into ~5-second chunks at sentence boundaries.
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        chunk_dur = 5.0
        segs = []
        t = 0.0
        for sent in sentences:
            segs.append(CaptionSegment(start=t, end=t + chunk_dur, text=sent))
            t += chunk_dur
        if not segs:
            segs = [CaptionSegment(0.0, chunk_dur, text or "Action.India")]
        return Captioner().to_ass(segs, out_path)

    def _compose(self, plan, clips, narration, captions, music, sfx_list, outro):
        with self._phase(3, "editor (vertical + horizontal)"):
            inputs = CompositionInputs(
                clips=clips,
                narration_audio=narration,
                music_audio=music,
                captions_ass=captions,
                outro_image=outro,
                sfx_list=sfx_list,
            )
            gen_cfg = self.settings.video.get("generation", {})
            render_cfg = self.settings.video.get("render", {})
            v_cfg = render_cfg.get("vertical", {"width": 1080, "height": 1920})
            h_cfg = render_cfg.get("horizontal", {"width": 1920, "height": 1080})
            result = Editor(
                fps=24,  # Force 24fps output for monetization standards (16fps generated clips are interpolated)
                loudness_lufs=self.settings.video.get("loudness_lufs", -14.0),
                outro_duration_sec=self.settings.video.get("outro_duration_sec", 2.0),
                vertical=(int(v_cfg.get("width", 1080)), int(v_cfg.get("height", 1920))),
                horizontal=(int(h_cfg.get("width", 1920)), int(h_cfg.get("height", 1080))),
                transition_type=self.settings.video.get("transition", {}).get("type", "fade"),
                transition_duration_sec=self.settings.video.get("transition", {}).get("duration_sec", 0.5),
                split_threshold_sec=self.settings.video.get("split_threshold_sec", 60.0),
            ).compose(inputs, self.run_dir / "final")
            self.store.update_post_paths(self.post_id, str(result.vertical),
                                         str(result.horizontal))
            return result

    def _publish(self, plan, rendered) -> dict[str, str | None]:
        with self._phase(4, "publisher"):
            if self.dry_run:
                log.info("publisher — DRY RUN, skipping publish")
                urls = {p: None for p in self.settings.publishing.get(
                    "platforms", ["youtube", "facebook", "instagram", "threads", "tiktok"])}
                self.events.emit(
                    KIND_PHASE_FINISHED, message="publisher skipped (dry-run)",
                )
                return urls
            assets = PostAssets(
                vertical=rendered.vertical,
                horizontal=rendered.horizontal,
                title=plan.title,
                captions={p: plan.captions.for_platform(p) for p in
                          ["youtube", "facebook", "instagram", "threads", "tiktok"]},
                hashtags=plan.hashtags,
                vertical_part1=getattr(rendered, "vertical_part1", None),
                vertical_part2=getattr(rendered, "vertical_part2", None),
            )
            outcome = PublishCoordinator(self.settings).publish_all(assets)
            for platform, url in outcome.urls.items():
                if url:
                    self.store.record_publish_url(self.post_id, platform, url)
            if outcome.all_failed:
                self.store.mark_post_failed(self.post_id)
            return outcome.urls

    # -------------------------------------------------------------- run loop
    def run(self) -> int:
        """Returns process exit code (0 = success)."""
        set_run_context(self.run_id)
        self.events.emit(
            KIND_RUN_STARTED, message=f"Pipeline starting (run {self.run_id[:8]})",
            detail={"dry_run": self.dry_run,
                    "theme": self.theme_override or "(random)"},
        )
        try:
            self._preflight_budget_check()
            plan = self._plan_content()
            clips = self._generate_clips(plan)

            # ── Stop GPU pod immediately after clips are generated ─────────────
            # Audio mixing, editing and publishing all run locally — the H200
            # has no further work to do. Stopping here avoids billing the GPU
            # for the ~5–10 min of local post-processing.
            self._stop_runpod_if_configured()

            narration, captions, music, sfx_list, outro = self._build_audio_captions(plan, clips)
            rendered = self._compose(plan, clips, narration, captions, music, sfx_list, outro)
            urls = self._publish(plan, rendered)
            log.info("done — %s", self.run_dir)
            self.events.emit(
                KIND_RUN_FINISHED, post_id=self.post_id,
                message=f"Pipeline finished — {plan.title}",
                detail={"title": plan.title, "theme": plan.theme,
                         "dry_run": self.dry_run},
            )
            self.notifier.notify_success(
                title=plan.title, theme=plan.theme, urls=urls,
                monthly_spend_usd=self.store.monthly_spend_usd(),
                budget_cap_usd=self.settings.budget_cap_usd,
            )
            return 0
        except GenerationFailed as exc:
            log.error("generation failed: %s", exc)
            self.events.emit(
                KIND_RUN_FAILED, post_id=self.post_id,
                message=f"Generation failed: {exc}",
                detail={"error": str(exc)},
            )
            self.notifier.notify_failure(stage="generation", error=str(exc))
            if getattr(self, "post_id", None):
                self.store.mark_post_failed(self.post_id)
            return 1
        except BudgetExceeded as exc:
            log.error("budget exceeded: %s", exc)
            self.events.emit(
                KIND_RUN_FAILED, post_id=self.post_id,
                message=f"Budget exceeded: {exc}",
                detail={"error": str(exc)},
            )
            self.notifier.notify_failure(stage="generation", error=str(exc))
            if getattr(self, "post_id", None):
                self.store.mark_post_failed(self.post_id)
            return 2
        except Exception as exc:  # noqa: BLE001 — surface + alert
            log.error("pipeline failed: %s\n%s", exc, traceback.format_exc())
            self.events.emit(
                KIND_RUN_FAILED, post_id=self.post_id,
                message=f"Pipeline failed: {type(exc).__name__}: {exc}",
                detail={"error": str(exc)[:500],
                         "exc_type": type(exc).__name__,
                         "traceback": traceback.format_exc()[:2000]},
            )
            self.notifier.notify_failure(stage="pipeline",
                                         error=f"{type(exc).__name__}: {exc}")
            if getattr(self, "post_id", None):
                self.store.mark_post_failed(self.post_id)
            return 1
        finally:
            # Safety net: ensure pod is stopped even if generation failed
            # (the happy-path stop happens right after generation above).
            self._stop_runpod_if_configured()

            # Clear context so a subsequent run in the same process (rare —
            # the dashboard's runner) doesn't bleed events into this run_id.
            set_run_context(None)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="action-clip-bot")
    parser.add_argument("--quick-run", action="store_true",
                        help="Fast smoke-test: 3 clips × 5s = 15s video, skips publishing")
    parser.add_argument("--theme", default=None,
                        help="Force a specific theme (else random from settings)")
    parser.add_argument("--script-json", default=None,
                        help="Path to a script JSON file or raw JSON string to run manually")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    
    script_content = None
    if args.script_json:
        if Path(args.script_json).exists():
            script_content = Path(args.script_json).read_text(encoding="utf-8")
        else:
            script_content = args.script_json

    return Pipeline(
        dry_run=args.quick_run,   # quick-run skips publish (dry run)
        theme=args.theme,
        quick_test=args.quick_run,
        script_json=script_content,
    ).run()


if __name__ == "__main__":
    sys.exit(main())
