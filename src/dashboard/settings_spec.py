"""
Settings spec — the single registry that drives the dashboard Settings page.

Each group lists fields by their dotted settings.yaml key. The page renders
inputs from this spec, and saving writes DB overrides (settings.yaml itself
is never rewritten, so comments and formatting are preserved).

Field types: int | float | text | bool | select | list
  * select needs "options": [values]
  * list is edited as one-value-per-line textarea (e.g. prompt enhancers)
"""
from __future__ import annotations

SPEC: list[dict] = [
    {"group": "Quick run", "desc": "Smoke-test shape for --quick-run (fast GPU check).", "fields": [
        {"key": "quickrun.clips", "label": "Clips", "type": "int", "default": 3, "min": 1, "max": 13,
         "help": "How many clips a quick test generates."},
        {"key": "quickrun.clip_sec", "label": "Seconds per clip", "type": "int", "default": 5, "min": 1, "max": 12,
         "help": "Duration of each quick-test clip."},
    ]},
    {"group": "Budget", "desc": "Monthly spend guardrails.", "fields": [
        {"key": "budget.monthly_cap_usd", "label": "Monthly cap (USD)", "type": "float", "default": 5.0, "min": 0, "step": 0.5,
         "help": "BUDGET_CAP env var wins over this file value."},
        {"key": "budget_warning_pct", "label": "Warning threshold (%)", "type": "float", "default": 80, "min": 1, "max": 100,
         "help": "Telegram pre-flight warning when spend passes this % of cap."},
        {"key": "credits.local.cap", "label": "Local daily clip cap", "type": "int", "default": 1000000, "min": 1,
         "help": "Free-credit ceiling for the local GPU provider."},
        {"key": "credits.colab.cap", "label": "Colab daily clip cap", "type": "int", "default": 1000000, "min": 1,
         "help": "Free-credit ceiling for the Colab provider."},
    ]},
    {"group": "Video plan", "desc": "Target shape of a full video.", "fields": [
        {"key": "video.target_duration_sec", "label": "Target duration (s)", "type": "int", "default": 70, "min": 5, "max": 600},
        {"key": "video.min_clips", "label": "Min clips", "type": "int", "default": 12, "min": 1, "max": 30},
        {"key": "video.max_clips", "label": "Max clips", "type": "int", "default": 13, "min": 1, "max": 30},
        {"key": "video.clip_duration_sec", "label": "Clip duration (s)", "type": "int", "default": 6, "min": 1, "max": 12},
        {"key": "video.split_threshold_sec", "label": "Shorts split threshold (s)", "type": "float", "default": 60.0, "min": 10,
         "help": "Vertical videos at/over this length split into Part 1 & 2."},
        {"key": "video.loudness_lufs", "label": "Loudness (LUFS)", "type": "float", "default": -14.0, "min": -30, "max": -5, "step": 0.5},
        {"key": "video.outro_duration_sec", "label": "Outro (s)", "type": "float", "default": 2, "min": 0, "max": 10},
        {"key": "video.enable_voice", "label": "Voiceover narration", "type": "bool", "default": False,
         "help": "ENABLE_VOICE env var also forces this on."},
        {"key": "video.sfx.enable", "label": "Scene SFX", "type": "bool", "default": True},
        {"key": "video.sfx.provider", "label": "SFX provider", "type": "select", "default": "freesound",
         "options": ["freesound", "elevenlabs", "mmaudio"]},
        {"key": "video.transition.type", "label": "Transition", "type": "select", "default": "fade",
         "options": ["fade", "wipeleft", "wiperight", "circleopen", "smoothleft", "smoothright"]},
        {"key": "video.transition.duration_sec", "label": "Transition (s)", "type": "float", "default": 0.5, "min": 0, "max": 2, "step": 0.1},
    ]},
    {"group": "Generation quality", "desc": "What the GPU renders per clip.", "fields": [
        {"key": "video.generation.fps", "label": "Generation fps", "type": "int", "default": 16, "min": 8, "max": 30},
        {"key": "video.generation.anchor_steps", "label": "Anchor steps (clip 0)", "type": "int", "default": 12, "min": 1, "max": 40},
        {"key": "video.generation.subsequent_steps", "label": "Later-clip steps", "type": "int", "default": 12, "min": 1, "max": 40},
        {"key": "video.generation.resolution.width", "label": "Width", "type": "int", "default": 832, "min": 256, "max": 1920, "step": 16},
        {"key": "video.generation.resolution.height", "label": "Height", "type": "int", "default": 480, "min": 256, "max": 1080, "step": 16},
        {"key": "video.generation.negative_prompt", "label": "Negative prompt", "type": "text",
         "default": "cgi, 3d render, video game, anime, cartoon, sketch, painting, drawing, unreal engine, blender, smooth surfaces, low quality",
         "help": "What the diffusion model must avoid."},
    ]},
    {"group": "Generator client", "desc": "How the bot talks to the GPU server.", "fields": [
        {"key": "generator_client.ready_wait_sec", "label": "Cold-start wait (s)", "type": "int", "default": 2400, "min": 60,
         "help": "Max wait for /ready (first run downloads ~28 GB)."},
        {"key": "generator_client.ready_poll_sec", "label": "Ready poll (s)", "type": "int", "default": 15, "min": 2},
        {"key": "generator_client.ready_timeout_sec", "label": "Ready HTTP timeout (s)", "type": "float", "default": 10.0, "min": 2},
        {"key": "generator_client.submit_timeout_sec", "label": "Submit timeout (s)", "type": "float", "default": 30.0, "min": 5},
        {"key": "generator_client.job_wait_sec", "label": "Max job wait (s)", "type": "int", "default": 3600, "min": 60},
        {"key": "generator_client.job_poll_sec", "label": "Job poll (s)", "type": "int", "default": 3, "min": 1},
        {"key": "generator_client.status_timeout_sec", "label": "Status timeout (s)", "type": "float", "default": 15.0, "min": 2},
        {"key": "generator_client.result_timeout_sec", "label": "Result timeout (s)", "type": "float", "default": 120.0, "min": 10},
        {"key": "generator_client.breaker_502_count", "label": "502 breaker count", "type": "int", "default": 20, "min": 3,
         "help": "Consecutive 502s before failing fast (crashed server)."},
        {"key": "generator_client.prompt_enhancers", "label": "Prompt enhancers", "type": "list",
         "default": ["photorealistic", "live-action feature film", "shot on 35mm", "organic textures", "raw photograph"],
         "help": "One per line. Auto-appended photorealism anchors; empty disables."},
    ]},
    {"group": "RunPod", "desc": "Pod lifecycle automation (used when the Default Gpu backend is runpod).", "fields": [        {"key": "runpod.base_url", "label": "API base URL", "type": "text", "default": "https://rest.runpod.io/v1"},
        {"key": "runpod.proxy_port", "label": "GPU server port", "type": "int", "default": 8000, "min": 1, "max": 65535},
        {"key": "runpod.default_model", "label": "Default model", "type": "text",
         "default": "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "help": "WAN_MODEL_ID env var wins."},
        {"key": "runpod.api_timeout_sec", "label": "API timeout (s)", "type": "float", "default": 15.0, "min": 2},
        {"key": "runpod.action_timeout_sec", "label": "Start/stop timeout (s)", "type": "float", "default": 30.0, "min": 5},
        {"key": "runpod.running_wait_sec", "label": "RUNNING wait (s)", "type": "int", "default": 300, "min": 30},
        {"key": "runpod.running_poll_sec", "label": "RUNNING poll (s)", "type": "int", "default": 10, "min": 2},
        {"key": "runpod.ip_wait_sec", "label": "IP wait (s)", "type": "int", "default": 60, "min": 5},
        {"key": "runpod.ssh_wait_sec", "label": "SSH wait (s)", "type": "int", "default": 60, "min": 5},
        {"key": "runpod.bootstrap_timeout_sec", "label": "Bootstrap timeout (s)", "type": "int", "default": 30, "min": 5},
        {"key": "runpod.restart_age_sec", "label": "Restart window (s)", "type": "int", "default": 60, "min": 0,
         "help": "Freshly-uploaded server restarts within this window."},
        {"key": "runpod.remote_path", "label": "Remote server path", "type": "text", "default": "/workspace/gpu_server.py"},
    ]},
    {"group": "Render sizes", "desc": "Final output canvases.", "fields": [
        {"key": "video.render.vertical.width", "label": "Vertical w", "type": "int", "default": 1080, "min": 256, "max": 2160},
        {"key": "video.render.vertical.height", "label": "Vertical h", "type": "int", "default": 1920, "min": 256, "max": 3840},
        {"key": "video.render.horizontal.width", "label": "Horizontal w", "type": "int", "default": 1920, "min": 256, "max": 3840},
        {"key": "video.render.horizontal.height", "label": "Horizontal h", "type": "int", "default": 1080, "min": 256, "max": 2160},
        {"key": "pipeline.output_fps", "label": "Output fps", "type": "int", "default": 24, "min": 8, "max": 60},
    ]},
    {"group": "Editor mix", "desc": "FFmpeg assembly, levels and encoding.", "fields": [
        {"key": "editor.norm_width", "label": "Normalise w", "type": "int", "default": 1280, "min": 256, "max": 1920},
        {"key": "editor.norm_height", "label": "Normalise h", "type": "int", "default": 720, "min": 256, "max": 1080},
        {"key": "editor.crf", "label": "CRF quality", "type": "int", "default": 18, "min": 10, "max": 30,
         "help": "Lower = better and bigger."},
        {"key": "editor.preset", "label": "x264 preset", "type": "select", "default": "medium",
         "options": ["ultrafast", "veryfast", "fast", "medium", "slow"]},
        {"key": "editor.audio_bitrate", "label": "Audio bitrate", "type": "select", "default": "192k",
         "options": ["128k", "160k", "192k", "256k", "320k"]},
        {"key": "editor.sfx_volume", "label": "SFX level", "type": "float", "default": 0.5, "min": 0, "max": 2, "step": 0.05},
        {"key": "editor.music_volume", "label": "Music under voice", "type": "float", "default": 0.55, "min": 0, "max": 2, "step": 0.05},
        {"key": "editor.music_solo", "label": "Music-only level", "type": "float", "default": 0.85, "min": 0, "max": 2, "step": 0.05},
        {"key": "editor.duck_ratio", "label": "Ducking ratio", "type": "float", "default": 4, "min": 1, "max": 20, "step": 0.5},
        {"key": "editor.fade_out_sec", "label": "End fade (s)", "type": "float", "default": 2, "min": 0, "max": 10, "step": 0.5},
        {"key": "editor.hq_bitrate", "label": "HQ bitrate (1080p+)", "type": "text", "default": "8M"},
        {"key": "editor.sq_bitrate", "label": "SQ bitrate", "type": "text", "default": "5M"},
        {"key": "editor.loudnorm_tp", "label": "True-peak (dBTP)", "type": "float", "default": -1.5, "min": -9, "max": 0, "step": 0.5},
        {"key": "editor.loudnorm_lra", "label": "Loudness range (LU)", "type": "float", "default": 11, "min": 1, "max": 20},
        {"key": "editor.outro_text", "label": "Outro text", "type": "text", "default": "FOLLOW FOR DAILY ACTION"},
        {"key": "editor.outro_fontsize", "label": "Outro font size", "type": "int", "default": 72, "min": 24, "max": 200},
    ]},
    {"group": "Voice & captions", "desc": "Narration voice and transcription.", "fields": [
        {"key": "tts.voice", "label": "Voice", "type": "select", "default": "en-US-ChristopherNeural",
         "options": ["en-US-ChristopherNeural", "en-US-GuyNeural", "en-US-AriaNeural", "en-GB-RyanNeural", "en-AU-NatashaNeural"]},
        {"key": "tts.rate", "label": "Rate", "type": "text", "default": "+4%"},
        {"key": "tts.pitch", "label": "Pitch", "type": "text", "default": "+2Hz"},
        {"key": "whisper.model", "label": "Whisper model", "type": "select", "default": "base",
         "options": ["tiny", "base", "small", "medium"], "help": "WHISPER_MODEL env var wins."},
        {"key": "whisper.device", "label": "Whisper device", "type": "select", "default": "cpu",
         "options": ["cpu", "cuda"]},
        {"key": "whisper.compute_type", "label": "Compute type", "type": "select", "default": "int8",
         "options": ["int8", "float16", "float32"]},
        {"key": "whisper.vad_filter", "label": "VAD filter", "type": "bool", "default": True},
        {"key": "pipeline.caption_chunk_sec", "label": "Fallback caption chunk (s)", "type": "float", "default": 5.0, "min": 1, "max": 30},
    ]},
    {"group": "Music & SFX network", "desc": "API timeouts, retries and paging.", "fields": [
        {"key": "music.page_size", "label": "Jamendo page size", "type": "int", "default": 15, "min": 1, "max": 100},
        {"key": "music.retries", "label": "Download retries", "type": "int", "default": 3, "min": 0, "max": 10},
        {"key": "music.read_timeout_sec", "label": "Read timeout (s)", "type": "float", "default": 15.0, "min": 2},
        {"key": "music.duration_slack_sec", "label": "Duration slack (s)", "type": "int", "default": 10, "min": 0},
        {"key": "music.min_duration_sec", "label": "Min track (s)", "type": "int", "default": 30, "min": 5},
        {"key": "sfx_api.page_size", "label": "Freesound candidates", "type": "int", "default": 1, "min": 1, "max": 10},
        {"key": "sfx_api.prompt_influence", "label": "Prompt influence", "type": "float", "default": 0.3, "min": 0, "max": 1, "step": 0.05},
        {"key": "sfx_api.replicate_wait_sec", "label": "Replicate wait (s)", "type": "int", "default": 90, "min": 10},
        {"key": "sfx_api.replicate_version", "label": "MMAudio version pin", "type": "text",
         "default": "62871fb59889b2d7c13777f08deb3b36bdff88f7e1d53a50ad7694548a41b484"},
    ]},
    {"group": "LLM tuning", "desc": "Scriptwriting creativity and quota retries.", "fields": [
        {"key": "llm.model.gemini", "label": "Gemini model", "type": "text", "default": "gemini-3.5-flash"},
        {"key": "llm.model.groq", "label": "Groq model", "type": "text", "default": "llama-3.1-8b-instant"},
        {"key": "llm_tuning.temperature", "label": "Temperature", "type": "float", "default": 0.9, "min": 0, "max": 2, "step": 0.05},
        {"key": "llm_tuning.rate_limit_cycles", "label": "Quota retry cycles", "type": "int", "default": 3, "min": 1, "max": 10},
        {"key": "llm_tuning.rate_limit_wait_sec", "label": "Quota wait (s)", "type": "int", "default": 65, "min": 10},
        {"key": "llm_tuning.words_per_sec", "label": "Words per second", "type": "float", "default": 2.5, "min": 1, "max": 5, "step": 0.1},
    ]},
    {"group": "Publishing", "desc": "Caption compliance and disclosure.", "fields": [
        {"key": "publishing.ai_disclosure_prefix", "label": "AI disclosure", "type": "text",
         "default": "🤖 AI-generated cinematic action. "},
        {"key": "publishing.caption_max.youtube", "label": "YouTube cap", "type": "int", "default": 4900, "min": 100},
        {"key": "publishing.caption_max.instagram", "label": "Instagram cap", "type": "int", "default": 2200, "min": 100},
        {"key": "publishing.caption_max.threads", "label": "Threads cap", "type": "int", "default": 490, "min": 50},
        {"key": "publishing.caption_max.tiktok", "label": "TikTok cap", "type": "int", "default": 2200, "min": 100},
        {"key": "publishing.caption_max.facebook", "label": "Facebook cap", "type": "int", "default": 4900, "min": 100},
    ]},
    {"group": "Notifications", "desc": "Telegram transport.", "fields": [
        {"key": "notify.timeout_sec", "label": "HTTP timeout (s)", "type": "float", "default": 15.0, "min": 2},
        {"key": "notify.parse_mode", "label": "Parse mode", "type": "select", "default": "Markdown",
         "options": ["Markdown", "MarkdownV2", "HTML"]},
        {"key": "notify.preview_disabled", "label": "Disable link previews", "type": "bool", "default": True},
    ]},
    {"group": "Dashboard", "desc": "UI paging, limits and live-update cadence.", "fields": [
        {"key": "dashboard.index_runs", "label": "Overview run rows", "type": "int", "default": 5, "min": 1, "max": 50},
        {"key": "dashboard.history_limit", "label": "History page size", "type": "int", "default": 50, "min": 5, "max": 200},
        {"key": "dashboard.history_max", "label": "History max (?limit=)", "type": "int", "default": 200, "min": 10, "max": 500},
        {"key": "dashboard.activity_days", "label": "Activity window (days)", "type": "int", "default": 14, "min": 2, "max": 90},
        {"key": "dashboard.script_json_max", "label": "Script JSON max (chars)", "type": "int", "default": 200000, "min": 1000},
        {"key": "dashboard.log_default", "label": "Log tail (lines)", "type": "int", "default": 500, "min": 50, "max": 2000},
                {"key": "dashboard.log_max", "label": "Log tail max", "type": "int", "default": 2000, "min": 100, "max": 5000},
        {"key": "dashboard.gpu_log_ttl_sec", "label": "GPU log cache (s)", "type": "float", "default": 10, "min": 2, "max": 120,
         "help": "Higher = fewer hung-pod SSH attempts."},        {"key": "dashboard.event_poll_sec", "label": "Run events poll (s)", "type": "int", "default": 2, "min": 1, "max": 30},
        {"key": "dashboard.logs_poll_ms", "label": "Logs refresh (ms)", "type": "int", "default": 1500, "min": 500, "max": 10000, "step": 100},
        {"key": "dashboard.status_poll_sec", "label": "Status pill (s)", "type": "int", "default": 10, "min": 5, "max": 120},
        {"key": "dashboard.toast_ms", "label": "Toast lifetime (ms)", "type": "int", "default": 3500, "min": 1000, "max": 15000, "step": 250},
    ]},
]


def all_keys() -> list[str]:
    return [f["key"] for g in SPEC for f in g["fields"]]


def field_for(key: str) -> dict | None:
    for g in SPEC:
        for f in g["fields"]:
            if f["key"] == key:
                return f
    return None
