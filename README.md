# Action Clip Bot 🎬🤖

AI-powered cinematic action video generator and multi-platform auto-publisher — running on a high-fidelity video generation backend via a Cloud GPU VPS (e.g. RunPod, Lambda Labs) running Wan 2.2.

**What it does:** Generates 2–3 short (30–60s) cinematic action videos per week from 100% AI-generated clips, then auto-publishes them to **Facebook, YouTube, TikTok, Instagram, and Threads**.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│               Local / VPS Dashboard                   │
│   · Trigger runs, configure prompts, view logs         │
│   · Managed by SQLite database                         │
└──────────────────────────┬─────────────────────────────┘
                           │ 1. Sends Scene Prompt
                           ▼
┌────────────────────────────────────────────────────────┐
│      GPU VPS / RunPod (RTX 4090 / L4 / A100)           │
│   · Runs FastAPI + Wan 2.2 Model                       │
│   · Exposed publicly via Direct Port / Proxy / Tunnel  │
└──────────────────────────┬─────────────────────────────┘
                           │ 2. Returns Generated MP4
                           ▼
┌────────────────────────────────────────────────────────┐
│                 Video Assembly Engine                  │
│   · Narration (Edge-TTS) + Captions (Whisper)          │
│   · Sound Effects & Foley Sync via MMAudio API         │
│   · Stitched with Cinematic Transitions via FFmpeg     │
│   · Published automatically to 5 platforms             │
└────────────────────────────────────────────────────────┘
```

---

## Getting Started

### 1. Set Up the Bot
```bash
# Clone the repository
git clone <repo-url> action-clip-bot && cd action-clip-bot

# Run setup script (installs python environment, ffmpeg, dependencies)
bash scripts/setup.sh
```

### 2. Set Up Video Generator (GPU VPS)
1. Provision a GPU instance (e.g., RunPod RTX 4090 or L4 GPU).
2. Clone or transfer this repository to your GPU server.
3. Start the GPU server API by running:
   ```bash
   python gpu_server.py
   ```
4. Copy the host URL (e.g. your RunPod proxy URL).

### 3. Configure the Bot
Copy `.env.example` to `.env` and fill in the fields:
```env
GEMINI_API_KEY=your_gemini_api_key
LOCAL_GENERATOR_URL=https://your-gpu-server-url
```

*Or*, go to the **Accounts** page on the dashboard and add a **local** account with the label `gpu_server` and set the **API Key** field to your GPU Server URL.

### 4. Run the Dashboard
```bash
bash scripts/run_dashboard.sh
```
Open `http://localhost:8080` in your web browser. You can trigger runs directly from the dashboard and monitor them live!

---

## Project Structure

```
action-clip-bot/
├── config/
│   ├── settings.yaml          # Schedule, themes, video composition settings
│   └── providers.yaml         # Video-gen provider chain (only 'local' enabled)
├── scripts/
│   ├── setup.sh               # Bootstrap script for VPS/Local PC
│   ├── run_job.sh             # CLI execution runner (for cron jobs)
│   ├── run_dashboard.sh       # Starts the web dashboard
│   ├── test_gpu_server.py     # Smoke test script for GPU VPS url
│   └── mock_gpu_server.py     # Mock GPU FastAPI server for dry runs
├── src/
│   ├── generators/
│   │   ├── base.py            # VideoGenerator interface
│   │   ├── pool.py            # Fallback coordinator
│   │   └── local.py           # Client calling GPU/local server
│   ├── content/
│   │   ├── scriptwriter.py    # LLM scriptwriting via Gemini / Groq
│   │   ├── narrator.py        # Voiceover synthesis via Edge-TTS
│   │   ├── captions.py        # Subtitle styling and timestamping
│   │   ├── sfx.py             # MMAudio / Foley audio selection & generation
│   │   └── music.py           # Jamendo theme-aware music downloader
│   ├── compose/
│   │   └── editor.py          # FFmpeg assembler with xfade transitions
│   └── dashboard/
│       └── app.py             # FastAPI Dashboard application
├── manifest.yaml              # Tech stack manifest
├── gpu_server.py              # Script to run on Cloud GPU VPS
└── README.md
```

## Running Tests
Run the test suite using:
```bash
./.venv/bin/python -m pytest
```

---

## License
MIT
