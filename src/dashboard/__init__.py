"""
Web dashboard for action-clip-bot.

Adds a FastAPI app for:
  * managing multiple API accounts per provider (account rotation)
  * triggering pipeline runs on demand
  * watching live generation progress with per-attempt reasons
  * inspecting spend / credit usage per account

Entry points:
  * `python -m src.dashboard`       (uvicorn)
  * `action-clip-bot-dashboard`     (installed console script)
"""
