"""
Telegram notifier — success/failure/budget alerts.

Why Telegram: free, simple Bot API, no per-message cost, ideal for a
bootstrap-budget cron job that needs to ping you on every run.

Messages are best-effort: if Telegram is unreachable, the bot logs and
continues — notification failure must never block the pipeline.
"""
from __future__ import annotations

import logging

from src.config import Settings, get_settings

log = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/sendMessage"


def _notify_cfg() -> dict:
    """Transport tunables from settings.yaml `notify:` (dashboard-editable)."""
    cfg = get_settings().get("notify", {}) or {}
    return cfg if isinstance(cfg, dict) else {}


class TelegramNotifier:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.token = self.settings.env("TELEGRAM_BOT_TOKEN")
        self.chat_id = self.settings.env("TELEGRAM_CHAT_ID")

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> bool:
        """Send one message. Returns False if not configured or send failed."""
        if not self.configured:
            log.info("Telegram not configured — skipping alert: %s", text[:80])
            return False
        try:
            import httpx

            cfg = _notify_cfg()
            with httpx.Client(timeout=float(cfg.get("timeout_sec", 15.0))) as client:
                resp = client.post(
                    str(cfg.get("api_url", API)).format(token=self.token),
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": str(cfg.get("parse_mode", "Markdown")),
                        "disable_web_page_preview": bool(cfg.get("preview_disabled", True)),
                    },
                )
                resp.raise_for_status()
                return True
        except Exception as exc:  # noqa: BLE001 — never block the pipeline
            log.warning("Telegram send failed: %s", exc)
            return False

    # ------------------------------------------------------------ formatters
    def notify_success(self, *, title: str, theme: str, urls: dict[str, str | None],
                       monthly_spend_usd: float, budget_cap_usd: float) -> None:
        live = [f"  • {p}: {u}" for p, u in urls.items() if u]
        missed = [p for p, u in urls.items() if not u]
        msg = (
            f"✅ *Action Clip Bot — published*\n\n"
            f"*{title}* (`{theme}`)\n\n"
            + ("\n".join(live) if live else "  (no platforms succeeded)")
            + (f"\n\n⚠️ Missed: {', '.join(missed)}" if missed else "")
            + f"\n\n💰 Spend this month: ${monthly_spend_usd:.2f} / ${budget_cap_usd:.2f}"
        )
        self.send(msg)

    def notify_failure(self, *, stage: str, error: str) -> None:
        self.send(
            f"❌ *Action Clip Bot — failed*\n\n"
            f"Stage: `{stage}`\nError: `{error[:300]}`"
        )

    def notify_budget_warning(self, *, monthly_spend_usd: float, budget_cap_usd: float) -> None:
        self.send(
            f"⚠️ *Budget warning*\n\n"
            f"Monthly paid spend ${monthly_spend_usd:.2f} is near the cap "
            f"${budget_cap_usd:.2f}. Subsequent jobs may run free-tier only."
        )
