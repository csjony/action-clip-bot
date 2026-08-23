"""
Native YouTube publisher (fallback when aggregators are exhausted).

Uses YouTube Data API v3 (resumable upload). Requires an OAuth client secret
JSON at config/youtube_client_secret.json + a cached token.

The bot stays within the default 10,000-quota/day plan: a video upload + a
single thumbnail call ≈ 1,601 units, so 2-3 uploads/week is far under cap.
"""
from __future__ import annotations

import logging

from src.config import Settings, get_settings, PROJECT_ROOT
from src.publish.base import PostAssets, apply_ai_disclosure

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRET = PROJECT_ROOT / "config" / "youtube_client_secret.json"
TOKEN_CACHE = PROJECT_ROOT / "config" / "youtube_token.json"


class YouTubePublisher:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def configured(self) -> bool:
        return CLIENT_SECRET.exists()

    def _creds(self):
        """OAuth2 credentials with caching."""
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request

        creds = None
        if TOKEN_CACHE.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_CACHE), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_CACHE.write_text(creds.to_json())
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET), SCOPES)
            creds = flow.run_local_server(port=0)
            TOKEN_CACHE.write_text(creds.to_json())
        return creds

    def publish(self, assets: PostAssets, ai_prefix: str) -> str | None:
        if not self.configured:
            log.info("YouTube native: no client_secret — skipping.")
            return None
        try:
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
        except ImportError:
            log.warning("google-api-python-client not installed — skipping YouTube.")
            return None

        youtube = build("youtube", "v3", credentials=self._creds())
        body = {
            "snippet": {
                "title": assets.title[:100],
                "description": apply_ai_disclosure(assets.captions.get("youtube", ""),
                                                    ai_prefix),
                "tags": assets.hashtags,
                "categoryId": "1",   # Film & Animation
            },
            "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
        }
        # Pick horizontal for YouTube (full-width), vertical otherwise.
        media_path = assets.horizontal if (assets.horizontal and assets.horizontal.exists()) else assets.vertical
        media = MediaFileUpload(str(media_path), chunksize=-1, resumable=True,
                                mimetype="video/mp4")
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        while response is None:
            status, response = request.next_chunk()
        video_id = response["id"]
        url = f"https://youtu.be/{video_id}"
        log.info("YouTube uploaded: %s", url)
        return url
