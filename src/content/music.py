"""
Music — selects background tracks from the local music library templates/music/.

Strategy (3 tiers, in order):

  Tier 1 — Jamendo API search (Trending-First)
    Uses the video's theme to pick a mood/tag query, searches
    https://api.jamendo.com/v3.0/tracks/ and downloads the best match.
    Searches are ordered by ``popularity_month`` to target currently
    trending tracks — mimicking the native platform trending audio
    strategy for maximum algorithmic reach.
    Each theme has multiple fallback queries tried in order until a
    suitable track is found.
    Result is cached in data/music_cache/ — same track is reused on
    subsequent runs so we don't re-download every time.
    Requires a free Jamendo client_id (register at developer.jamendo.com).

  Tier 2 — Local library  templates/music/
    Any .mp3/.wav/.m4a/.ogg/.flac files you drop here are picked at random.
    Previously downloaded Jamendo tracks land in data/music_cache/ (not here)
    so this folder stays clean for hand-curated favourites.

  Tier 3 — Synthetic fallback
    If both upper tiers fail, generate an action drum loop via ffmpeg and
    save it as templates/music/action_pulse.mp3 for future re-use.
"""
from __future__ import annotations

import json
import logging
import random
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

MUSIC_DIR   = Path(__file__).resolve().parent.parent.parent / "templates" / "music"
CACHE_DIR   = Path(__file__).resolve().parent.parent.parent / "data" / "music_cache"
JAMENDO_API = "https://api.jamendo.com/v3.0/tracks/"


def _music_cfg() -> dict:
    """Network/retry tunables from settings.yaml `music:` (dashboard-editable)."""
    from src.config import get_settings
    cfg = get_settings().get("music", {}) or {}
    return cfg if isinstance(cfg, dict) else {}

# Jamendo order strategy:
# popularity_month → currently trending this month (best for algorithm boost)
# popularity_total → all-time most played (reliable quality fallback)
_ORDER_TRENDING = "popularity_month"
_ORDER_POPULAR  = "popularity_total"

# ---------------------------------------------------------------------------
# Theme → Jamendo search query list
# Each value is a list of query strings tried in order. The first query
# that returns results with suitable duration is used. This multi-query
# fallback ensures we always find music even for niche themes.
# ---------------------------------------------------------------------------
_THEME_QUERIES: dict[str, list[str]] = {
    # ── All themes from settings.yaml ─────────────────────────────────────
    "neon_city_chase":         ["electronic cinematic speed chase", "electronic chase dark", "action electronic"],
    "cyberpunk_duel":          ["electronic dark action cinematic", "cyberpunk electronic", "dark electronic intense"],
    "desert_gunfight":         ["western epic tense dramatic", "spaghetti western action", "epic orchestral dramatic"],
    "underwater_infiltration": ["dark ambient underwater cinematic", "suspense tense ambient", "cinematic thriller"],
    "rooftop_parkour":         ["electronic urban chase parkour", "upbeat action electronic", "electronic sport"],
    "heist_getaway":           ["heist suspense tense cinematic", "thriller suspense", "action cinematic"],
    "sword_clash":             ["epic orchestral battle action", "orchestral dramatic", "epic battle"],
    "snowmobile_pursuit":      ["electronic action chase speed", "sport electronic", "cinematic chase"],
    "warehouse_ambush":        ["dark industrial action tense", "dark cinematic suspense", "action tense"],
    "train_top_fight":         ["action cinematic chase intense", "orchestral action epic", "epic cinematic"],
    # ── Generic fallback categories ────────────────────────────────────────
    "action":                  ["action cinematic epic", "action electronic", "epic"],
    "chase":                   ["chase electronic speed", "electronic chase", "action speed"],
    "fight":                   ["fight action intense dramatic", "action intense", "dramatic"],
    "heist":                   ["heist suspense tense", "thriller suspense", "cinematic tense"],
    "sci_fi":                  ["electronic futuristic cinematic", "electronic ambient", "futuristic"],
    "war":                     ["war epic orchestral dramatic", "orchestral epic battle", "dramatic orchestral"],
    "sports":                  ["sport upbeat motivational", "upbeat electronic sport", "motivational"],
    "horror":                  ["dark horror suspense", "dark ambient horror", "suspense horror"],
    "adventure":               ["adventure epic orchestral", "orchestral adventure", "epic adventure"],
    # ── Catch-all ──────────────────────────────────────────────────────────
    "_default":                ["action cinematic epic", "action electronic cinematic", "epic cinematic"],
}


def _resolve_queries(theme: str) -> list[str]:
    """Map a theme string to an ordered list of Jamendo search queries.

    Dashboard-saved `music.theme_queries` entries extend/override the
    builtin map so moods can change without code edits.
    """
    extra = _music_cfg().get("theme_queries") or {}
    table = dict(_THEME_QUERIES)
    if isinstance(extra, dict):
        for k, v in extra.items():
            if isinstance(v, list) and v:
                table[str(k)] = [str(q) for q in v]
    if theme in table:
        return table[theme]
    # Try partial match — e.g. "urban_heist_2025" → "heist"
    for key in table:
        if key != "_default" and key in theme.lower():
            return table[key]
    return table["_default"]


class MusicPicker:
    def __init__(
        self,
        api_key: str | None = None,       # kept for backward compat (was legacy key)
        account_id: int | None = None,
        store=None,
        *,
        jamendo_client_id: str | None = None,
    ) -> None:
        self.account_id = account_id
        self.store = store
        # jamendo_client_id can be passed explicitly or resolved from the
        # dashboard accounts / .env by the pipeline before calling pick().
        self._jamendo_id = jamendo_client_id

    @property
    def configured(self) -> bool:
        return True  # Local music is always available

    # ---------------------------------------------------------------------- #
    # Public entry point                                                       #
    # ---------------------------------------------------------------------- #
    def pick(
        self,
        duration_sec: int | float,
        out_path: Path | str,
        *,
        theme: str | None = None,
    ) -> Path | None:
        """
        Select a background music track and copy it to *out_path*.

        Args:
            duration_sec: Minimum acceptable track length in seconds.
            out_path:     Destination path for the chosen track.
            theme:        The video's theme string (used for Jamendo search).

        Returns:
            The resolved *out_path* on success, or ``None`` if all tiers fail.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        MUSIC_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # ── Tier 1: Jamendo API ─────────────────────────────────────────────
        if self._jamendo_id:
            try:
                track = self._jamendo_search(theme or "_default", int(duration_sec))
                if track:
                    shutil.copy(track, out_path)
                    log.info("BGM (Jamendo): %s -> %s", track.name, out_path.name)
                    return out_path
            except Exception as exc:
                log.warning("Jamendo search failed (%s) — falling back to local library", exc)
        else:
            log.debug("No Jamendo client_id — skipping API search")

        # ── Tier 2: Local library ────────────────────────────────────────────
        audio_files = [
            f for f in MUSIC_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
        ]
        if audio_files:
            chosen = random.choice(audio_files)
            try:
                shutil.copy(chosen, out_path)
                log.info("BGM (local library): %s -> %s", chosen.name, out_path.name)
                return out_path
            except Exception as exc:
                log.warning("Failed to copy local track: %s", exc)

        # ── Tier 3: Synthetic fallback ───────────────────────────────────────
        fallback = MUSIC_DIR / "action_pulse.mp3"
        if not fallback.exists():
            log.info("Generating synthetic action pulse BGM...")
            try:
                synth_floor = int(_music_cfg().get("min_synth_sec", 120))
                self._generate_action_pulse(fallback, max(int(duration_sec) + 10, synth_floor))
            except Exception as exc:
                log.warning("Fallback BGM generation failed: %s", exc)
                return None
        try:
            shutil.copy(fallback, out_path)
            log.info("BGM (synthetic fallback): %s -> %s", fallback.name, out_path.name)
            return out_path
        except Exception as exc:
            log.warning("Failed to copy fallback track: %s", exc)
            return None

    # ---------------------------------------------------------------------- #
    # Tier 1: Jamendo                                                          #
    # ---------------------------------------------------------------------- #
    def _jamendo_search(self, theme: str, duration_sec: int) -> Path | None:
        """
        Search Jamendo for a track matching *theme* and at least *duration_sec*
        long. Tries multiple queries per theme in order, preferring currently
        trending tracks (popularity_month) before falling back to all-time
        popular tracks.

        Returns the local cache path on success, None otherwise.
        """
        cfg = _music_cfg()
        queries = _resolve_queries(theme)
        slack = int(cfg.get("duration_slack_sec", 10))
        min_dur = max(duration_sec - slack, int(cfg.get("min_duration_sec", 30)))  # allow slightly shorter tracks

        # Strategy: try trending first, then popular, across all queries.
        # This mirrors the "native platform trending audio" approach.
        search_attempts = [
            (q, _ORDER_TRENDING) for q in queries
        ] + [
            (q, _ORDER_POPULAR) for q in queries[:2]  # popular fallback for top 2 queries only
        ]

        for tags_query, order in search_attempts:
            log.info(
                "Jamendo search: theme=%r query=%r order=%s",
                theme, tags_query, order,
            )
            try:
                track = self._query_jamendo(tags_query, order, min_dur)
            except RuntimeError as exc:
                log.warning("Jamendo query failed (%s) — trying next query", exc)
                continue

            if track:
                return track

        log.warning("Jamendo: exhausted all queries for theme %r — falling back", theme)
        return None

    def _query_jamendo(
        self, tags_query: str, order: str, min_dur: int
    ) -> Path | None:
        """Execute a single Jamendo API search and return the cached track path."""
        cfg = _music_cfg()
        params = urllib.parse.urlencode({
            "client_id":   self._jamendo_id,
            "format":      "json",
            "limit":       int(cfg.get("page_size", 15)),
            "fuzzytags":   tags_query,   # fuzzytags gives broader matches
            "audioformat": "mp32",       # guaranteed 320 kbps MP3 download
            "order":       order,
            "include":     "musicinfo",
        })
        url = f"{JAMENDO_API}?{params}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "action-clip-bot/2.0"})
            with urllib.request.urlopen(req, timeout=float(cfg.get("legacy_timeout_sec", 15))) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Jamendo network error: {exc}") from exc

        status = data.get("headers", {}).get("status")
        if status != "success":
            warnings = data.get("headers", {}).get("warnings", "")
            raise RuntimeError(f"Jamendo API error: status={status} warnings={warnings}")

        results = data.get("results", [])
        log.info("Jamendo: %d results for query=%r order=%s", len(results), tags_query, order)

        # Filter by minimum duration
        candidates = [t for t in results if int(t.get("duration", 0)) >= min_dur]
        if not candidates:
            candidates = results  # Relax if nothing qualifies
        if not candidates:
            return None

        # Pick first candidate (API returns in requested popularity order)
        track      = candidates[0]
        track_id   = str(track["id"])
        track_name = track.get("name", "unknown")
        artist     = track.get("artist_name", "unknown")
        audio_url  = track.get("audiodownload") or track.get("audio")

        if not audio_url:
            log.warning("Jamendo track %s has no audio URL", track_id)
            return None

        log.info(
            'Jamendo selected: "%s" by %s (%ss) [%s] id=%s',
            track_name, artist, track.get("duration", "?"), order, track_id,
        )

        # Check cache first
        cfg = _music_cfg()
        cache_path = CACHE_DIR / f"jamendo_{track_id}.mp3"
        if cache_path.exists() and cache_path.stat().st_size > int(cfg.get("min_bytes", 50_000)):
            log.info("Jamendo cache hit: %s", cache_path.name)
            return cache_path

        # Download with retry
        return self._download_jamendo(audio_url, cache_path, track_name, artist)

    def _download_jamendo(
        self, url: str, dest: Path, name: str, artist: str
    ) -> Path | None:
        """Download a Jamendo audio URL to *dest* with retry."""
        import httpx
        log.info("Downloading Jamendo track: %s...", url[:80])
        dest.parent.mkdir(parents=True, exist_ok=True)

        headers = {
            "User-Agent": "action-clip-bot/2.0",
            "Accept":     "audio/mpeg, audio/*, */*",
        }
        cfg = _music_cfg()
        _timeout = httpx.Timeout(
            connect=float(cfg.get("connect_timeout_sec", 10)),
            read=float(cfg.get("read_timeout_sec", 15)),
            write=float(cfg.get("write_timeout_sec", 10)),
            pool=float(cfg.get("pool_timeout_sec", 10)),
        )
        retries = int(cfg.get("retries", 3))

        for attempt in range(retries):
            try:
                with httpx.Client(timeout=_timeout, follow_redirects=True) as client:
                    with client.stream("GET", url, headers=headers) as resp:
                        if resp.status_code != 200:
                            raise RuntimeError(f"Jamendo download failed with status {resp.status_code}")
                        
                        # Write stream chunk by chunk
                        temp_dest = dest.with_suffix(".tmp_bgm")
                        with open(temp_dest, "wb") as f:
                            for chunk in resp.iter_bytes(chunk_size=int(cfg.get("chunk_bytes", 65536))):
                                f.write(chunk)
                        
                        file_size = temp_dest.stat().st_size
                        if file_size < int(cfg.get("min_bytes", 50_000)):
                            temp_dest.unlink(missing_ok=True)
                            raise RuntimeError(f"Downloaded file too small ({file_size} bytes)")
                        
                        temp_dest.rename(dest)
                        log.info(
                            'Downloaded "%s" by %s -> %s (%.1f KB)',
                            name, artist, dest.name, file_size / 1024,
                        )
                        return dest
            except Exception as exc:
                log.warning("Jamendo download attempt %d failed: %s", attempt + 1, exc)
                if attempt < retries - 1:
                    time.sleep(int(cfg.get("backoff_base_sec", 2)) ** attempt)
        return None

    # ---------------------------------------------------------------------- #
    # Tier 3: Synthetic drum-loop fallback                                    #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _generate_action_pulse(out_path: Path, duration_sec: int) -> None:
        """
        Generate an audible multi-layer action drum loop as the BGM fallback.

        Four separate ffmpeg passes (keeps aevalsrc expressions simple):
          • Kick  — 80 Hz decaying sine, 4-on-the-floor at 120 BPM
          • Hi-hat — high-pass noise burst, 8th-notes
          • Snare  — band-passed noise + tone on beats 2 & 4
          • Bass   — 55 Hz + harmonics, low-passed

        All layers are mixed and limited to roughly -15 dBFS mean.
        """
        tmp = Path(tempfile.mkdtemp())

        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi",
            "-i", (f"aevalsrc=0.8*sin(2*PI*80*t)*exp(-8*(t*2-floor(t*2)))"
                   f"+0.4*sin(2*PI*160*t)*exp(-15*(t*2-floor(t*2))):s=44100:d={duration_sec}"),
            "-af", "volume=6.0,acompressor=threshold=0.3:ratio=6:attack=1:release=80",
            "-c:a", "flac", str(tmp / "kick.flac"),
        ], check=True)

        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"aevalsrc=0.3*(2*random(0)-1)*exp(-80*(t*4-floor(t*4))):s=44100:d={duration_sec}",
            "-af", "highpass=f=8000,volume=3.0",
            "-c:a", "flac", str(tmp / "hat.flac"),
        ], check=True)

        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi",
            "-i", (f"aevalsrc=0.4*(2*random(1)-1)*exp(-20*(t*2+0.5-floor(t*2+0.5)))"
                   f"+0.2*sin(2*PI*200*t)*exp(-18*(t*2+0.5-floor(t*2+0.5))):s=44100:d={duration_sec}"),
            "-af", "bandpass=f=400:width_type=o:w=3,volume=4.0",
            "-c:a", "flac", str(tmp / "snare.flac"),
        ], check=True)

        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi",
            "-i", (f"aevalsrc=0.5*sin(2*PI*55*t)+0.2*sin(2*PI*110*t)"
                   f"+0.1*sin(2*PI*165*t):s=44100:d={duration_sec}"),
            "-af", "lowpass=f=300,volume=3.0",
            "-c:a", "flac", str(tmp / "bass.flac"),
        ], check=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(tmp / "kick.flac"),
            "-i", str(tmp / "hat.flac"),
            "-i", str(tmp / "snare.flac"),
            "-i", str(tmp / "bass.flac"),
            "-filter_complex",
            ("[0:a][1:a][2:a][3:a]amix=inputs=4:normalize=0,"
             "alimiter=level_in=1:level_out=0.9:limit=0.85:attack=5:release=50,"
             "volume=0.25[a]"),
            "-map", "[a]",
            "-c:a", "libmp3lame", "-b:a", "192k",
            str(out_path),
        ], check=True)
        log.info("Generated synthetic action pulse BGM -> %s", out_path)
