import hashlib
from pathlib import Path
import json
import logging
import urllib.request
import urllib.parse

log = logging.getLogger(__name__)


def _sfx_cfg() -> dict:
    """Timeouts/URLs tunables from settings.yaml `sfx_api:` (dashboard-editable)."""
    from src.config import get_settings
    cfg = get_settings().get("sfx_api", {}) or {}
    return cfg if isinstance(cfg, dict) else {}


class SFXPicker:
    def __init__(self, *, freesound_api_key: str | None = None, elevenlabs_api_key: str | None = None, replicate_api_key: str | None = None, provider: str = "freesound", cache_dir: Path | None = None) -> None:
        self.freesound_api_key = freesound_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        self.replicate_api_key = replicate_api_key
        self.provider = provider
        
        if cache_dir is None:
            # Default to cache/sfx in the project root
            self.cache_dir = Path(__file__).resolve().parent.parent.parent / "cache" / "sfx"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def pick(self, query: str, duration_sec: int, out_path: Path, video_path: Path | None = None) -> Path | None:
        """
        Resolve sound effect for a query.
        Returns the path to the sound effect file, or None if skipped/failed.
        """
        if not query or not query.strip():
            return None

        # Clean query
        query = query.strip()
        
        # Check cache
        if self.provider == "mmaudio":
            if not video_path or not video_path.exists():
                log.warning("No video path provided for MMAudio V2A")
                return None
            # Hash video file content + query + duration
            hasher = hashlib.md5()
            with open(video_path, "rb") as f:
                hasher.update(f.read())
            hasher.update(query.encode("utf-8"))
            hasher.update(str(duration_sec).encode("utf-8"))
            cache_hash = hasher.hexdigest()
        else:
            cache_key = f"{self.provider}_{query}_{duration_sec}"
            cache_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()

        cached_file = self.cache_dir / f"{cache_hash}.mp3"
        
        if cached_file.exists() and cached_file.stat().st_size > 0:
            log.info("Using cached SFX for query '%s' -> %s", query, cached_file)
            import shutil
            try:
                shutil.copy(cached_file, out_path)
                return out_path
            except Exception as e:
                log.error("Failed to copy cached file: %s", e)
                return None

        # Generate / download fresh
        try:
            success = False
            if self.provider == "elevenlabs" and self.elevenlabs_api_key:
                success = self._fetch_elevenlabs(query, duration_sec, cached_file)
            elif self.provider == "freesound" and self.freesound_api_key:
                success = self._fetch_freesound(query, duration_sec, cached_file)
            elif self.provider == "mmaudio" and self.replicate_api_key:
                success = self._fetch_mmaudio(query, duration_sec, video_path, cached_file)
            else:
                log.warning("No API key or valid provider set for SFX: provider=%s", self.provider)
                return None
                
            if success and cached_file.exists() and cached_file.stat().st_size > 0:
                import shutil
                shutil.copy(cached_file, out_path)
                return out_path
        except Exception as e:
            log.error("Failed to retrieve SFX for query '%s': %s", query, e)
            
        return None

    def _fetch_elevenlabs(self, query: str, duration_sec: int, dest_path: Path) -> bool:
        import httpx
        cfg = _sfx_cfg()
        url = str(cfg.get("elevenlabs_url", "https://api.elevenlabs.io/v1/sound-effects"))
        headers = {
            "xi-api-key": self.elevenlabs_api_key,
            "Content-Type": "application/json"
        }
        data = {
            "text": query,
            "duration_seconds": float(duration_sec),
            "prompt_influence": float(cfg.get("prompt_influence", 0.3))
        }
        _timeout = httpx.Timeout(connect=float(cfg.get("connect_timeout_sec", 10)),
                                 read=float(cfg.get("elevenlabs_read_sec", 30)),
                                 write=10.0, pool=10.0)
        try:
            with httpx.Client(timeout=_timeout) as client:
                resp = client.post(url, json=data, headers=headers)
                if resp.status_code == 200:
                    dest_path.write_bytes(resp.content)
                    log.info("Successfully generated ElevenLabs SFX for query: %s", query)
                    return True
                else:
                    log.warning("ElevenLabs API failed with status %d: %s", resp.status_code, resp.text)
        except Exception as e:
            log.warning("ElevenLabs API request failed: %s", e)
        return False

    def _fetch_freesound(self, query: str, duration_sec: int, dest_path: Path) -> bool:
        import httpx
        # 1. Search Freesound
        cfg = _sfx_cfg()
        params = {
            "query": query,
            "token": self.freesound_api_key,
            "fields": "id,name,previews",
            "page_size": int(cfg.get("page_size", 1))
        }
        headers = {"User-Agent": "ActionClipBot/1.0"}
        search_url = str(cfg.get("freesound_url", "https://freesound.org/apiv2/search/text/"))
        try:
            with httpx.Client(timeout=float(cfg.get("freesound_search_sec", 15))) as client:
                resp = client.get(search_url, params=params, headers=headers)
                if resp.status_code != 200:
                    log.warning("Freesound search failed with status %d: %s", resp.status_code, resp.text)
                    return False
                data = resp.json()
        except Exception as e:
            log.warning("Freesound search failed: %s", e)
            return False
            
        results = data.get("results", [])
        if not results:
            log.warning("No Freesound matches for query: %s", query)
            return False
            
        preview_url = results[0].get("previews", {}).get("preview-hq-mp3") or results[0].get("previews", {}).get("preview-lq-mp3")
        if not preview_url:
            return False
            
        # 2. Download preview — use streaming so a stalled CDN never hangs indefinitely.
        #    httpx.Timeout applies per-chunk, so even if the server sends the 200 header
        #    and then stops sending bytes, we raise ReadTimeout after `read` seconds.
        _timeout = httpx.Timeout(connect=float(cfg.get("connect_timeout_sec", 10)),
                                 read=float(cfg.get("preview_timeout_sec", 10)),
                                 write=10.0, pool=10.0)
        try:
            with httpx.Client(timeout=_timeout, follow_redirects=True) as client:
                with client.stream("GET", preview_url, headers=headers) as resp_dl:
                    if resp_dl.status_code == 200:
                        with open(dest_path, "wb") as f:
                            for chunk in resp_dl.iter_bytes(chunk_size=int(cfg.get("chunk_bytes", 65536))):
                                f.write(chunk)
                        log.info("Successfully downloaded Freesound SFX preview for query: %s", query)
                        return True
                    else:
                        log.warning("Freesound download failed with status %d", resp_dl.status_code)
        except Exception as e:
            log.warning("Freesound download failed: %s", e)
            # Remove partial file so a zero-byte cache entry never poisons future runs
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)
        return False

    def _fetch_mmaudio(self, query: str, duration_sec: int, video_path: Path, dest_path: Path) -> bool:
        import httpx
        import time
        import base64
        import mimetypes
        import subprocess

        # 1. Base64 encode the video file
        mime, _ = mimetypes.guess_type(str(video_path))
        if not mime:
            mime = "video/mp4"
        with open(video_path, "rb") as f:
            video_data = base64.b64encode(f.read()).decode("utf-8")
        data_uri = f"data:{mime};base64,{video_data}"

        headers = {
            "Authorization": f"Token {self.replicate_api_key}",
            "Content-Type": "application/json"
        }
        
        cfg = _sfx_cfg()
        payload = {
            "version": str(cfg.get("replicate_version", "62871fb59889b2d7c13777f08deb3b36bdff88f7e1d53a50ad7694548a41b484")),
            "input": {
                "prompt": query,
                "video": data_uri,
                "duration": duration_sec
            }
        }

        log.info("Sending prediction request to Replicate for MMAudio V2A...")
        try:
            wait_sec = int(cfg.get("replicate_wait_sec", 90))
            poll_sec = int(cfg.get("replicate_poll_sec", 2))
            with httpx.Client(timeout=float(cfg.get("replicate_timeout_sec", 30))) as client:
                resp = client.post(
                    "https://api.replicate.com/v1/predictions",
                    headers=headers,
                    json=payload
                )
                if resp.status_code != 201:
                    log.warning("Replicate prediction creation failed: %d %s", resp.status_code, resp.text)
                    return False
                
                pred_data = resp.json()
                poll_url = pred_data.get("urls", {}).get("get")
                if not poll_url:
                    log.warning("No poll URL in Replicate prediction response")
                    return False

                # Poll prediction status until the configured ceiling
                start_time = time.time()
                while time.time() - start_time < wait_sec:
                    poll_resp = client.get(poll_url, headers=headers)
                    if poll_resp.status_code != 200:
                        log.warning("Replicate prediction poll failed: %d", poll_resp.status_code)
                        return False
                    
                    status_data = poll_resp.json()
                    status = status_data.get("status")
                    if status == "succeeded":
                        output = status_data.get("output")
                        audio_url = None
                        if isinstance(output, list) and len(output) > 0:
                            audio_url = next((u for u in output if not u.endswith(".mp4")), output[0])
                        elif isinstance(output, str):
                            audio_url = output
                        
                        if not audio_url:
                            log.warning("Replicate output is empty or invalid format: %s", output)
                            return False
                        
                        log.info("MMAudio succeeded, downloading result from %s...", audio_url)
                        temp_audio_path = dest_path.with_suffix(".tmp_audio")
                        dl_resp = client.get(audio_url)
                        if dl_resp.status_code != 200:
                            log.warning("Failed to download MMAudio generated audio: %d", dl_resp.status_code)
                            return False
                        
                        with open(temp_audio_path, "wb") as f:
                            f.write(dl_resp.content)
                        
                        # Convert to MP3 using FFmpeg
                        try:
                            cmd = [
                                "ffmpeg", "-y",
                                "-i", str(temp_audio_path),
                                "-c:a", "libmp3lame",
                                "-q:a", str(cfg.get("mp3_quality", 2)),
                                "-vn",
                                str(dest_path)
                            ]
                            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            log.info("Successfully converted MMAudio output to MP3 and cached: %s", dest_path)
                            return True
                        except Exception as e:
                            log.error("Failed to convert MMAudio output to MP3: %s", e)
                            return False
                        finally:
                            if temp_audio_path.exists():
                                temp_audio_path.unlink()
                    elif status == "failed":
                        log.warning("Replicate prediction failed: %s", status_data.get("error"))
                        return False
                    
                    time.sleep(poll_sec)

                log.warning("Replicate prediction timed out after %d seconds", wait_sec)
                return False
        except Exception as e:
            log.warning("Failed calling Replicate API for MMAudio: %s", e)
            return False

