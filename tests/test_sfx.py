import pytest
from pathlib import Path
from unittest.mock import MagicMock
import json

from src.content.sfx import SFXPicker
from src.compose.editor import Editor, SceneSFX


def test_sfx_picker_cache_hit(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    
    import hashlib
    query = "heavy rain"
    duration = 6
    provider = "freesound"
    cache_key = f"{provider}_{query}_{duration}"
    cache_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
    cached_file = cache_dir / f"{cache_hash}.mp3"
    cached_file.write_bytes(b"dummy_mp3_content")
    
    picker = SFXPicker(freesound_api_key="fake_key", provider=provider, cache_dir=cache_dir)
    out_path = tmp_path / "out.mp3"
    res = picker.pick(query, duration, out_path)
    
    assert res == out_path
    assert out_path.exists()
    assert out_path.read_bytes() == b"dummy_mp3_content"


def test_sfx_picker_freesound_api(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    
    import httpx
    mock_calls = []

    class MockStreamResponse:
        def __init__(self, data, status_code=200):
            self.data = data
            self.status_code = status_code
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
            
        def iter_bytes(self, chunk_size=None):
            yield self.data

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
            
        def get(self, url, params=None, headers=None, **kwargs):
            # Reconstruct URL with params to match the asserts
            full_url = url
            if params:
                import urllib.parse
                full_url = f"{url}?{urllib.parse.urlencode(params)}"
            mock_calls.append(full_url)
            
            if "search/text" in url:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "results": [
                        {
                            "id": 12345,
                            "name": "rain sound",
                            "previews": {
                                "preview-hq-mp3": "https://freesound.org/preview-hq.mp3"
                            }
                        }
                    ]
                }
                return mock_resp
            return MagicMock(status_code=404)
            
        def stream(self, method, url, headers=None, **kwargs):
            mock_calls.append(url)
            if "preview-hq.mp3" in url:
                return MockStreamResponse(b"preview_audio_data")
            return MockStreamResponse(b"", status_code=404)

    monkeypatch.setattr(httpx, "Client", MockClient)

    picker = SFXPicker(freesound_api_key="fake_token", provider="freesound", cache_dir=cache_dir)
    out_path = tmp_path / "out.mp3"
    res = picker.pick("heavy rain", 6, out_path)

    assert res == out_path
    assert out_path.exists()
    assert out_path.read_bytes() == b"preview_audio_data"
    assert len(mock_calls) == 2
    assert "token=fake_token" in mock_calls[0]
    assert "preview-hq.mp3" in mock_calls[1]


def test_sfx_picker_elevenlabs_api(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"

    import httpx
    mock_calls = []

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def post(self, url, json=None, headers=None, **kwargs):
            mock_calls.append((url, json, headers))
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b"elevenlabs_audio_data"
            return mock_resp

    monkeypatch.setattr(httpx, "Client", MockClient)

    picker = SFXPicker(elevenlabs_api_key="fake_el_key", provider="elevenlabs", cache_dir=cache_dir)
    out_path = tmp_path / "out.mp3"
    res = picker.pick("heavy rain", 6, out_path)

    assert res == out_path
    assert out_path.exists()
    assert out_path.read_bytes() == b"elevenlabs_audio_data"
    assert len(mock_calls) == 1
    url, json_data, headers = mock_calls[0]
    assert url == "https://api.elevenlabs.io/v1/sound-effects"
    assert headers.get("xi-api-key") == "fake_el_key"


def test_sfx_picker_no_keys(tmp_path):
    picker = SFXPicker(provider="freesound", cache_dir=tmp_path / "cache")
    out_path = tmp_path / "out.mp3"
    res = picker.pick("heavy rain", 6, out_path)
    assert res is None
    assert not out_path.exists()


def test_editor_mix_sfx_tracks(tmp_path, monkeypatch):
    editor = Editor()
    
    mock_run_ffmpeg = MagicMock()
    monkeypatch.setattr("src.compose.editor._run_ffmpeg", mock_run_ffmpeg)
    
    sfx1 = tmp_path / "sfx1.mp3"
    sfx1.write_bytes(b"sfx1")
    sfx2 = tmp_path / "sfx2.mp3"
    sfx2.write_bytes(b"sfx2")
    
    sfx_list = [
        SceneSFX(path=sfx1, start_sec=0.0, duration_sec=5.0),
        SceneSFX(path=sfx2, start_sec=5.0, duration_sec=6.0),
    ]
    
    res = editor._mix_sfx_tracks(sfx_list, tmp_path, target_duration=11.0)
    assert res == tmp_path / "_sfx_track.m4a"
    
    mock_run_ffmpeg.assert_called_once()
    args = mock_run_ffmpeg.call_args[0][0]
    
    assert "-i" in args
    filter_complex = args[args.index("-filter_complex") + 1]
    assert "adelay=0:all=true" in filter_complex
    assert "adelay=5000:all=true" in filter_complex
    assert "amix=inputs=2" in filter_complex


def test_editor_compose_splits_vertical(tmp_path, monkeypatch):
    editor = Editor()
    
    mock_run_ffmpeg = MagicMock()
    monkeypatch.setattr("src.compose.editor._run_ffmpeg", mock_run_ffmpeg)
    
    mock_ffprobe_duration = MagicMock(return_value=72.0)
    monkeypatch.setattr("src.compose.editor._ffprobe_duration", mock_ffprobe_duration)
    
    monkeypatch.setattr(editor, "_concat_with_outro", lambda inputs, out_dir: tmp_path / "concat.mp4")
    monkeypatch.setattr(editor, "_mix_audio", lambda inputs, out_dir, target_duration: tmp_path / "mixed.m4a")
    
    # Mock _render_variant to write a dummy file
    def mock_render_variant(concat, audio, captions, out_path, w, h):
        out_path.write_bytes(b"dummy_rendered")
    monkeypatch.setattr(editor, "_render_variant", mock_render_variant)
    monkeypatch.setattr(editor, "_normalise_loudness", lambda out_path: None)
    
    from src.compose.editor import CompositionInputs
    inputs = CompositionInputs(clips=[], narration_audio=None, music_audio=None, captions_ass=None, outro_image=None)
    
    res = editor.compose(inputs, tmp_path, render=("vertical",))
    
    assert res.vertical == tmp_path / "vertical.mp4"
    assert (tmp_path / "vertical.mp4").exists()
    
    # Verify that splitting was called and the parts would be generated
    assert mock_run_ffmpeg.call_count == 2
    
    # Check the args passed to _run_ffmpeg for the splitting commands
    call1_args = mock_run_ffmpeg.call_args_list[0][0][0]
    call2_args = mock_run_ffmpeg.call_args_list[1][0][0]
    
    assert "-ss" in call1_args
    assert "-to" in call1_args
    assert "36.000" in call1_args  # half of 72.0
    assert "-c" in call1_args
    assert "copy" in call1_args
    
    assert "-ss" in call2_args
    assert "36.000" in call2_args
    assert "-c" in call2_args
    assert "copy" in call2_args


def test_sfx_picker_mmaudio_api(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"mock_video_bytes")
    
    import httpx
    
    mock_requests = []
    
    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
            
        def post(self, url, headers=None, json=None, **kwargs):
            mock_requests.append(("POST", url, json))
            return MagicMock(
                status_code=201,
                json=lambda: {"id": "pred_123", "status": "starting", "urls": {"get": "https://api.replicate.com/pred_123"}}
            )
            
        def get(self, url, headers=None, **kwargs):
            mock_requests.append(("GET", url, None))
            if url == "https://api.replicate.com/pred_123":
                return MagicMock(
                    status_code=200,
                    json=lambda: {"status": "succeeded", "output": "https://replicate.delivery/audio.flac"}
                )
            elif url == "https://replicate.delivery/audio.flac":
                return MagicMock(
                    status_code=200,
                    content=b"flac_bytes"
                )
            return MagicMock(status_code=404)
            
    monkeypatch.setattr(httpx, "Client", MockClient)
    
    # Mock subprocess.run for converting flac to mp3
    mock_run = MagicMock()
    import subprocess
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    def side_effect_run(cmd, *args, **kwargs):
        dest_file = cmd[-1]
        Path(dest_file).write_bytes(b"dummy_converted_mp3")
    mock_run.side_effect = side_effect_run

    picker = SFXPicker(replicate_api_key="fake_replicate_key", provider="mmaudio", cache_dir=cache_dir)
    out_path = tmp_path / "out.mp3"
    res = picker.pick("explosion", 5, out_path, video_path=video_path)
    
    assert res == out_path
    assert out_path.exists()
    assert out_path.read_bytes() == b"dummy_converted_mp3"
    
    # Verify requests sent
    assert len(mock_requests) == 3
    assert mock_requests[0][0] == "POST"
    assert mock_requests[0][1] == "https://api.replicate.com/v1/predictions"
    assert mock_requests[0][2]["input"]["prompt"] == "explosion"
    assert mock_requests[0][2]["input"]["duration"] == 5
    assert mock_requests[0][2]["input"]["video"].startswith("data:video/mp4;base64,")


def test_xfade_chain_two_clips(tmp_path, monkeypatch):
    """Verify the xfade filter chain for 2 clips produces correct filter_complex."""
    editor = Editor()

    captured_args = []
    def mock_run_ffmpeg(args):
        captured_args.append(args)
    monkeypatch.setattr("src.compose.editor._run_ffmpeg", mock_run_ffmpeg)

    clip0 = tmp_path / "c0.mp4"
    clip1 = tmp_path / "c1.mp4"
    clip0.write_bytes(b"c0")
    clip1.write_bytes(b"c1")

    out = tmp_path / "out.mp4"
    editor._xfade_chain(
        clips=[clip0, clip1],
        durations=[6.0, 6.0],
        td=0.5,
        tt="fade",
        out_path=out,
    )

    assert len(captured_args) == 1
    args = captured_args[0]
    # Should have 2 inputs
    assert args.count("-i") == 2
    fc_idx = args.index("-filter_complex")
    fc = args[fc_idx + 1]
    # Single xfade filter: offset = 6.0 - 0.5 = 5.5
    assert "xfade=transition=fade:duration=0.500:offset=5.500[vout]" in fc
    assert "[0:v]" in fc
    assert "[1:v]" in fc
    # Should map [vout]
    assert "[vout]" in args


def test_xfade_chain_four_clips(tmp_path, monkeypatch):
    """Verify chained xfade for 4 clips produces 3 filter stages."""
    editor = Editor()

    captured_args = []
    def mock_run_ffmpeg(args):
        captured_args.append(args)
    monkeypatch.setattr("src.compose.editor._run_ffmpeg", mock_run_ffmpeg)

    clips = []
    for i in range(4):
        c = tmp_path / f"c{i}.mp4"
        c.write_bytes(f"c{i}".encode())
        clips.append(c)

    out = tmp_path / "out.mp4"
    editor._xfade_chain(
        clips=clips,
        durations=[6.0, 6.0, 6.0, 2.0],
        td=0.5,
        tt="circleopen",
        out_path=out,
    )

    args = captured_args[0]
    assert args.count("-i") == 4
    fc_idx = args.index("-filter_complex")
    fc = args[fc_idx + 1]
    # 3 xfade filters chained:
    #   offset_1 = 6.0 - 0.5 = 5.5
    #   offset_2 = 6.0 + 6.0 - 0.5 - 0.5 = 11.0
    #   offset_3 = 6.0 + 6.0 + 6.0 - 0.5 - 0.5 - 0.5 = 16.5
    parts = fc.split(";")
    assert len(parts) == 3
    assert "xfade=transition=circleopen" in parts[0]
    assert "[xf1]" in parts[0]
    assert "[xf2]" in parts[1]
    assert "[vout]" in parts[2]
    # Verify offsets
    assert "offset=5.500" in parts[0]
    assert "offset=11.000" in parts[1]
    assert "offset=16.500" in parts[2]


def test_editor_transition_fallback_single_clip(tmp_path, monkeypatch):
    """With only 1 clip, Editor should fall back to concat-demuxer (no xfade)."""
    editor = Editor(transition_type="fade", transition_duration_sec=0.5)

    captured_args = []
    def mock_run_ffmpeg(args):
        captured_args.append(args)
    monkeypatch.setattr("src.compose.editor._run_ffmpeg", mock_run_ffmpeg)
    monkeypatch.setattr("src.compose.editor._ffprobe_duration", lambda p: 6.0)
    monkeypatch.setattr("src.compose.editor._ffprobe_resolution", lambda p: (1280, 720))

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video_data")

    from src.compose.editor import CompositionInputs
    inputs = CompositionInputs(
        clips=[clip],
        narration_audio=None,
        music_audio=None,
        captions_ass=None,
        outro_image=None,
    )
    result = editor._concat_with_outro(inputs, tmp_path)

    # Should have used concat demuxer (the normalisation call + the concat call)
    # The normalize step generates 1 ffmpeg call, then the concat step generates 1 more
    assert len(captured_args) == 2
    concat_args = captured_args[-1]
    assert "-f" in concat_args
    assert "concat" in concat_args
    # Should NOT contain filter_complex / xfade
    assert "-filter_complex" not in concat_args




class TestSelfHostedFoley:
    """_fetch_foley uses the async job API (submit → poll → download)."""

    def _client(self, monkeypatch, status_seq, dl_bytes=b"RIFF....WAVEfake"):
        import httpx as _httpx_mod

        calls = {"n": 0}

        class FakeResp:
            def __init__(self, status, json_data=None, content=b""):
                self.status_code = status
                self._json = json_data or {}
                self.text = str(json_data)
                self.content = content
            def json(self):
                return self._json

        class FakeClient:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def post(self, url, **kwargs):
                assert url.endswith("/foley")
                assert "video" in (kwargs.get("files") or {})
                return FakeResp(200, {"job_id": "job1", "status": "queued"})
            def get(self, url, **kwargs):
                if url.endswith("/foley-status/job1"):
                    calls["n"] += 1
                    item = status_seq[min(calls["n"] - 1, len(status_seq) - 1)]
                    if isinstance(item, Exception):
                        raise item
                    return FakeResp(200, {"status": item})
                if url.endswith("/foley-result/job1"):
                    return FakeResp(200, content=dl_bytes)
                return FakeResp(404)

        monkeypatch.setattr(_httpx_mod, "Client", FakeClient)
        import time as _time_mod
        monkeypatch.setattr(_time_mod, "sleep", lambda s: None)
        return calls

    def _run(self, monkeypatch, tmp_path, status_seq):
        import subprocess
        clip = tmp_path / "clip_0.mp4"
        clip.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100)
        dest = tmp_path / "sfx_1.mp3"
        calls = self._client(monkeypatch, status_seq)

        def fake_run(cmd, **kwargs):
            assert cmd[0] == "ffmpeg"
            Path(cmd[-1]).write_bytes(b"ID3fake-mp3")
            from unittest.mock import MagicMock
            return MagicMock(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr("src.generators.colab.resolve_colab_url",
                            lambda *a, **k: "https://tunnel.example")
        picker = SFXPicker(provider="foley")
        return picker, clip, dest, calls

    def test_async_flow_posts_polls_downloads(self, tmp_path, monkeypatch):
        picker, clip, dest, calls = self._run(
            monkeypatch, tmp_path, ["running", "done"])
        assert picker._fetch_foley("heavy rain", 5, clip, dest) is True
        assert calls["n"] == 2
        assert dest.exists()

    def test_failed_job_skips(self, tmp_path, monkeypatch):
        picker, clip, dest, _ = self._run(
            monkeypatch, tmp_path, ["running", "failed"])
        assert picker._fetch_foley("rain", 5, clip, dest) is False
        assert not dest.exists()

    def test_missing_tunnel_skips(self, tmp_path, monkeypatch):
        clip = tmp_path / "clip_0.mp4"
        clip.write_bytes(b"x")
        monkeypatch.setattr("src.generators.colab.resolve_colab_url", lambda *a, **k: "")
        picker = SFXPicker(provider="foley")
        assert picker._fetch_foley("rain", 5, clip, tmp_path / "o.mp3") is False
