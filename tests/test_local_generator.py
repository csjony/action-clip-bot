import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import httpx

from src.generators.local import LocalGenerator


def test_local_generator_ready_success(tmp_path):
    gen = LocalGenerator(server_url="https://test-server.com")
    
    mock_ready_resp = MagicMock()
    mock_ready_resp.status_code = 200
    
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.json.return_value = {"job_id": "job123"}
    
    mock_status_resp = MagicMock()
    mock_status_resp.status_code = 200
    mock_status_resp.json.return_value = {"status": "done"}
    
    mock_result_resp = MagicMock()
    mock_result_resp.status_code = 200
    mock_result_resp.content = b"fake_mp4_bytes"
    
    def mock_get(url, *args, **kwargs):
        if url.endswith("/ready"):
            return mock_ready_resp
        elif "/status/" in url:
            return mock_status_resp
        elif "/result/" in url:
            return mock_result_resp
        return MagicMock(status_code=404)

    def mock_post(url, *args, **kwargs):
        if url.endswith("/generate"):
            return mock_post_resp
        return MagicMock(status_code=404)

    mock_client_instance = MagicMock()
    mock_client_instance.get.side_effect = mock_get
    mock_client_instance.post.side_effect = mock_post
    mock_client_instance.__enter__.return_value = mock_client_instance

    with patch("httpx.Client", return_value=mock_client_instance), patch("time.sleep", return_value=None):
        out_file = tmp_path / "clip.mp4"
        gen._run(prompt="test prompt", duration_sec=5, out_path=out_file)
        assert out_file.exists()
        assert out_file.read_bytes() == b"fake_mp4_bytes"
        assert gen._model_ready is True


def test_local_generator_ready_fail_fast_500(tmp_path):
    gen = LocalGenerator(server_url="https://test-server.com")
    
    mock_ready_resp = MagicMock()
    mock_ready_resp.status_code = 500
    mock_ready_resp.text = '{"detail": "Model loading failed on GPU server: CUDA out of memory"}'
    mock_ready_resp.json.return_value = {"detail": "Model loading failed on GPU server: CUDA out of memory"}

    mock_client_instance = MagicMock()
    mock_client_instance.get.return_value = mock_ready_resp
    mock_client_instance.__enter__.return_value = mock_client_instance

    with patch("httpx.Client", return_value=mock_client_instance), patch("time.sleep", return_value=None):
        out_file = tmp_path / "clip.mp4"
        with pytest.raises(RuntimeError) as exc_info:
            gen._run(prompt="test prompt", duration_sec=5, out_path=out_file)
        assert "Model loading failed on GPU server: CUDA out of memory" in str(exc_info.value)


def test_local_generator_uses_quality_settings_for_subsequent_clips(tmp_path):
    gen = LocalGenerator(server_url="https://test-server.com")

    mock_ready_resp = MagicMock()
    mock_ready_resp.status_code = 200

    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.json.return_value = {"job_id": "job123"}

    mock_status_resp = MagicMock()
    mock_status_resp.status_code = 200
    mock_status_resp.json.return_value = {"status": "done"}

    mock_result_resp = MagicMock()
    mock_result_resp.status_code = 200
    mock_result_resp.content = b"fake_mp4_bytes"

    def mock_get(url, *args, **kwargs):
        if url.endswith("/ready"):
            return mock_ready_resp
        elif "/status/" in url:
            return mock_status_resp
        elif "/result/" in url:
            return mock_result_resp
        return MagicMock(status_code=404)

    def mock_post(url, *args, **kwargs):
        if url.endswith("/generate"):
            return mock_post_resp
        return MagicMock(status_code=404)

    mock_client_instance = MagicMock()
    mock_client_instance.get.side_effect = mock_get
    mock_client_instance.post.side_effect = mock_post
    mock_client_instance.__enter__.return_value = mock_client_instance

    mock_settings = MagicMock()
    mock_settings.video = {
        "generation": {
            "fps": 16,
            "anchor_steps": 20,
            "subsequent_steps": 12,
            "resolution": {"width": 1280, "height": 720},
        }
    }

    with (
        patch("src.generators.local.get_settings", return_value=mock_settings),
        patch("httpx.Client", return_value=mock_client_instance),
        patch("time.sleep", return_value=None),
    ):
        out_file = tmp_path / "clip.mp4"
        gen._run(prompt="test prompt", duration_sec=5, out_path=out_file, scene_index=1)

    post_payload = mock_client_instance.post.call_args.kwargs["json"]
    assert post_payload["num_steps"] == 12
    assert post_payload["fps"] == 16
    assert post_payload["width"] == 1280
    assert post_payload["height"] == 720


def test_local_generator_includes_negative_prompt(tmp_path):
    gen = LocalGenerator(server_url="https://test-server.com")

    mock_ready_resp = MagicMock()
    mock_ready_resp.status_code = 200

    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.json.return_value = {"job_id": "job456"}

    mock_status_resp = MagicMock()
    mock_status_resp.status_code = 200
    mock_status_resp.json.return_value = {"status": "done"}

    mock_result_resp = MagicMock()
    mock_result_resp.status_code = 200
    mock_result_resp.content = b"fake_mp4_bytes"

    def mock_get(url, *args, **kwargs):
        if url.endswith("/ready"):
            return mock_ready_resp
        elif "/status/" in url:
            return mock_status_resp
        elif "/result/" in url:
            return mock_result_resp
        return MagicMock(status_code=404)

    def mock_post(url, *args, **kwargs):
        if url.endswith("/generate"):
            return mock_post_resp
        return MagicMock(status_code=404)

    mock_client_instance = MagicMock()
    mock_client_instance.get.side_effect = mock_get
    mock_client_instance.post.side_effect = mock_post
    mock_client_instance.__enter__.return_value = mock_client_instance

    mock_settings = MagicMock()
    mock_settings.video = {
        "generation": {
            "fps": 16,
            "anchor_steps": 30,
            "subsequent_steps": 30,
            "resolution": {"width": 1280, "height": 720},
            "negative_prompt": "some custom negative prompt",
        }
    }

    with (
        patch("src.generators.local.get_settings", return_value=mock_settings),
        patch("httpx.Client", return_value=mock_client_instance),
        patch("time.sleep", return_value=None),
    ):
        out_file = tmp_path / "clip.mp4"
        gen._run(prompt="test prompt", duration_sec=5, out_path=out_file, scene_index=0)

    post_payload = mock_client_instance.post.call_args.kwargs["json"]
    assert post_payload["num_steps"] == 30
    assert post_payload["fps"] == 16
    assert post_payload["width"] == 1280
    assert post_payload["height"] == 720
    assert post_payload["negative_prompt"] == "some custom negative prompt"

