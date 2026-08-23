import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx

from src.generators.runpod_manager import RunPodManager
from src.generators.local import LocalGenerator


class MockResponse:
    def __init__(self, json_data, status_code=200, text=""):
        self._json_data = json_data
        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8") if isinstance(text, str) else text

    def json(self):
        return self._json_data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_runpod_manager_get_status(monkeypatch):
    manager = RunPodManager(api_key="fake_key", pod_id="fake_pod_id")
    
    mock_get = MagicMock(return_value=MockResponse({"status": "RUNNING"}))
    monkeypatch.setattr(httpx.Client, "get", mock_get)

    status = manager.get_status()
    assert status == "RUNNING"
    mock_get.assert_called_once_with("https://rest.runpod.io/v1/pods/fake_pod_id")


def test_runpod_manager_start_pod(monkeypatch):
    manager = RunPodManager(api_key="fake_key", pod_id="fake_pod_id")
    
    mock_post = MagicMock(return_value=MockResponse({}, status_code=200))
    monkeypatch.setattr(httpx.Client, "post", mock_post)

    # First get_status returns HALTED, second returns RUNNING
    status_responses = [
        MockResponse({"status": "HALTED"}),
        MockResponse({"status": "RUNNING"})
    ]
    mock_get = MagicMock(side_effect=status_responses)
    monkeypatch.setattr(httpx.Client, "get", mock_get)

    # Mock bootstrap to prevent real SSH connection / loop
    mock_bootstrap = MagicMock()
    monkeypatch.setattr(manager, "_bootstrap_gpu_server", mock_bootstrap)

    # Monkeypatch time.sleep to run instantly
    monkeypatch.setattr("time.sleep", lambda x: None)

    url = manager.start_pod()
    assert url == "https://fake_pod_id-8000.proxy.runpod.net"
    assert mock_post.call_count == 1
    assert mock_get.call_count == 2
    assert mock_bootstrap.call_count == 1


def test_runpod_manager_stop_pod(monkeypatch):
    manager = RunPodManager(api_key="fake_key", pod_id="fake_pod_id")
    
    mock_post = MagicMock(return_value=MockResponse({}, status_code=200))
    monkeypatch.setattr(httpx.Client, "post", mock_post)

    manager.stop_pod()
    mock_post.assert_called_once_with("https://rest.runpod.io/v1/pods/fake_pod_id/stop")


def test_local_generator_autostarts_runpod(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake_key")
    monkeypatch.setenv("RUNPOD_POD_ID", "fake_pod_id")

    # Mock RunPodManager start_pod
    mock_start_pod = MagicMock(return_value="https://resolved-pod-url.com")
    monkeypatch.setattr(RunPodManager, "start_pod", mock_start_pod)
    monkeypatch.setattr("time.sleep", lambda x: None)

    # Mock ready check and generation request to succeed
    mock_post = MagicMock(return_value=MockResponse({"job_id": "test_job_123"}))
    
    # Mock polling job status to return done
    def mock_status_poll(*args, **kwargs):
        url = args[1]
        if "ready" in url:
            return MockResponse({}, status_code=200)
        elif "status" in url:
            return MockResponse({"status": "done"})
        elif "result" in url:
            return MockResponse({}, text="dummy_video_bytes")
        return MockResponse({}, status_code=404)
        
    monkeypatch.setattr(httpx.Client, "get", mock_status_poll)
    monkeypatch.setattr(httpx.Client, "post", mock_post)

    # Initialize generator with empty string url
    generator = LocalGenerator(server_url="")
    out_path = tmp_path / "clip.mp4"
    
    generator.generate("scenic view of Eiffel tower", duration_sec=6, out_path=out_path)
    
    assert mock_start_pod.call_count == 1
    assert generator.env_value == "https://resolved-pod-url.com"
