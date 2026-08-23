"""
Tests to verify that LLM (Gemini, Groq) and music (Jamendo) API keys can be
loaded dynamically from database accounts, falling back to env vars.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import get_settings
from src.content.scriptwriter import Scriptwriter
from src.dashboard.accounts import AccountStore
from src.pipeline import Pipeline
from src.store import Store


@pytest.mark.skip(reason="LLM script generation backends are disabled in the current manual-script-first workflow")
def test_scriptwriter_gemini_groq_from_db(tmp_env):
    store = Store(tmp_env / "state.db")

    # 1. Add gemini and groq accounts to the store
    acct_store = AccountStore(store)
    acct_store.add("gemini", "my_gemini_label", "db_gemini_api_key")
    acct_store.add("groq", "my_groq_label", "db_groq_api_key")

    # 2. Instantiate Scriptwriter passing the store
    sw = Scriptwriter(get_settings(), store=store)

    # 3. Check resolved backends
    backends = sw._backends

    # Filter backends by class name
    gemini_backends = [b for b in backends if b.__class__.__name__ == "GeminiBackend"]
    groq_backends = [b for b in backends if b.__class__.__name__ == "GroqBackend"]

    assert len(gemini_backends) == 1
    assert gemini_backends[0].api_key == "db_gemini_api_key"

    assert len(groq_backends) == 1
    assert groq_backends[0].api_key == "db_groq_api_key"

    store.close()


def test_pipeline_jamendo_from_db(tmp_env, monkeypatch):
    store = Store(tmp_env / "state.db")

    # 1. Add jamendo account to the store
    acct_store = AccountStore(store)
    acct_store.add("jamendo", "my_jamendo_label", "db_jamendo_api_key")

    # Mock MusicPicker to inspect the key passed during instantiation
    mock_picker_cls = MagicMock()
    monkeypatch.setattr("src.pipeline.MusicPicker", mock_picker_cls)

    # Mock Captioner and Narrator to prevent model loading and API calls
    mock_captioner_cls = MagicMock()
    mock_captioner_cls.return_value.transcribe.side_effect = RuntimeError("whisper disabled in test")
    monkeypatch.setattr("src.pipeline.Captioner", mock_captioner_cls)

    mock_narrator_cls = MagicMock()
    mock_narrator_cls.return_value.synthesize.side_effect = RuntimeError("narrator disabled in test")
    monkeypatch.setattr("src.pipeline.Narrator", mock_narrator_cls)

    # Mock other pipeline dependencies to run _build_audio_captions in isolation
    pipeline = Pipeline(dry_run=True, theme="test", run_id="test_run")
    pipeline.store = store

    plan = MagicMock()
    plan.narration = "Hello world"
    plan.total_duration_sec = 10

    # Mock fallback audio & captions functions to avoid subprocesses
    monkeypatch.setattr(pipeline, "_fallback_audio", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_fallback_captions", lambda *args, **kwargs: Path("captions.ass"))
    monkeypatch.setattr("src.pipeline.make_outro_image", lambda *args, **kwargs: None)

    # Run the audio captions composition phase
    pipeline._build_audio_captions(plan)

    # Assert that MusicPicker was instantiated with the API key from DB plus account_id and store
    mock_picker_cls.assert_called_once_with(account_id=1, store=store, jamendo_client_id="db_jamendo_api_key")

    store.close()


def test_pipeline_disable_voice(tmp_env, monkeypatch):
    store = Store(tmp_env / "state.db")

    # Mock MusicPicker to return a dummy path
    mock_picker_cls = MagicMock()
    mock_picker_cls.return_value.pick.return_value = Path("music.mp3")
    monkeypatch.setattr("src.pipeline.MusicPicker", mock_picker_cls)

    # Mock Captioner and Narrator to prevent model loading/API calls
    mock_captioner_cls = MagicMock()
    monkeypatch.setattr("src.pipeline.Captioner", mock_captioner_cls)

    mock_narrator_cls = MagicMock()
    monkeypatch.setattr("src.pipeline.Narrator", mock_narrator_cls)

    # Force enable_voice to False in pipeline settings
    pipeline = Pipeline(dry_run=True, theme="test", run_id="test_run")
    pipeline.store = store
    pipeline.settings.video["enable_voice"] = False

    plan = MagicMock()
    plan.narration = "Hello world"
    plan.total_duration_sec = 10

    # Mock fallback captions function to avoid subprocesses
    mock_fallback_captions = MagicMock(return_value=Path("captions.ass"))
    monkeypatch.setattr(pipeline, "_fallback_captions", mock_fallback_captions)
    monkeypatch.setattr("src.pipeline.make_outro_image", lambda *args, **kwargs: None)

    # Run the audio captions composition phase
    narration, captions, music, sfx_list, outro = pipeline._build_audio_captions(plan)

    # Assert narration is None
    assert narration is None
    assert isinstance(sfx_list, list)
    # Assert Narrator was never called
    mock_narrator_cls.return_value.synthesize.assert_not_called()
    # Assert Whisper (Captioner) was never called (since no narration to transcribe)
    mock_captioner_cls.return_value.transcribe.assert_not_called()
    # Assert fallback captions were built
    mock_fallback_captions.assert_called_once_with("Hello world", pipeline.run_dir / "assets" / "captions.ass")

    store.close()



