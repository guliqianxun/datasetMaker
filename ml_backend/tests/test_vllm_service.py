"""
Unit tests for app/services/vllm_service.py

All network I/O (httpx) is mocked so tests run without a real vLLM server.
"""

from __future__ import annotations

import json

import pytest

from app.services.vllm_service import build_vision_messages, call_vllm


# ------------------------------------------------------------------ build_vision_messages

class TestBuildVisionMessages:
    def test_structure(self):
        msgs = build_vision_messages("data:image/png;base64,abc", "Find objects.")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_content_has_text_and_image(self):
        msgs = build_vision_messages("data:image/png;base64,xyz", "Detect cars.")
        user_content = msgs[1]["content"]
        types = [c["type"] for c in user_content]
        assert "text" in types
        assert "image_url" in types

    def test_image_url_embedded(self):
        data_uri = "data:image/jpeg;base64,AAAA"
        msgs = build_vision_messages(data_uri, "Test prompt.")
        image_parts = [c for c in msgs[1]["content"] if c["type"] == "image_url"]
        assert image_parts[0]["image_url"]["url"] == data_uri

    def test_text_prompt_included(self):
        msgs = build_vision_messages("data:image/jpeg;base64,X", "My custom prompt")
        text_parts = [c for c in msgs[1]["content"] if c["type"] == "text"]
        assert text_parts[0]["text"] == "My custom prompt"


# ------------------------------------------------------------------ call_vllm

class TestCallVllm:
    def _mock_httpx(self, mocker, response_text: str, status_code: int = 200):
        response_json = {
            "choices": [{"message": {"content": response_text}}]
        }
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = response_json
        mock_response.raise_for_status = mocker.MagicMock()

        mock_client = mocker.MagicMock()
        mock_client.__enter__ = mocker.MagicMock(return_value=mock_client)
        mock_client.__exit__ = mocker.MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mocker.patch("app.services.vllm_service.httpx.Client", return_value=mock_client)
        return mock_client

    def test_returns_model_text(self, mocker):
        expected = '[{"label": "car", "bbox": [0,0,100,100], "score": 0.9}]'
        self._mock_httpx(mocker, expected)

        result = call_vllm("data:image/jpeg;base64,ABC")
        assert result == expected

    def test_sends_correct_model_name(self, mocker):
        from app.config import settings
        self._mock_httpx(mocker, "[]")

        mock_client = mocker.patch("app.services.vllm_service.httpx.Client")
        instance = mock_client.return_value.__enter__.return_value
        instance.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "[]"}}]
        }
        instance.post.return_value.raise_for_status = lambda: None

        call_vllm("data:image/jpeg;base64,X")
        call_args = instance.post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert payload["model"] == settings.vllm_model

    def test_guided_json_included_in_extra_body(self, mocker):
        schema = {"type": "array"}
        mock_client = mocker.patch("app.services.vllm_service.httpx.Client")
        instance = mock_client.return_value.__enter__.return_value
        instance.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "[]"}}]
        }
        instance.post.return_value.raise_for_status = lambda: None

        call_vllm("data:image/jpeg;base64,X", guided_json=schema)
        call_args = instance.post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert "extra_body" in payload
        assert payload["extra_body"]["guided_json"] == schema

    def test_raises_on_bad_response_structure(self, mocker):
        mock_client = mocker.patch("app.services.vllm_service.httpx.Client")
        instance = mock_client.return_value.__enter__.return_value
        instance.post.return_value.json.return_value = {"unexpected": "data"}
        instance.post.return_value.raise_for_status = lambda: None

        with pytest.raises(ValueError):
            call_vllm("data:image/jpeg;base64,X")
