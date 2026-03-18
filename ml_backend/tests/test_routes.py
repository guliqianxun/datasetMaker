"""
Integration-style tests for the FastAPI routes using httpx.AsyncClient / TestClient.

All downstream services (MinIO, vLLM) are mocked so no real infrastructure
is needed.
"""

from __future__ import annotations

import base64
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


# ------------------------------------------------------------------ helpers

def _make_tiny_jpeg(width: int = 100, height: int = 100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(128, 64, 32)).save(buf, format="JPEG")
    return buf.getvalue()


def _data_uri(raw: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


# ------------------------------------------------------------------ /health

class TestHealth:
    def test_returns_up(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "UP"}


# ------------------------------------------------------------------ /setup

class TestSetup:
    def test_returns_model_version(self):
        resp = client.post("/setup", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "model_version" in data

    def test_no_body_still_works(self):
        resp = client.post("/setup")
        assert resp.status_code == 200


# ------------------------------------------------------------------ /predict

class TestPredict:
    def _patch_services(self, mocker, width: int = 100, height: int = 100):
        """Patch download_image_as_base64 and call_vllm for one image."""
        raw = _make_tiny_jpeg(width, height)
        data_uri = _data_uri(raw)

        mocker.patch(
            "app.api.routes.download_image_as_base64",
            return_value=(data_uri, "image/jpeg"),
        )
        mocker.patch(
            "app.api.routes.call_vllm",
            return_value=json.dumps([
                {"label": "car", "bbox": [0, 0, width // 2, height // 2], "score": 0.95},
            ]),
        )

    def test_returns_results_list(self, mocker):
        self._patch_services(mocker)
        body = {
            "tasks": [{"id": 1, "data": {"image": "http://example.com/img.jpg"}}],
        }
        resp = client.post("/predict", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1

    def test_result_contains_rectanglelabels(self, mocker):
        self._patch_services(mocker)
        body = {
            "tasks": [{"id": 1, "data": {"image": "http://example.com/img.jpg"}}],
        }
        resp = client.post("/predict", json=body)
        results = resp.json()["results"][0]["result"]
        assert len(results) == 1
        assert results[0]["type"] == "rectanglelabels"

    def test_percentage_coordinates_in_range(self, mocker):
        self._patch_services(mocker, width=200, height=150)
        body = {
            "tasks": [{"id": 2, "data": {"image": "s3://bucket/img.jpg"}}],
        }
        resp = client.post("/predict", json=body)
        value = resp.json()["results"][0]["result"][0]["value"]
        for key in ("x", "y", "width", "height"):
            assert 0.0 <= value[key] <= 100.0, f"{key} out of range"

    def test_task_without_image_returns_empty_result(self, mocker):
        body = {"tasks": [{"id": 3, "data": {"text": "no image here"}}]}
        resp = client.post("/predict", json=body)
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["result"] == []
        assert result["score"] == 0.0

    def test_empty_tasks_returns_empty_results(self, mocker):
        body = {"tasks": []}
        resp = client.post("/predict", json=body)
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_label_config_from_to_names_respected(self, mocker):
        raw = _make_tiny_jpeg(100, 100)
        data_uri = _data_uri(raw)
        mocker.patch(
            "app.api.routes.download_image_as_base64",
            return_value=(data_uri, "image/jpeg"),
        )
        mocker.patch(
            "app.api.routes.call_vllm",
            return_value=json.dumps([
                {"label": "truck", "bbox": [0, 0, 50, 50], "score": 0.8},
            ]),
        )
        label_config = (
            '<View>'
            '<RectangleLabels name="bbox_label" toName="photo">'
            '<Label value="Truck"/>'
            '</RectangleLabels>'
            '<Image name="photo" value="$image"/>'
            '</View>'
        )
        body = {
            "tasks": [{"id": 4, "data": {"image": "http://example.com/img.jpg"}}],
            "label_config": label_config,
        }
        resp = client.post("/predict", json=body)
        result = resp.json()["results"][0]["result"][0]
        assert result["from_name"] == "bbox_label"
        assert result["to_name"] == "photo"

    def test_vllm_error_returns_502(self, mocker):
        import httpx as _httpx

        raw = _make_tiny_jpeg()
        data_uri = _data_uri(raw)
        mocker.patch(
            "app.api.routes.download_image_as_base64",
            return_value=(data_uri, "image/jpeg"),
        )
        mocker.patch(
            "app.api.routes.call_vllm",
            side_effect=RuntimeError("vLLM unreachable"),
        )
        body = {"tasks": [{"id": 5, "data": {"image": "http://example.com/img.jpg"}}]}
        resp = client.post("/predict", json=body)
        assert resp.status_code == 502
