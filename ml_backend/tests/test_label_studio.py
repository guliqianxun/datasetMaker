"""
Unit tests for app/services/label_studio.py

Tests cover:
  * JSON extraction from raw VLM text
  * Detection parsing (happy path + edge cases)
  * Coordinate conversion
  * Full Label Studio result builder
"""

from __future__ import annotations

import json
import math

import pytest

from app.services.label_studio import (
    _extract_json_array,
    abs_to_ls_percent,
    build_ls_result,
    parse_vllm_detections,
)


# ------------------------------------------------------------------ _extract_json_array

class TestExtractJsonArray:
    def test_plain_json(self):
        text = '[{"label": "car", "bbox": [10, 20, 100, 200], "score": 0.9}]'
        result = _extract_json_array(text)
        assert isinstance(result, list)
        assert result[0]["label"] == "car"

    def test_markdown_fenced(self):
        text = "```json\n[{\"label\": \"person\", \"bbox\": [0,0,50,50], \"score\": 0.8}]\n```"
        result = _extract_json_array(text)
        assert result[0]["label"] == "person"

    def test_prose_before_json(self):
        text = "Here are the detections:\n[{\"label\": \"bike\", \"bbox\": [5,5,20,30], \"score\": 0.7}]"
        result = _extract_json_array(text)
        assert result[0]["label"] == "bike"

    def test_empty_array(self):
        result = _extract_json_array("[]")
        assert result == []

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _extract_json_array("not json at all")


# ------------------------------------------------------------------ parse_vllm_detections

class TestParseVllmDetections:
    def test_valid_detections(self):
        text = json.dumps([
            {"label": "car", "bbox": [10, 20, 110, 120], "score": 0.95},
            {"label": "person", "bbox": [200, 150, 280, 400], "score": 0.85},
        ])
        dets = parse_vllm_detections(text)
        assert len(dets) == 2
        assert dets[0]["label"] == "car"
        assert dets[0]["score"] == pytest.approx(0.95)

    def test_missing_score_defaults_to_one(self):
        text = json.dumps([{"label": "dog", "bbox": [0, 0, 100, 100]}])
        dets = parse_vllm_detections(text)
        assert dets[0]["score"] == pytest.approx(1.0)

    def test_malformed_bbox_skipped(self):
        text = json.dumps([
            {"label": "car", "bbox": [10, 20, 110]},       # only 3 coords
            {"label": "person", "bbox": [0, 0, 50, 50], "score": 0.9},
        ])
        dets = parse_vllm_detections(text)
        assert len(dets) == 1
        assert dets[0]["label"] == "person"

    def test_non_dict_items_skipped(self):
        text = json.dumps(["not a dict", {"label": "cat", "bbox": [1, 2, 3, 4], "score": 0.5}])
        dets = parse_vllm_detections(text)
        assert len(dets) == 1

    def test_empty_array(self):
        dets = parse_vllm_detections("[]")
        assert dets == []


# ------------------------------------------------------------------ abs_to_ls_percent

class TestAbsToLsPercent:
    def test_full_image(self):
        pct = abs_to_ls_percent(0, 0, 100, 200, 100, 200)
        assert pct["x"] == pytest.approx(0.0)
        assert pct["y"] == pytest.approx(0.0)
        assert pct["width"] == pytest.approx(100.0)
        assert pct["height"] == pytest.approx(100.0)

    def test_centre_box(self):
        pct = abs_to_ls_percent(250, 250, 750, 750, 1000, 1000)
        assert pct["x"] == pytest.approx(25.0)
        assert pct["y"] == pytest.approx(25.0)
        assert pct["width"] == pytest.approx(50.0)
        assert pct["height"] == pytest.approx(50.0)

    def test_non_square_image(self):
        # image 1920 × 1080, box top-left quarter
        pct = abs_to_ls_percent(0, 0, 960, 540, 1920, 1080)
        assert pct["x"] == pytest.approx(0.0)
        assert pct["y"] == pytest.approx(0.0)
        assert pct["width"] == pytest.approx(50.0)
        assert pct["height"] == pytest.approx(50.0)

    def test_clamping_out_of_bounds(self):
        # bbox slightly outside image due to model error
        pct = abs_to_ls_percent(-5, -5, 105, 205, 100, 200)
        assert pct["x"] == pytest.approx(0.0)
        assert pct["y"] == pytest.approx(0.0)
        assert pct["width"] <= 100.0
        assert pct["height"] <= 100.0

    def test_invalid_image_size_raises(self):
        with pytest.raises(ValueError):
            abs_to_ls_percent(0, 0, 100, 100, 0, 100)

    def test_values_in_percentage_range(self):
        pct = abs_to_ls_percent(100, 200, 300, 400, 640, 480)
        for key in ("x", "y", "width", "height"):
            assert 0.0 <= pct[key] <= 100.0, f"{key} out of range: {pct[key]}"


# ------------------------------------------------------------------ build_ls_result

class TestBuildLsResult:
    def _sample_detections(self):
        return [
            {"label": "car", "bbox": [0, 0, 320, 240], "score": 0.9},
            {"label": "person", "bbox": [320, 240, 640, 480], "score": 0.8},
        ]

    def test_result_count(self):
        dets = self._sample_detections()
        results = build_ls_result(dets, 640, 480)
        assert len(results) == 2

    def test_result_structure(self):
        dets = self._sample_detections()
        result = build_ls_result(dets, 640, 480)[0]
        assert result["type"] == "rectanglelabels"
        assert result["from_name"] == "label"
        assert result["to_name"] == "image"
        assert result["original_width"] == 640
        assert result["original_height"] == 480
        assert "id" in result

    def test_rectanglelabels_value(self):
        dets = [{"label": "car", "bbox": [0, 0, 640, 480], "score": 1.0}]
        result = build_ls_result(dets, 640, 480)[0]
        assert result["value"]["rectanglelabels"] == ["car"]

    def test_percentage_coords_correct(self):
        dets = [{"label": "car", "bbox": [64, 48, 192, 144], "score": 1.0}]
        result = build_ls_result(dets, 640, 480)[0]
        assert result["value"]["x"] == pytest.approx(10.0)
        assert result["value"]["y"] == pytest.approx(10.0)
        assert result["value"]["width"] == pytest.approx(20.0)
        assert result["value"]["height"] == pytest.approx(20.0)

    def test_custom_from_to_names(self):
        dets = [{"label": "bike", "bbox": [0, 0, 100, 100], "score": 0.7}]
        result = build_ls_result(dets, 200, 200, from_name="bbox", to_name="photo")[0]
        assert result["from_name"] == "bbox"
        assert result["to_name"] == "photo"

    def test_empty_detections(self):
        results = build_ls_result([], 640, 480)
        assert results == []

    def test_each_result_has_unique_id(self):
        dets = self._sample_detections()
        results = build_ls_result(dets, 640, 480)
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))
