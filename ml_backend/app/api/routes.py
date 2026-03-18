"""
Label Studio ML Backend API routes.

Implements the three endpoints required by the Label Studio ML Backend
specification:

  GET  /health   – liveness probe
  POST /setup    – called once when LS connects to the backend
  POST /predict  – called per-task when LS requests predictions
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from PIL import Image
import httpx
import io

from app.config import settings
from app.services.label_studio import build_ls_result, parse_vllm_detections
from app.services.minio_service import download_image_as_base64
from app.services.vllm_service import DETECTION_JSON_SCHEMA, call_vllm

logger = logging.getLogger(__name__)
router = APIRouter()


# ------------------------------------------------------------------ helpers

def _extract_label_config_names(label_config: str | None) -> tuple[str, str]:
    """
    Extract ``from_name`` and ``to_name`` from an LS label configuration XML.

    Falls back to ``("label", "image")`` when the config is absent or
    unparseable.
    """
    if not label_config:
        return "label", "image"

    import re

    from_name_match = re.search(r'<RectangleLabels[^>]+name=["\']([^"\']+)["\']', label_config)
    to_name_match = re.search(r'<Image[^>]+name=["\']([^"\']+)["\']', label_config)

    from_name = from_name_match.group(1) if from_name_match else "label"
    to_name = to_name_match.group(1) if to_name_match else "image"
    return from_name, to_name


# ------------------------------------------------------------------ endpoints

@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe – always returns ``{"status": "UP"}``."""
    return {"status": "UP"}


@router.post("/setup")
def setup(body: dict[str, Any] | None = None) -> dict[str, str]:
    """
    Called by Label Studio when it first connects to the ML Backend.

    Returns the model version so the LS UI can display it.
    """
    return {"model_version": settings.model_version}


@router.post("/predict")
def predict(body: dict[str, Any]) -> dict[str, Any]:
    """
    Generate bounding-box predictions for a batch of Label Studio tasks.

    Expected request body (Label Studio ML Backend format)::

        {
          "tasks": [
            {
              "id": 1,
              "data": {"image": "s3://bucket/path/image.jpg"}
            }
          ],
          "label_config": "<View>...</View>",
          "params": {}
        }

    Returns::

        {
          "results": [
            {
              "result": [...],   // list of rectanglelabels annotations
              "score": 0.95
            }
          ]
        }
    """
    tasks: list[dict[str, Any]] = body.get("tasks", [])
    label_config: str | None = body.get("label_config")
    from_name, to_name = _extract_label_config_names(label_config)

    results = []
    for task in tasks:
        task_data: dict[str, Any] = task.get("data", {})
        # Prefer "image" key; fall back to first value that looks like a URL/URI
        image_ref: str | None = task_data.get("image") or _find_image_ref(task_data)

        if not image_ref:
            logger.warning("Task %s has no image reference, skipping", task.get("id"))
            results.append({"result": [], "score": 0.0})
            continue

        try:
            data_uri, _ = download_image_as_base64(image_ref)
            width, height = _get_image_dimensions_from_data_uri(data_uri)
            raw_text = call_vllm(
                data_uri,
                guided_json=DETECTION_JSON_SCHEMA,
            )
            detections = parse_vllm_detections(raw_text)
            ls_results = build_ls_result(
                detections,
                image_width=width,
                image_height=height,
                from_name=from_name,
                to_name=to_name,
            )
            avg_score = (
                sum(d["score"] for d in detections) / len(detections)
                if detections
                else 0.0
            )
            results.append({"result": ls_results, "score": avg_score})

        except Exception as exc:
            logger.exception("Error processing task %s: %s", task.get("id"), exc)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to process task {task.get('id')}: {exc}",
            ) from exc

    return {"results": results}


# ------------------------------------------------------------------ private helpers

def _find_image_ref(task_data: dict[str, Any]) -> str | None:
    """Return the first string value in *task_data* that looks like an image ref."""
    for value in task_data.values():
        if isinstance(value, str) and (
            value.startswith(("http://", "https://", "s3://", "minio://"))
        ):
            return value
    return None


def _get_image_dimensions_from_data_uri(data_uri: str) -> tuple[int, int]:
    """
    Decode a Base-64 data-URI and return ``(width, height)`` via Pillow.

    This avoids a second network download inside the predict handler.
    """
    import base64
    import re

    b64_data = re.split(r",", data_uri, maxsplit=1)[1]
    raw = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(raw))
    return img.size  # (width, height)
