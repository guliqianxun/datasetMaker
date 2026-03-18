"""
Label Studio result format utilities.

Responsibilities
----------------
1. Parse the JSON array returned by the VLM (list of
   ``{"label": str, "bbox": [xmin, ymin, xmax, ymax], "score": float}``).
2. Convert absolute pixel coordinates to Label Studio's percentage-based
   coordinate system.
3. Build the ``result`` list expected by the Label Studio ML Backend API.

Coordinate conversion
---------------------
Label Studio requires coordinates **relative to the image size**, expressed
as percentages (0–100):

    x      = (xmin / image_width)  * 100   # left edge
    y      = (ymin / image_height) * 100   # top edge
    width  = ((xmax - xmin) / image_width)  * 100
    height = ((ymax - ymin) / image_height) * 100
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ parsing


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """
    Extract the first JSON array from *text*.

    The model sometimes wraps its output in markdown code fences or adds
    prose before/after the JSON.  This function strips those artefacts.

    Raises
    ------
    ValueError
        When no valid JSON array can be found.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    # Try direct parse first
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to find the first [...] block
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract a JSON array from model output: {text!r}")


def parse_vllm_detections(text: str) -> list[dict[str, Any]]:
    """
    Parse the raw text returned by the VLM into a list of detection dicts.

    Each detection dict is expected to have:
      * ``label``  – class name (str)
      * ``bbox``   – [xmin, ymin, xmax, ymax] in absolute pixels
      * ``score``  – confidence (float, 0–1)

    Unknown or malformed detections are skipped with a warning.
    """
    raw = _extract_json_array(text)
    detections: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict detection item: %r", item)
            continue
        label = item.get("label")
        bbox = item.get("bbox")
        score = item.get("score", 1.0)
        if not label or not isinstance(bbox, list) or len(bbox) != 4:
            logger.warning("Skipping malformed detection: %r", item)
            continue
        try:
            xmin, ymin, xmax, ymax = (float(v) for v in bbox)
        except (TypeError, ValueError):
            logger.warning("Skipping detection with non-numeric bbox: %r", item)
            continue
        detections.append(
            {
                "label": str(label),
                "bbox": [xmin, ymin, xmax, ymax],
                "score": float(score),
            }
        )
    return detections


# ------------------------------------------------------------------ coordinate conversion


def abs_to_ls_percent(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    """
    Convert absolute pixel bounding-box to Label Studio percentage coordinates.

    Parameters
    ----------
    xmin, ymin, xmax, ymax:
        Absolute pixel coordinates (origin = top-left corner).
    image_width, image_height:
        Dimensions of the source image in pixels.

    Returns
    -------
    dict with keys ``x``, ``y``, ``width``, ``height`` (all in 0–100 range).
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"image_width and image_height must be positive; got {image_width}×{image_height}"
        )

    x = (xmin / image_width) * 100.0
    y = (ymin / image_height) * 100.0
    width = ((xmax - xmin) / image_width) * 100.0
    height = ((ymax - ymin) / image_height) * 100.0

    # Clamp to [0, 100] to handle minor floating-point drift or model errors
    x = max(0.0, min(100.0, x))
    y = max(0.0, min(100.0, y))
    width = max(0.0, min(100.0 - x, width))
    height = max(0.0, min(100.0 - y, height))

    return {"x": x, "y": y, "width": width, "height": height}


# ------------------------------------------------------------------ Label Studio result builder


def build_ls_result(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    from_name: str = "label",
    to_name: str = "image",
) -> list[dict[str, Any]]:
    """
    Build the Label Studio ``result`` list from a list of detections.

    Parameters
    ----------
    detections:
        Output of :func:`parse_vllm_detections`.
    image_width, image_height:
        Dimensions of the source image in pixels.
    from_name:
        The ``name`` attribute of the ``<RectangleLabels>`` tag in the LS
        label configuration.
    to_name:
        The ``name`` attribute of the ``<Image>`` tag in the LS label
        configuration.

    Returns
    -------
    A list of Label Studio annotation result objects, one per detection.
    """
    results = []
    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        pct = abs_to_ls_percent(xmin, ymin, xmax, ymax, image_width, image_height)
        results.append(
            {
                "id": uuid.uuid4().hex,
                "type": "rectanglelabels",
                "from_name": from_name,
                "to_name": to_name,
                "original_width": image_width,
                "original_height": image_height,
                "image_rotation": 0,
                "value": {
                    "x": pct["x"],
                    "y": pct["y"],
                    "width": pct["width"],
                    "height": pct["height"],
                    "rotation": 0,
                    "rectanglelabels": [det["label"]],
                },
                "score": det["score"],
            }
        )
    return results
