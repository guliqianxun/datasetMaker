"""
vLLM Vision API client.

Calls the vLLM OpenAI-compatible ``/chat/completions`` endpoint with an
image embedded as a Base-64 data-URI and returns the raw model response text.

The caller is responsible for parsing the structured JSON that the model
returns (see ``label_studio.py``).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ public API


def build_vision_messages(image_data_uri: str, prompt: str) -> list[dict[str, Any]]:
    """
    Build the ``messages`` payload for a single-turn Vision chat request.

    Parameters
    ----------
    image_data_uri:
        A ``data:<mime>;base64,<b64>`` string (or any URL) for the image.
    prompt:
        The text instruction placed *before* the image in the user turn.
    """
    return [
        {
            "role": "system",
            "content": settings.detection_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_uri},
                },
            ],
        },
    ]


def call_vllm(
    image_data_uri: str,
    prompt: str = "请找出图中所有目标，输出它们的类别和边界框。",
    *,
    guided_json: dict[str, Any] | None = None,
) -> str:
    """
    Send a Vision chat-completion request to the vLLM server.

    Parameters
    ----------
    image_data_uri:
        Base-64 data-URI (or HTTP URL) of the image to analyse.
    prompt:
        User-visible instruction appended inside the user turn.
    guided_json:
        Optional JSON Schema passed to vLLM's ``guided_decoding``
        (``extra_body``) to constrain the output to valid JSON.

    Returns
    -------
    str
        The raw text content of the first choice in the model response.

    Raises
    ------
    httpx.HTTPStatusError
        When vLLM returns a non-2xx response.
    ValueError
        When the response cannot be parsed.
    """
    url = f"{settings.vllm_base_url.rstrip('/')}/chat/completions"
    messages = build_vision_messages(image_data_uri, prompt)

    payload: dict[str, Any] = {
        "model": settings.vllm_model,
        "messages": messages,
        "temperature": settings.vllm_temperature,
        "max_tokens": settings.vllm_max_tokens,
    }

    if guided_json is not None:
        # vLLM guided-decoding extension (JSON mode)
        payload["extra_body"] = {
            "guided_decoding_backend": "outlines",
            "guided_json": guided_json,
        }

    logger.debug("Sending request to vLLM: url=%s model=%s", url, settings.vllm_model)

    with httpx.Client(timeout=settings.vllm_timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()

    data = response.json()

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected vLLM response structure: {data}") from exc

    logger.debug("vLLM response text: %s", text)
    return text


# ------------------------------------------------------------------ JSON schema used for guided decoding

DETECTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["label", "bbox", "score"],
        "properties": {
            "label": {"type": "string"},
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
            "score": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "additionalProperties": False,
    },
}
