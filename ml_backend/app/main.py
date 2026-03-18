"""
Label Studio ML Backend – FastAPI application factory.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)

app = FastAPI(
    title="Label Studio ML Backend (vLLM Vision)",
    description=(
        "AI-powered pre-annotation backend that bridges Label Studio with a "
        "vLLM-served vision language model (e.g. Qwen2.5-VL-7B-Instruct).  "
        "Supports MinIO/S3 image storage and returns bounding-box predictions "
        "in Label Studio's native percentage-coordinate format."
    ),
    version=settings.model_version,
    debug=settings.debug,
)

app.include_router(router)
