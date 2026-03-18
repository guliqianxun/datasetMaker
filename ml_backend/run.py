#!/usr/bin/env python
"""
Entry-point for the Label Studio ML Backend.

Usage::

    python run.py               # defaults from config.py / .env
    python run.py --port 9090   # override port
"""

from __future__ import annotations

import argparse

import uvicorn

from app.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Label Studio ML Backend (vLLM Vision)")
    parser.add_argument("--host", default=settings.ml_backend_host)
    parser.add_argument("--port", type=int, default=settings.ml_backend_port)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
