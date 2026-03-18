"""
Centralised configuration loaded from environment variables.

All settings have safe defaults so the service starts without additional
configuration, but everything can be overridden via environment variables or
a .env file (loaded automatically by Pydantic Settings).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ vLLM
    vllm_base_url: str = "http://localhost:8000/v1"
    """Base URL of the vLLM OpenAI-compatible API server."""

    vllm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    """Model identifier passed to the vLLM /chat/completions endpoint."""

    vllm_timeout: int = 120
    """HTTP timeout in seconds when calling vLLM."""

    vllm_temperature: float = 0.0
    """Sampling temperature (0 = deterministic)."""

    vllm_max_tokens: int = 1024
    """Maximum number of tokens in the model response."""

    # --------------------------------------------------------------- MinIO / S3
    minio_endpoint: str = "http://localhost:9000"
    """MinIO (or any S3-compatible) endpoint URL."""

    minio_access_key: str = "minioadmin"
    """S3 access key / MinIO root user."""

    minio_secret_key: str = "minioadmin"
    """S3 secret key / MinIO root password."""

    minio_region: str = "us-east-1"
    """AWS / MinIO region name."""

    minio_presign_expiry: int = 3600
    """Lifetime of generated presigned URLs in seconds."""

    # ------------------------------------------------------- Label Studio / ML Backend
    ls_api_token: str = ""
    """Label Studio API token (used when the backend needs to call LS APIs)."""

    ml_backend_host: str = "0.0.0.0"
    """Host the FastAPI service should bind to."""

    ml_backend_port: int = 9090
    """Port the FastAPI service should listen on."""

    model_version: str = "1.0.0"
    """Semantic version string reported in /setup responses."""

    # ----------------------------------------------------------------- Prompts
    detection_prompt: str = (
        "You are an object detection assistant. "
        "Find all objects in the image and return a JSON array. "
        "Each element must have these keys: "
        '"label" (string class name), '
        '"bbox" ([xmin, ymin, xmax, ymax] in absolute pixel coordinates), '
        '"score" (confidence float 0-1). '
        "Output ONLY the JSON array, no other text."
    )
    """System prompt injected into every vLLM request."""

    allowed_image_hosts: list[str] = []
    """
    Optional allowlist of hostnames from which images may be fetched.

    When non-empty, any plain HTTP/HTTPS image URL whose hostname is not in
    this list will be rejected, providing a defence-in-depth layer against
    SSRF in production deployments.  Example value::

        ALLOWED_IMAGE_HOSTS=label-studio,minio,cdn.example.com

    Leave empty to allow any HTTP/HTTPS host (suitable for development only).
    """

    debug: bool = False
    """Enable debug logging and FastAPI debug mode."""


settings = Settings()
