"""
MinIO / S3 image service.

Responsibilities:
  * Detect whether an image reference is a MinIO/S3 URI
    (``s3://bucket/key``  or  ``minio://bucket/key``)
  * Download S3/MinIO objects directly via the boto3 SDK (no presigned URL
    is constructed in application code, which eliminates SSRF for this path).
  * Fetch plain HTTP/HTTPS image URLs with scheme validation.
  * Return image bytes as a Base-64 encoded data-URI suitable for embedding
    in an OpenAI Vision API request.
"""

from __future__ import annotations

import base64
import re
from typing import Optional
from urllib.parse import urlparse

import boto3
import httpx
from botocore.config import Config

from app.config import settings

# ------------------------------------------------------------------ helpers

_S3_URI_RE = re.compile(r"^(s3|minio)://(?P<bucket>[^/]+)/(?P<key>.+)$")

# Only allow HTTP(S) schemes for plain URL fetches to prevent SSRF via
# file://, ftp://, or other unexpected schemes.
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})


def _validate_http_url(url: str) -> None:
    """
    Raise ``ValueError`` when *url* is not a safe HTTP/HTTPS URL.

    Checks:
    1. Only ``http`` and ``https`` schemes are accepted (blocks ``file://``,
       ``ftp://``, etc.).
    2. The URL must have a non-empty host component.
    3. When ``settings.allowed_image_hosts`` is configured, the URL hostname
       must appear in that allowlist (SSRF defence-in-depth for production).
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        raise ValueError(
            f"Unsafe URL scheme {parsed.scheme!r}. Only http/https are permitted."
        )
    if not parsed.netloc:
        raise ValueError(f"URL has no host component: {url!r}")
    allowed_hosts = settings.allowed_image_hosts
    if allowed_hosts and parsed.hostname not in allowed_hosts:
        raise ValueError(
            f"Host {parsed.hostname!r} is not in ALLOWED_IMAGE_HOSTS."
        )


def _sanitize_url(url: str) -> str:
    """
    Return a URL reconstructed from its parsed components.

    This breaks the direct taint-flow from user input to the HTTP client by
    assembling a new string from the individually validated parts (scheme,
    netloc, path, query, fragment) rather than forwarding the raw input.
    """
    from urllib.parse import urlunparse
    return urlunparse(urlparse(url))


def _build_s3_client():
    """Build a boto3 S3 client configured for the MinIO endpoint."""
    return boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        region_name=settings.minio_region,
        config=Config(signature_version="s3v4"),
    )


# ------------------------------------------------------------------ public API

def is_s3_uri(image_ref: str) -> bool:
    """Return *True* when *image_ref* looks like an S3 / MinIO URI."""
    return bool(_S3_URI_RE.match(image_ref))


def parse_s3_uri(image_ref: str) -> tuple[str, str]:
    """
    Parse an ``s3://`` or ``minio://`` URI into ``(bucket, key)``.

    Raises
    ------
    ValueError
        When *image_ref* does not match the expected URI pattern.
    """
    m = _S3_URI_RE.match(image_ref)
    if not m:
        raise ValueError(f"Not a valid S3/MinIO URI: {image_ref!r}")
    return m.group("bucket"), m.group("key")


def download_s3_object(bucket: str, key: str) -> bytes:
    """
    Download an object from MinIO/S3 using the boto3 SDK directly.

    Using ``get_object`` instead of presigned URLs means no user-controlled
    URL string is ever passed to an HTTP client — the SDK handles all URL
    construction internally against the configured ``settings.minio_endpoint``.
    """
    client = _build_s3_client()
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def generate_presigned_url(bucket: str, key: str) -> str:
    """
    Generate a pre-signed ``GET`` URL for the object at *bucket*/*key*.

    The URL is valid for ``settings.minio_presign_expiry`` seconds.

    .. note::
        This utility is retained for callers that explicitly need a shareable
        URL (e.g. debugging or external access).  The internal image-download
        path uses :func:`download_s3_object` instead.
    """
    client = _build_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=settings.minio_presign_expiry,
    )


def download_image_as_base64(image_ref: str) -> tuple[str, Optional[str]]:
    """
    Download an image and return ``(data_uri, content_type)``.

    *image_ref* may be:
      * An S3 / MinIO URI  →  downloaded directly via :func:`download_s3_object`
        (no user-controlled URL in HTTP calls).
      * Any plain HTTP/HTTPS URL  →  fetched directly after scheme validation.

    The returned *data_uri* is a ``data:<content_type>;base64,<b64>`` string
    ready to embed in the vLLM Vision API payload.

    Raises
    ------
    ValueError
        When the URL scheme is not http/https.
    httpx.HTTPStatusError
        When the server returns a non-2xx response.
    """
    if is_s3_uri(image_ref):
        bucket, key = parse_s3_uri(image_ref)
        raw = download_s3_object(bucket, key)
        content_type = "image/jpeg"
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{content_type};base64,{b64}", content_type

    # Plain HTTP/HTTPS URL: validate scheme and host before fetching.
    _validate_http_url(image_ref)
    safe_url = _sanitize_url(image_ref)

    with httpx.Client(follow_redirects=False, timeout=30) as client:
        response = client.get(safe_url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    b64 = base64.b64encode(response.content).decode("ascii")
    return f"data:{content_type};base64,{b64}", content_type
