"""
Unit tests for app/services/minio_service.py

All external I/O (boto3, httpx) is mocked so no real MinIO or HTTP server is
needed.
"""

from __future__ import annotations

import base64

import pytest

from app.services.minio_service import (
    _sanitize_url,
    _validate_http_url,
    download_image_as_base64,
    download_s3_object,
    generate_presigned_url,
    is_s3_uri,
    parse_s3_uri,
)


class TestValidateHttpUrl:
    @pytest.mark.parametrize("url", [
        "http://example.com/image.jpg",
        "https://minio.internal:9000/bucket/key?sig=abc",
    ])
    def test_valid_urls_pass(self, url):
        _validate_http_url(url)  # should not raise

    @pytest.mark.parametrize("url", [
        "file:///etc/passwd",
        "ftp://ftp.example.com/image.jpg",
        "data:image/jpeg;base64,abc",
        "/local/path/image.jpg",
    ])
    def test_unsafe_urls_raise(self, url):
        with pytest.raises(ValueError):
            _validate_http_url(url)

    def test_url_without_host_raises(self):
        with pytest.raises(ValueError):
            _validate_http_url("http:///no-host/path")

    def test_allowed_hosts_enforced_when_configured(self, mocker):
        mocker.patch(
            "app.services.minio_service.settings",
            allowed_image_hosts=["trusted.example.com"],
        )
        with pytest.raises(ValueError, match="not in ALLOWED_IMAGE_HOSTS"):
            _validate_http_url("http://evil.internal/image.jpg")

    def test_allowed_hosts_passes_when_in_list(self, mocker):
        mocker.patch(
            "app.services.minio_service.settings",
            allowed_image_hosts=["trusted.example.com"],
        )
        _validate_http_url("http://trusted.example.com/image.jpg")  # should not raise


class TestSanitizeUrl:
    def test_preserves_valid_url(self):
        url = "https://example.com/path/img.jpg?v=1"
        assert _sanitize_url(url) == url

    def test_reconstructs_from_components(self):
        url = "http://minio:9000/bucket/key.jpg"
        result = _sanitize_url(url)
        assert result == url


# ------------------------------------------------------------------ is_s3_uri

class TestIsS3Uri:
    @pytest.mark.parametrize("uri", [
        "s3://my-bucket/path/to/image.jpg",
        "minio://my-bucket/images/photo.png",
        "s3://bucket/nested/deep/key.jpeg",
    ])
    def test_valid_uris(self, uri):
        assert is_s3_uri(uri) is True

    @pytest.mark.parametrize("uri", [
        "http://example.com/image.jpg",
        "https://example.com/img.png",
        "/local/path/image.jpg",
        "",
        "ftp://bucket/key",
    ])
    def test_invalid_uris(self, uri):
        assert is_s3_uri(uri) is False


# ------------------------------------------------------------------ parse_s3_uri

class TestParseS3Uri:
    def test_s3_scheme(self):
        bucket, key = parse_s3_uri("s3://my-bucket/images/test.jpg")
        assert bucket == "my-bucket"
        assert key == "images/test.jpg"

    def test_minio_scheme(self):
        bucket, key = parse_s3_uri("minio://data-bucket/raw/photo.png")
        assert bucket == "data-bucket"
        assert key == "raw/photo.png"

    def test_deep_key(self):
        bucket, key = parse_s3_uri("s3://bucket/a/b/c/d.jpg")
        assert bucket == "bucket"
        assert key == "a/b/c/d.jpg"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_s3_uri("http://not-an-s3-uri/image.jpg")


# ------------------------------------------------------------------ generate_presigned_url

class TestGeneratePresignedUrl:
    def test_returns_url(self, mocker):
        mock_client = mocker.MagicMock()
        mock_client.generate_presigned_url.return_value = "https://minio/bucket/key?X-Amz-Signature=abc"
        mocker.patch("app.services.minio_service._build_s3_client", return_value=mock_client)

        url = generate_presigned_url("bucket", "key/image.jpg")
        assert url.startswith("https://")
        mock_client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "bucket", "Key": "key/image.jpg"},
            ExpiresIn=mocker.ANY,
        )


# ------------------------------------------------------------------ download_s3_object

class TestDownloadS3Object:
    def test_calls_boto3_get_object(self, mocker):
        mock_body = mocker.MagicMock()
        mock_body.read.return_value = b"fake-image-bytes"
        mock_client = mocker.MagicMock()
        mock_client.get_object.return_value = {"Body": mock_body}
        mocker.patch("app.services.minio_service._build_s3_client", return_value=mock_client)

        result = download_s3_object("my-bucket", "path/image.jpg")

        mock_client.get_object.assert_called_once_with(Bucket="my-bucket", Key="path/image.jpg")
        assert result == b"fake-image-bytes"


# ------------------------------------------------------------------ download_image_as_base64

class TestDownloadImageAsBase64:
    def _make_tiny_png(self) -> bytes:
        """Return a 1×1 red PNG in raw bytes."""
        import io
        from PIL import Image

        buf = io.BytesIO()
        img = Image.new("RGB", (1, 1), color=(255, 0, 0))
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_http_url_downloaded_and_encoded(self, mocker):
        png_bytes = self._make_tiny_png()
        mock_response = mocker.MagicMock()
        mock_response.content = png_bytes
        mock_response.headers = {"content-type": "image/png"}
        mock_response.raise_for_status = mocker.MagicMock()

        mock_client = mocker.MagicMock()
        mock_client.__enter__ = mocker.MagicMock(return_value=mock_client)
        mock_client.__exit__ = mocker.MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mocker.patch("app.services.minio_service.httpx.Client", return_value=mock_client)

        data_uri, content_type = download_image_as_base64("http://example.com/img.png")

        assert content_type == "image/png"
        assert data_uri.startswith("data:image/png;base64,")
        # Verify round-trip integrity
        b64_part = data_uri.split(",", 1)[1]
        assert base64.b64decode(b64_part) == png_bytes

    def test_s3_uri_calls_download_s3_object(self, mocker):
        png_bytes = self._make_tiny_png()

        mocker.patch(
            "app.services.minio_service.download_s3_object",
            return_value=png_bytes,
        )

        data_uri, content_type = download_image_as_base64("s3://bucket/key/image.jpg")

        # Should NOT make any external HTTP call – data comes directly from boto3
        assert data_uri.startswith("data:image/jpeg;base64,")
