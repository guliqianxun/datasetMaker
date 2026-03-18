# Label Studio ML Backend (vLLM Vision)

AI-powered pre-annotation backend that bridges **Label Studio** with a
**vLLM-served vision language model** (e.g. Qwen2.5-VL-7B-Instruct).

## Architecture

```
[ Label Studio ]
      │  POST /predict  (task + image reference)
      ▼
[ ML Backend  ]  ← this service
      │  1. Resolve MinIO/S3 URI → presigned URL → download image bytes
      │  2. Encode image as Base64 data-URI
      │  3. Call vLLM /chat/completions (OpenAI Vision format)
      │  4. Parse JSON detections → convert absolute px → LS % coordinates
      │  5. Return Label Studio `result` array
      ▼
[ vLLM API Server ]   (port 8000, OpenAI-compatible)
```

## API Endpoints

| Method | Path       | Description                          |
|--------|------------|--------------------------------------|
| GET    | /health    | Liveness probe → `{"status": "UP"}` |
| POST   | /setup     | LS connects; returns model version   |
| POST   | /predict   | Generate bounding-box predictions    |

### `/predict` – request

```json
{
  "tasks": [
    {
      "id": 1,
      "data": { "image": "s3://my-bucket/images/photo.jpg" }
    }
  ],
  "label_config": "<View><RectangleLabels name=\"label\" toName=\"image\">...</RectangleLabels><Image name=\"image\" value=\"$image\"/></View>",
  "params": {}
}
```

### `/predict` – response

```json
{
  "results": [
    {
      "result": [
        {
          "id": "a1b2c3...",
          "type": "rectanglelabels",
          "from_name": "label",
          "to_name": "image",
          "original_width": 1920,
          "original_height": 1080,
          "image_rotation": 0,
          "value": {
            "x": 10.5,
            "y": 20.3,
            "width": 15.2,
            "height": 25.1,
            "rotation": 0,
            "rectanglelabels": ["car"]
          },
          "score": 0.95
        }
      ],
      "score": 0.95
    }
  ]
}
```

## Coordinate Conversion

The model returns bounding boxes in **absolute pixel coordinates**
`[xmin, ymin, xmax, ymax]`.  Label Studio requires **percentage coordinates**
relative to the image dimensions:

```
x      = (xmin / image_width)  × 100
y      = (ymin / image_height) × 100
width  = ((xmax - xmin) / image_width)  × 100
height = ((ymax - ymin) / image_height) × 100
```

## Configuration

All settings are read from environment variables (or a `.env` file).

| Variable              | Default                              | Description                          |
|-----------------------|--------------------------------------|--------------------------------------|
| `VLLM_BASE_URL`       | `http://localhost:8000/v1`           | vLLM API base URL                    |
| `VLLM_MODEL`          | `Qwen/Qwen2.5-VL-7B-Instruct`        | Model identifier                     |
| `VLLM_TIMEOUT`        | `120`                                | HTTP timeout (seconds)               |
| `VLLM_TEMPERATURE`    | `0.0`                                | Sampling temperature                 |
| `VLLM_MAX_TOKENS`     | `1024`                               | Max tokens in model response         |
| `MINIO_ENDPOINT`      | `http://localhost:9000`              | MinIO/S3 endpoint URL                |
| `MINIO_ACCESS_KEY`    | `minioadmin`                         | S3 access key                        |
| `MINIO_SECRET_KEY`    | `minioadmin`                         | S3 secret key                        |
| `MINIO_REGION`        | `us-east-1`                          | S3 region                            |
| `MINIO_PRESIGN_EXPIRY`| `3600`                               | Presigned URL lifetime (seconds)     |
| `ML_BACKEND_PORT`     | `9090`                               | Port the backend listens on          |
| `DEBUG`               | `false`                              | Enable debug logging                 |
| `DETECTION_PROMPT`    | *(system prompt)*                    | System prompt sent to the model      |

## Quick Start

### 1. Local development

```bash
cd ml_backend
pip install -r requirements.txt
python run.py --reload
```

The service starts on `http://0.0.0.0:9090`.

### 2. Docker Compose (full stack)

```bash
# From the repo root:
cp .env.example .env   # fill in HUGGING_FACE_HUB_TOKEN if needed
docker compose up --build
```

Services:
- Label Studio UI → http://localhost:8080
- ML Backend      → http://localhost:9090
- MinIO Console   → http://localhost:9001  (minioadmin / minioadmin)
- vLLM API        → http://localhost:8000

> **Note:** The vLLM service requires an NVIDIA GPU and the
> [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
> Comment out the `vllm` service in `docker-compose.yml` for CPU-only development.

### 3. Register in Label Studio

1. Open Label Studio → **Settings → Machine Learning**.
2. Click **Add Model**.
3. Enter URL: `http://ml-backend:9090` (or `http://localhost:9090` for local).
4. Click **Validate and Save**.
5. Open any labelling task and click **Auto-Annotate** or **Get Predictions**.

## Running Tests

```bash
cd ml_backend
python -m pytest tests/ -v
```

## Project Layout

```
ml_backend/
├── app/
│   ├── config.py                  # Pydantic-Settings configuration
│   ├── main.py                    # FastAPI application factory
│   ├── api/
│   │   └── routes.py              # /health, /setup, /predict endpoints
│   └── services/
│       ├── minio_service.py       # MinIO/S3 image download → Base64
│       ├── vllm_service.py        # vLLM Vision API client
│       └── label_studio.py        # Result parsing & coordinate conversion
├── tests/
│   ├── test_label_studio.py
│   ├── test_minio_service.py
│   ├── test_routes.py
│   └── test_vllm_service.py
├── Dockerfile
├── requirements.txt
├── run.py
└── pyproject.toml
```
