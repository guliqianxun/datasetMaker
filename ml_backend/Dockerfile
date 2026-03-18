# ---- builder stage: install dependencies ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- runtime stage ----
FROM python:3.11-slim

LABEL org.opencontainers.image.title="label-studio-ml-backend"
LABEL org.opencontainers.image.description="Label Studio ML Backend powered by vLLM Vision"

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY run.py .

USER appuser

EXPOSE 9090

CMD ["python", "run.py"]
