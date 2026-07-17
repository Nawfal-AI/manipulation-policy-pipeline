# Multi-stage build: keep the final image lean by not shipping build tools.
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --target=/build/deps -r requirements-serve.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages
COPY models.py serve.py ./
# Deliberately NOT copying checkpoints/ into the image: model weights are
# mounted as a volume at runtime (see docker-compose.yml / k8s), so a new
# training run doesn't require rebuilding and redeploying the image.

ENV CHECKPOINT_PATH=/app/checkpoints/bc_policy.pt
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
