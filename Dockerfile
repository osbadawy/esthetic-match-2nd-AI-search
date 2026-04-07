FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (lxml, bs4, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
# Install CPU-only torch first (~300MB vs ~2GB for full torch+CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code + assets
COPY . /app

# Reasonable defaults (override in DO env vars)
ENV TORCH_NUM_THREADS=2
ENV MIN_ISSUE_CHARS=5

EXPOSE 8080

CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8080}"]
