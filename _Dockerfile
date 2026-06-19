FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache \
    TORCH_HOME=/app/torch_cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt requirements/base.txt
COPY requirements/gpu.txt requirements/gpu.txt

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        -r requirements/base.txt \
        -r requirements/gpu.txt

COPY app app
COPY run.py .

EXPOSE 8000

CMD ["python3", "run.py"]
