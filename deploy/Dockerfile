# depodash-asr — M2 WebSocket container
#
# Base image: NVIDIA PyTorch container ships CUDA 12.6 + cuDNN + a tuned
# PyTorch build that matches our box's driver (580+). Pinned by digest in
# the docker-compose to prevent surprise upgrades.
#
# Layout inside the image:
#   /opt/depodash-asr/                 application code
#     engines/                         all engine adapters
#     diarizers/                       all diarizer adapters
#     server/                          FastAPI + WebSocket protocol shim
#     NOTICES                          third-party model attributions
#     requirements.txt                 Python deps (pinned)
#   /opt/hf-cache/                     HuggingFace cache mount point
#
# Model weights are NOT baked into the image — they're downloaded on first
# boot to /opt/hf-cache, which is a docker-compose volume so subsequent
# restarts skip the ~10 GB download. This keeps the image under 8 GB and
# lets us update weights without rebuilding.
FROM nvcr.io/nvidia/pytorch:25.01-py3

# ---------------------------------------------------------------------------
# System packages

# libsndfile1: NeMo audio I/O.
# ffmpeg:      audio resampling for engines that don't accept raw 16 kHz mono.
# curl:        the health check script uses it.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Application

WORKDIR /opt/depodash-asr

# Paths below are relative to the build context (the repo root —
# `..` from this Dockerfile's directory). docker-compose.yml sets that
# explicitly via build.context: ..

# Python dependencies. Install before app code so layer caches well.
COPY deploy/requirements.txt /opt/depodash-asr/requirements.txt
RUN pip install --no-cache-dir -r /opt/depodash-asr/requirements.txt

# App code. Order matters — engines/diarizers first because they import
# heavier ML deps and are slower to revalidate; server/ is small and
# changes more often.
COPY engines /opt/depodash-asr/engines
COPY diarizers /opt/depodash-asr/diarizers
COPY server /opt/depodash-asr/server
COPY tools /opt/depodash-asr/tools
COPY deploy/NOTICES /opt/depodash-asr/NOTICES
COPY deploy/health-check.sh /usr/local/bin/health-check.sh
RUN chmod +x /usr/local/bin/health-check.sh

# ---------------------------------------------------------------------------
# Runtime defaults

# HF cache lives on a volume so model weights persist across restarts.
ENV HF_HOME=/opt/hf-cache \
    HUGGINGFACE_HUB_CACHE=/opt/hf-cache/hub \
    TRANSFORMERS_CACHE=/opt/hf-cache/hub \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO

# Engine / diarizer / punctuator defaults — auto-multispeaker routes
# per-session based on transcription_config.speaker_diarization_config.max_speakers.
# Override any of these via env in docker-compose.yml or `-e` flags for
# single-engine testing.
ENV ENGINE=auto-multispeaker \
    DIARIZER=passthrough \
    PUNCTUATOR=passthrough \
    MAX_CONCURRENT_CONNECTIONS=4

# Cap VRAM per process — used by server/_gpu_cap.py to enforce a soft
# limit. Adjust per deployment shape (8 GB / 16 GB / 24 GB cards).
ENV VRAM_LIMIT_GIB=14

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD /usr/local/bin/health-check.sh

CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "9000", \
     "--log-level", "info"]
