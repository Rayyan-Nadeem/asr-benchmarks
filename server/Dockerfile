FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install -r /app/server/requirements.txt

COPY server /app/server

ENV ENGINE=noop \
    DIARIZER=passthrough \
    LOG_LEVEL=INFO

EXPOSE 9000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "9000"]
