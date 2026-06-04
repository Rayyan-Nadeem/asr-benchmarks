"""FastAPI app — exposes ws://host:9000/v2 in the Speechmatics protocol."""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from server.session import Session


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


app = FastAPI(title="depodash-asr", docs_url=None, redoc_url=None)

# Open CORS so the browser demo (served from any localhost port) can read
# /ready. WebSocket /v2 isn't subject to CORS, so it worked all along.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _prewarm() -> None:
    """Cap VRAM to 8 GB if CUDA available, then pre-warm the engine."""
    log = logging.getLogger("server.prewarm")
    try:
        from server._gpu_cap import apply as cap_gpu
        info = cap_gpu()
        log.info("vram cap: %s", info)
    except Exception as e:  # noqa: BLE001
        log.warning("vram cap failed: %r", e)

    try:
        from server.engine_registry import load_engine
        engine = load_engine()
        if hasattr(engine, "warm"):
            import time
            t0 = time.monotonic()
            engine.warm()
            log.info("pre-warmed %s in %.1fs", engine.name, time.monotonic() - t0)
        else:
            log.info("engine %s has no warm() — skipping pre-warm", engine.name)
    except Exception as e:  # noqa: BLE001
        log.warning("pre-warm failed: %r (server still starts)", e)


@app.get("/ready")
async def ready():
    return {
        "ready": True,
        "engine": os.environ.get("ENGINE", "noop"),
        "diarizer": os.environ.get("DIARIZER", "passthrough"),
    }


@app.websocket("/v2")
async def v2(ws: WebSocket) -> None:
    await ws.accept()
    session = Session(ws)
    await session.run()
