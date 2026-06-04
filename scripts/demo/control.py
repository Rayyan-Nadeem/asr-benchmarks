"""Lightweight orchestrator for the live demo.

Owns the depodash-asr uvicorn child process. Exposes a tiny HTTP API on
:9100 that live.html calls to swap the underlying engine + diarizer
combination — kills the running uvicorn, starts a new one with different
ENGINE/DIARIZER env vars, blocks until the new server's /ready endpoint
answers, and reports back.

Why this instead of multiple containers on different ports: only one
engine's model lives in memory at a time. Switching costs ~5–15 s
(engine pre-warm + model load on first use) but doesn't blow up RAM/VRAM.

Endpoints:
    GET  /stacks             list of available stack presets
    GET  /current            which preset is currently loaded
    POST /switch?stack=NAME  swap to the named preset; blocks until ready
"""
from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)  # asr-benchmarks/
SERVER_PORT = 9000
CONTROL_PORT = 9100
SERVER_LOG = "/tmp/m2_server.log"

# Pyannote checkpoint download requires a HF token with model-access perms.
# Sourced from the environment so we don't commit it. Both the local launch
# and the systemd unit set HF_TOKEN externally.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

ENGINES: dict[str, dict] = {
    "auto-multispeaker": {
        "label": "Auto  ·  Multitalker for ≤4 spks, fallback for >4",
        "platform": "gpu-only",
    },
    "multitalker-parakeet": {
        "label": "Multitalker Parakeet  ·  true overlap, parallel speakers (≤4 spks)",
        "platform": "gpu-only",
    },
    "nemotron-native": {
        "label": "Nemotron Native  ·  pipeline-style, arbitrary spks, 560 ms",
        "platform": "gpu-only",
    },
    "parakeet-nemo": {
        "label": "Parakeet TDT 0.6B  ·  best WER, end-of-segment finals",
        "platform": "gpu-only",
    },
    "fastconformer-hybrid": {
        "label": "FastConformer Hybrid 114M  ·  8 GB GPU / CPU",
        "platform": "all",
    },
    "nemotron-nemo": {
        "label": "Nemotron 0.6B chunked-offline",
        "platform": "gpu-only",
    },
    "parakeet-onnx": {
        "label": "Parakeet TDT INT8  ·  CPU fallback",
        "platform": "all",
    },
    "nemotron-streaming": {
        "label": "Nemotron sherpa-INT8  ·  CPU only",
        "platform": "all",
    },
    "whisper": {
        "label": "Whisper Large v3  ·  short clips only",
        "platform": "all",
    },
    "noop": {
        "label": "Noop  ·  protocol test",
        "platform": "all",
    },
}

DIARIZERS: dict[str, dict] = {
    "sortformer": {
        "label": "Sortformer 4-spk streaming v2.1  ·  GPU",
        "platform": "gpu-only",
    },
    "pyannote": {
        "label": "pyannote.audio 3.1  ·  best on AMI overlap",
        "platform": "all",
    },
    "passthrough": {
        "label": "— No diarization (use with Multitalker)",
        "platform": "all",
    },
}

PUNCTUATORS: dict[str, dict] = {
    "distilbert": {
        "label": "DistilBERT post-process  ·  adds periods, commas, ? marks + capitalization",
        "platform": "all",
    },
    "passthrough": {
        "label": "— Raw (engine output only, no post-process)",
        "platform": "all",
    },
}

DEFAULT_ENGINE = "auto-multispeaker"
DEFAULT_DIARIZER = "passthrough"
DEFAULT_PUNCTUATOR = "passthrough"

_child: subprocess.Popen | None = None
_current_engine: str | None = None
_current_diarizer: str | None = None
_current_punctuator: str | None = None


def _stop_child() -> None:
    """Kill the running uvicorn (anything bound to SERVER_PORT)."""
    global _child
    if _child and _child.poll() is None:
        _child.terminate()
        try:
            _child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _child.kill()
            _child.wait()
    # Belt-and-suspenders: also pkill in case a previous run left an orphan.
    subprocess.run(["pkill", "-f", "uvicorn server.app"], check=False)
    time.sleep(0.5)
    _child = None


def _start_combo(engine: str, diarizer: str, punctuator: str) -> None:
    """Spawn a new uvicorn with the given ENGINE / DIARIZER / PUNCTUATOR env vars."""
    global _child, _current_engine, _current_diarizer, _current_punctuator
    env = os.environ.copy()
    env["ENGINE"] = engine
    env["DIARIZER"] = diarizer
    env["PUNCTUATOR"] = punctuator
    env["HF_TOKEN"] = HF_TOKEN
    env["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
    # Append so we keep failure traces from prior stacks (overwrite was hiding bugs)
    log = open(SERVER_LOG, "ab")
    _child = subprocess.Popen(
        [
            "python3", "-m", "uvicorn", "server.app:app",
            "--host", "127.0.0.1",
            "--port", str(SERVER_PORT),
            "--log-level", "info",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    _current_engine = engine
    _current_diarizer = diarizer
    _current_punctuator = punctuator


def _wait_ready(timeout: float = 90.0) -> bool:
    """Poll /ready until the server answers, or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{SERVER_PORT}/ready", timeout=2
            ) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------

_DEMO_DIR = str(Path(__file__).resolve().parent)

app = Flask(__name__)
# Preserve dict insertion order in JSON responses — the engine/diarizer
# dropdowns surface in the order declared in ENGINES/DIARIZERS, so
# alphabetical sorting hides our intentional best-to-worst ranking.
app.json.sort_keys = False
CORS(app)


def _nocache(resp):
    """Strip all caching so users always get the freshest live.html."""
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/")
def index():
    return _nocache(send_from_directory(_DEMO_DIR, "live.html"))


@app.route("/<path:path>")
def static_files(path):
    # Serve anything else (css, images) from the demo dir; falls back to 404.
    return _nocache(send_from_directory(_DEMO_DIR, path))


@app.route("/engines")
def engines():
    return _nocache(jsonify(ENGINES))


@app.route("/diarizers")
def diarizers():
    return _nocache(jsonify(DIARIZERS))


@app.route("/punctuators")
def punctuators():
    return _nocache(jsonify(PUNCTUATORS))


@app.route("/current")
def current():
    return _nocache(jsonify({
        "engine": _current_engine,
        "diarizer": _current_diarizer,
        "punctuator": _current_punctuator,
        "ready": _child is not None and _child.poll() is None,
    }))


@app.route("/switch", methods=["POST"])
def switch():
    args = request.args
    payload = request.get_json(silent=True) or {}
    engine = args.get("engine") or payload.get("engine")
    diarizer = args.get("diarizer") or payload.get("diarizer")
    punctuator = (args.get("punctuator") or payload.get("punctuator")
                  or _current_punctuator or DEFAULT_PUNCTUATOR)
    for label, value, registry in (
        ("engine", engine, ENGINES),
        ("diarizer", diarizer, DIARIZERS),
        ("punctuator", punctuator, PUNCTUATORS),
    ):
        if value not in registry:
            return jsonify({
                "error": f"unknown {label}: {value!r}",
                "available": list(registry),
            }), 400

    if (engine == _current_engine and diarizer == _current_diarizer
            and punctuator == _current_punctuator
            and _child is not None and _child.poll() is None):
        return jsonify({
            "ok": True,
            "engine": _current_engine,
            "diarizer": _current_diarizer,
            "punctuator": _current_punctuator,
            "note": "no-op",
        })

    _stop_child()
    _start_combo(engine, diarizer, punctuator)
    if _wait_ready():
        return jsonify({
            "ok": True,
            "engine": _current_engine,
            "diarizer": _current_diarizer,
            "punctuator": _current_punctuator,
        })
    return jsonify({
        "error": "server failed to become ready",
        "log_tail": _read_log_tail(),
    }), 500


def _read_log_tail(lines: int = 20) -> str:
    try:
        with open(SERVER_LOG) as f:
            return "".join(f.readlines()[-lines:])
    except OSError:
        return ""


def main() -> None:
    _stop_child()
    _start_combo(DEFAULT_ENGINE, DEFAULT_DIARIZER, DEFAULT_PUNCTUATOR)
    print(
        f"control: booting {DEFAULT_ENGINE} + {DEFAULT_DIARIZER} + "
        f"{DEFAULT_PUNCTUATOR}, waiting for /ready…",
        flush=True,
    )
    if _wait_ready():
        print(f"control: depodash-asr ready on :{SERVER_PORT}", flush=True)
    else:
        print(f"control: WARNING — default combo didn't come up, see {SERVER_LOG}", flush=True)
    print(f"control: orchestrator listening on :{CONTROL_PORT}", flush=True)
    app.run(host="127.0.0.1", port=CONTROL_PORT, threaded=True)


if __name__ == "__main__":
    main()
