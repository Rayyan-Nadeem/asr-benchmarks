# `server/` — M2 Speechmatics-protocol WebSocket server

FastAPI + websockets server that speaks the Speechmatics realtime protocol verbatim, with a pluggable engine + diarizer behind it. DepoDash's middleware points at `ws://localhost:9000/v2` exactly like the on-prem Speechmatics container; only the `ENGINE`/`DIARIZER` env vars change what's actually transcribing.

## Run locally

```bash
pip install -r server/requirements.txt
ENGINE=noop DIARIZER=passthrough uvicorn server.app:app --host 0.0.0.0 --port 9000
```

Then in another terminal, point any Speechmatics client at it:

```bash
python tools/measure_via_ws.py \
    --url ws://localhost:9000/v2 \
    --case librispeech-test-clean-mini \
    --engine noop --diarizer passthrough --tag smoke
```

The `noop` engine echoes back stub partials/finals — useful for validating the protocol shim without standing up a real engine.

## Container

```bash
docker build -t depodash-asr:dev -f server/Dockerfile .
docker run -p 9000:9000 -e ENGINE=noop depodash-asr:dev
```

## Layout

```
server/
├── app.py                  FastAPI app + /v2 endpoint
├── protocol.py             Speechmatics message dataclasses
├── session.py              Per-connection state machine
├── engine_registry.py      Loads engine by ENGINE env var
├── diarizer_registry.py    Loads diarizer by DIARIZER env var
├── vad.py                  Silero VAD wrapper (Phase D)
├── engines/
│   ├── _base.py            StreamingEngine protocol
│   ├── noop.py             Echo stub for protocol testing
│   └── speechmatics_proxy.py  Passthrough to a real SM container
├── diarizers/
│   ├── _base.py
│   └── passthrough.py      Single-speaker no-op
├── Dockerfile
└── requirements.txt
```

## Status

- Phase B: protocol shim, session, noop engine — landed.
- Phase C: `parakeet_nemo`, `parakeet_onnx`, `whisper` engines plugged in.
- Phase D: `pyannote_streaming`, `sortformer_streaming` diarizers + `segment_first` integration.
- Phase E: re-measure every combo through this server at 1× realtime.

See the top-level `README.md` for project status.
