"""Passthrough engine — opens a child WebSocket to a real Speechmatics container.

Validates the M2 protocol shim byte-for-byte: any difference between our server's
output and a real Speechmatics container's output is a bug in our shim, not in
the engine. Blocked on a fresh Speechmatics license (eval expired 2026-05-10).
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterable

import websockets

from server.engines._base import StreamFinal, StreamingEngine, StreamPartial, StreamWord


class SpeechmaticsEngine(StreamingEngine):
    name = "speechmatics"

    def __init__(self, upstream_url: str | None = None):
        self.upstream_url = upstream_url or os.environ.get(
            "SPEECHMATICS_PROXY_URL", "ws://localhost:9001/v2"
        )

    async def stream(self, pcm_frames, transcription_config):
        async with websockets.connect(self.upstream_url, max_size=None) as upstream:
            await upstream.send(json.dumps({
                "message": "StartRecognition",
                "audio_format": {"type": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
                "transcription_config": transcription_config,
            }))

            out_queue: asyncio.Queue = asyncio.Queue()

            async def pump_audio():
                async for frame in pcm_frames:
                    await upstream.send(frame)
                await upstream.send(json.dumps({"message": "EndOfStream", "last_seq_no": 0}))

            async def pump_events():
                async for raw in upstream:
                    msg = json.loads(raw)
                    kind = msg.get("message")
                    if kind == "AddPartialTranscript":
                        text = msg.get("metadata", {}).get("transcript", "")
                        await out_queue.put(StreamPartial(transcript=text, start_time=0.0, end_time=0.0))
                    elif kind == "AddTranscript":
                        text = msg.get("metadata", {}).get("transcript", "")
                        results = msg.get("results", [])
                        words = [
                            StreamWord(
                                content=(r.get("alternatives") or [{}])[0].get("content", ""),
                                start_time=float(r.get("start_time", 0.0)),
                                end_time=float(r.get("end_time", 0.0)),
                                confidence=(r.get("alternatives") or [{}])[0].get("confidence"),
                                speaker=r.get("speaker"),
                                is_punctuation=r.get("type") == "punctuation",
                            )
                            for r in results
                        ]
                        await out_queue.put(StreamFinal(
                            transcript=text,
                            words=words,
                            start_time=words[0].start_time if words else 0.0,
                            end_time=words[-1].end_time if words else 0.0,
                        ))
                    elif kind == "EndOfTranscript":
                        await out_queue.put(None)
                        return

            asyncio.create_task(pump_audio())
            asyncio.create_task(pump_events())

            while True:
                event = await out_queue.get()
                if event is None:
                    return
                yield event
